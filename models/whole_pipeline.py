import io
import os
import re
from typing import List

import cv2
import easyocr
import numpy as np
import onnxruntime
import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights
from transformers import AutoTokenizer
from scipy import ndimage

import config
from utils import utils


class ImageSpamDetector:
    """Defines composite model for binary classification.

    Composite model integrates fine-tuned ViT image feature extractor, huggingface's language automodel 
    text feature extractor and pre-trained catboost model.
    """

    common_misspellings, short_forms, replacements_dict = utils.load_text_preprocessing_data()
    pattern = re.compile(r'\b(?:%s)\b' % '|'.join(map(re.escape, common_misspellings.keys())))

    def __init__(self):
        self.vit_feature_extractor = self._load_model(self._get_vit_feature_extractor_path())
        self.vit_transforms = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        self.language_model = self._load_model(self._get_language_model_path())
        self.tokenizer = AutoTokenizer.from_pretrained(config.models_dir)
        self.catboost = self._load_model(self._get_boosting_model_path())
        self.easyocr_reader = self._load_easyocr_reader()


    @staticmethod
    def _preprocess_image(img):
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray_sharpened_img = cv2.filter2D(gray_img, -1, sharpen_kernel)
        return gray_sharpened_img


    @staticmethod
    def _replace_symbols(text, symb_mapping):
        for symbol in symb_mapping.keys():
            if symbol in text:
                text = text.replace(symbol, symb_mapping[symbol])
        return text


    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def _get_vit_feature_extractor_path(self):
        """Generates a path to a file that stores ONNX representation of fine-tuned ViT image feature extractor.

        If generated file path does not exist, calls utility function that generates ONNX representation 
        of the model and saves file to that path.
        """
        vit_feature_extractor_path = os.path.join(config.models_dir, f"{config.vision_feature_extractor_filename}.onnx")
        if os.path.exists(vit_feature_extractor_path):
            return vit_feature_extractor_path
        else:
            utils.generate_vit_feature_extractor(vit_feature_extractor_path)
            return vit_feature_extractor_path

    
    def _get_language_model_path(self):
        """Generates a path to a file that stores ONNX representation of language model.

        If generated file path does not exist, calls utility function that generates ONNX representation 
        of the model and saves file to that path.
        """
        language_model_path = os.path.join(config.models_dir, f"{config.language_model_filename}.onnx")
        if os.path.exists(language_model_path):
            return language_model_path
        else:
            utils.generate_language_model(language_model_path)
            return language_model_path
        

    def _get_boosting_model_path(self):
        """Generates a path to a file that stores ONNX representation of boosting model.

        If generated file path does not exist, calls utility function that generates ONNX representation 
        of the model and saves file to that path.
        """
        boosting_model_path = os.path.join(config.models_dir, f"{config.boosting_model_filename}.onnx")
        if os.path.exists(boosting_model_path):
            return boosting_model_path
        else:
            utils.generate_boosting_model(boosting_model_path)
            return boosting_model_path


    def _load_easyocr_reader(self):
        """Loads OCR model that extracts text from images.

        If OCR model directory doesn't exist, creates it and saves model's files to that directory. 
        Otherwise loads model from local directory. 
        """
        if not os.path.exists(config.ocr_model_dir): 
            reader = easyocr.Reader(["ru", "en"], cudnn_benchmark=True, model_storage_directory=config.ocr_model_dir)
            return reader
        else:
            reader = easyocr.Reader(["ru", "en"], cudnn_benchmark=True, download_enabled=False, model_storage_directory=config.ocr_model_dir)
            return reader
        

    def _replace(self, match):
        return self.common_misspellings.get(match.group(0), match.group(0))


    def _correct_misspelled(self, text):
        return self.pattern.sub(self._replace, text)


    def _load_model(self, model_path):
        session_options = onnxruntime.SessionOptions()
        return onnxruntime.InferenceSession(model_path,
                                            session_options,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        
    def _extract_image_features(self, img):
        input_tensor = self.vit_transforms(img).unsqueeze(0)
        image_features = self.vit_feature_extractor.run(None, {"input": input_tensor.numpy()})
        return torch.tensor(image_features[0])

    
    def _extract_text_from_image(self, 
                                 img: Image, 
                                 rotation_angles: List[int] = config.ocr["rotation_angles"], 
                                 batch_size: int = config.ocr["batch_size"],
                                 confidence_threshold: float = config.ocr["confidence_threshold"]):

        text = []

        preprocessed_img = self._preprocess_image(img)

        for angle in rotation_angles + [0]:
            rotated_img = ndimage.rotate(preprocessed_img, angle)
            extracted_data = self.easyocr_reader.readtext(rotated_img, batch_size=batch_size)
            text.extend([item[1] for item in extracted_data if item[2] > confidence_threshold])
        return " ".join(text)

    
    def _preprocess_ocr_text(self, text):
        text = text.strip().lower()
        text = self._correct_misspelled(text)
        text = self._replace_symbols(text, self.replacements_dict)
        text = self._replace_symbols(text, self.short_forms)
        return " ".join(text.split())


    def _extract_text_features(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k:v.cpu().numpy() for k, v in inputs.items()}
        text_features = self.language_model.run(None, inputs)
        text_features = self._mean_pooling([torch.from_numpy(text_features[0])], torch.from_numpy(inputs['attention_mask']))
        return text_features

        
    def predict(self, input_file):
        """Predicts the probability of input image being a spam image. 

        Input image is transformed to appropriate representation and passed into pre-trained ViT feature
        extractor to obtain image feature vector. Then text is extracted from the image, preprocessed and passes into 
        language model feature extractor to obtain text feature vector. Image and text feature vectors are 
        concatenated and passed into catboost model that returns the probability of image being a spam image. 
        Classification labels is obtained by thresholding the probability with a value predefined in configuration file. 
        """
        input_file_bytes = io.BytesIO(input_file.file.read())
        img = Image.open(input_file_bytes)

        image_features = self._extract_image_features(img)

        input_text = self._extract_text_from_image(img)
        preprocessed_input_text = self._preprocess_ocr_text(input_text)
        text_features = self._extract_text_features(preprocessed_input_text)

        features = torch.cat([image_features, text_features], dim=1)

        probability = self.catboost.run(None, {"features": features.numpy()})[1][0][1]

        return {"prob": probability,
                "label": int(probability > config.cls_threshold)}