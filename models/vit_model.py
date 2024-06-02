import io
import os

import torch
from torchvision.models import ViT_B_16_Weights
from PIL import Image

import config
from utils import utils


class VitModel:
    """Defines fine-tuned Vision transformer model for binary classification.

    This class provides functionality to load pre-trained scripted ViT model and predict 
    whether an image is spam or not. 
    
    Attributes:
        scripted_vit_path: A path to scripted ViT model file.
        scripted_vit_model: A scripted ViT model.
        scripted_optimized_vit_model: A scripted optimize for inference ViT model.
        vit_transforms: torchvision transforms to perform on input image. 

    """
    def __init__(self):
        self.scripted_vit_path = self._get_scripted_vit()
        self.scripted_vit_model = torch.jit.load(self.scripted_vit_path).eval()
        self.scripted_optimized_vit_model = torch.jit.optimize_for_inference(self.scripted_vit_model).to(config.device)
        self.vit_transforms = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()


    def _get_scripted_vit(self):
        """Generates a path to scripted ViT model file.

        If generated file path does not exist, calls utility function that generates scripted ViT model
        and saves file to that path.  
        """
        scripted_vit_path = os.path.join(config.models_dir, f"{config.vision_model_filename}_scripted.pt")
        if os.path.exists(scripted_vit_path):
            return scripted_vit_path
        else:
            utils.generate_scripted_vit(scripted_vit_path)
            return scripted_vit_path


    def predict(self, input_file):
        """Predicts the probability of input image being a spam image. 

        Input image is transformed to appropriate representation and passed into pre-trained ViT model.
        Output logit value is converted to probability using sigmoid function. Classification labels is obtained
        by thresholding the probability with a value predefined in configuration file.
        """
        input_file_bytes = io.BytesIO(input_file.file.read())
        img = Image.open(input_file_bytes)

        input_tensor = self.vit_transforms(img).unsqueeze(0).to(config.device)

        probability = torch.sigmoid(self.scripted_optimized_vit_model(input_tensor)).item()

        return {"prob": probability,
                "label": int(probability > config.cls_threshold)}