import os
import pickle

import gdown
import torch 
from torchvision.models import vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoModel, AutoTokenizer
from catboost import CatBoostClassifier

import config


class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, base_model):
        super(ViTFeatureExtractor, self).__init__()
        self.feature_extractor = create_feature_extractor(base_model, return_nodes=["getitem_5"])

    def forward(self, x):
        return self.feature_extractor(x)[["getitem_5"][0]]


def download_pretrained(download_url: str,
                        save_path: str,
                        fuzzy: bool = True,
                        quiet: bool = True):

    if not os.path.exists(save_path):
        gdown.download(download_url, save_path, fuzzy=fuzzy, quiet=quiet)


def load_pretrained_vit(image_size: int):

    pretrained_path = os.path.join(config.models_dir, "temp", f"{config.vision_model_filename}.pth")

    download_pretrained(download_url=config.vision_model_url,
                        save_path=pretrained_path)
    
    vit_model = vit_b_16(image_size=image_size)
    vit_model.heads = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=1))
    vit_model.load_state_dict(torch.load(f=pretrained_path))

    os.remove(pretrained_path)

    return vit_model


def generate_scripted_vit(model_path: str,
                          image_size: int = 384):

    vit_model = load_pretrained_vit(image_size)

    vit_example_inputs = torch.rand(1, 3, image_size, image_size)
    scripted_model = torch.jit.script(vit_model.cpu().eval(), example_inputs=[vit_example_inputs])
    torch.jit.save(scripted_model, model_path)


def generate_vit_feature_extractor(model_path: str,
                                   image_size: int = 384):

    vit_model = load_pretrained_vit(image_size)
    vit_feature_extractor = ViTFeatureExtractor(vit_model).eval()
    vit_example_input = torch.rand(1, 3, image_size, image_size)

    torch.onnx.export(vit_feature_extractor,
                      vit_example_input,
                      model_path,
                      export_params=True,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0 : "batch_size"},
                                    "output": {0 : "batch_size"}})


def generate_language_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_language_model_name_or_path, resume_download=True)
    lang_model = AutoModel.from_pretrained(config.pretrained_language_model_name_or_path, resume_download=True)
    lang_model.eval()

    sample = ["пример предложения для языковой модели"]
    inputs = tokenizer(sample, padding=True, truncation=True, return_tensors="pt")

    with torch.inference_mode():
        symbolic_names = {0: "batch_size", 1: "max_seq_len"}
        torch.onnx.export(lang_model,
                          args=tuple(inputs.values()),
                          f=model_path,
                          opset_version=11,
                          do_constant_folding=True,  
                          input_names=["input_ids", "attention_mask", "token_type_ids"],
                          output_names=['start', 'end'],  
                          dynamic_axes={"input_ids": symbolic_names,
                                        "attention_mask": symbolic_names,
                                        "token_type_ids": symbolic_names,
                                        "start": symbolic_names,
                                        "end": symbolic_names})
      
    tokenizer.save_pretrained(config.models_dir)


def generate_boosting_model(model_path):

    pretrained_path = os.path.join(config.models_dir, "temp", f"{config.boosting_model_filename}.cbm")

    download_pretrained(download_url=config.boosting_model_url,
                        save_path=pretrained_path)

    cbm = CatBoostClassifier().load_model(pretrained_path)

    cbm.save_model(model_path,
                   format="onnx",
                   export_parameters={'onnx_domain': 'ai.catboost',
                                      'onnx_model_version': 1})
    
    os.remove(pretrained_path)


def load_text_preprocessing_data(data_dir: str = "data"):
    
    with open(os.path.join(data_dir, "common_misspellings.pickle"), "rb") as f:
        common_misspellings = pickle.load(f)

    with open(os.path.join(data_dir, "short_forms.pickle"), "rb") as f:
        short_forms = pickle.load(f)

    with open(os.path.join(data_dir, "replacements_dict.pickle"), "rb") as f:
        replacements_dict = pickle.load(f)
    
    return common_misspellings, short_forms, replacements_dict