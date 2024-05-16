import torch
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models_dir = "optimized_for_inference"
vision_model_filename = "vit_model"
vision_feature_extractor_filename = "vit_feature_extractor"
boosting_model_filename = "catboost"
language_model_filename = "language_model"

vision_model_url = "https://drive.google.com/file/d/1triN30b_f3O5tXKEcBSxH069FNQelnL2/view?usp=sharing"
boosting_model_url = "https://drive.google.com/file/d/1rkudEEQ-jxy61WiOdiDxXj6Wv-6zwBcn/view?usp=sharing"
pretrained_language_model_name_or_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

ocr_model_dir = os.path.join(models_dir, "ocr")
ocr = {"batch_size": 64,
       "confidence_threshold": 0.25,
       "rotation_angles": [20, -20]}

cls_threshold = 0.49