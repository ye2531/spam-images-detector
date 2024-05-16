from fastapi import FastAPI, File, UploadFile
from models.vit_model import VitModel
from models.whole_pipeline import ImageSpamDetector


app = FastAPI()

vit_model = VitModel()
whole_pipeline = ImageSpamDetector()


@app.post("/vit_predict")
async def predict_vit(input_file: UploadFile = File(...)):
    return vit_model.predict(input_file)


@app.post("/predict")
async def predict(input_file: UploadFile = File(...)):
    return whole_pipeline.predict(input_file)