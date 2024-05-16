import json
import os
import requests
import pytest


urls = {
    "vit_16": "http://localhost:8000/vit_predict",
    "whole_pipeline": "http://localhost:8000/predict",
}


def make_request(image_filename: str, endpoint: str):
    image_path = os.path.join("tests/test_data", image_filename)
    with open(image_path, "rb") as f: 
        response = requests.post(urls[endpoint], files={"input_file": f.read()})
    response = json.loads(response.content)
    return response


def test_vit_endpoint():
    response = make_request(image_filename="spam_image.jpg", endpoint="vit_16")
    assert response["label"] == 1

    response = make_request(image_filename="non_spam_image.jpg", endpoint="vit_16")
    assert response["label"] == 0


def test_whole_pipeline_endpoint():
    response = make_request(image_filename="spam_image.jpg", endpoint="whole_pipeline")
    assert response["label"] == 1

    response = make_request(image_filename="non_spam_image.jpg", endpoint="whole_pipeline")
    assert response["label"] == 0