import os
import torch
import json
import pickle
import torchvision
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List
from covidx.ct.models import CTNet
from PIL import Image as pil
from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel

MODELS_PATH = 'models'
MODEL_NAME = 'ct_net.pt'
model_wrappers_dict = {}
image_list = []
device = None
# Define application
app = FastAPI(
    title="CT-COVID",
    description="This API lets you make predictions of diseases analysing CT-scans.",
    version="0.1",
)


class Box(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def construct_response(f):

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,

        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_models():

    # Get the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    model = CTNet(num_classes=3)
    state_filepath = os.path.join(MODELS_PATH, MODEL_NAME)
    model.load_state_dict(torch.load(state_filepath)['model'])
    model_wrappers_dict['ctnet'] = model
    # Move the model to device
    model.to(device)
    # Make sure the model is set to evaluation mode
    model.eval()

@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to CT-COVID classifier!"},
    }
    return response


@app.get("/models", tags=["Prediction"])
@construct_response
def _get_models_list(request: Request):


    available_models = model_wrappers_dict.keys()

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": available_models,
    }

    return response

@app.post("/predict/")
async def upload_predict(b: Box, file: UploadFile = File(...)):
    img = create_upload_file(b, file)




def create_upload_file(b: Box, file: UploadFile = File(...)):



    box = (b.xmin, b.ymin, b.xmax, b.ymax)
    with pil.open(file.filename) as img:
        # Preprocess the image
        img = img.convert(mode='L').crop(box).resize((244,244), resample=pil.BICUBIC)

    return img


def predict(request: Request, img):

    tensor = torchvision.transforms.ToTensor(img)
    model = model_wrappers_dict['ctnet']
    # unsqueeze provides the batch dimension
    tensor = tensor.to(device).unsqueeze(0)
    prediction = model(tensor)
    prediction = torch.argmax(prediction, dim=1).item()

    prediction_dict = {
        0: 'normal',
        1: 'Pneumonia',
        2: 'COVID - 19'
    }


    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "prediction": prediction_dict[prediction]
        }

