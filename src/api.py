import io
import os
import torch
import torchvision
from datetime import datetime
from functools import wraps
from http import HTTPStatus

from covidx.ct.models import CTNet
from PIL import Image as pil
from fastapi import FastAPI, Request, UploadFile, File, Depends
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
def _index(request: Request):

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to CT-COVID classifier!"},
    }
    return response


@app.get("/models", tags=["Prediction"])
def _get_models_list(request: Request):


    available_models = list(model_wrappers_dict.keys())

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": available_models,
    }

    return response

@app.post("/predict")
async def upload_predict(request: Request,  xmin: int, ymin: int, xmax: int, ymax: int, file: UploadFile = File(...)):
    b = (xmin,ymin,xmax,ymax)
    img = create_upload_file(b, file)
    prediction = predict(img)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "prediction": prediction
    }


    return response



def create_upload_file(b: tuple, file: UploadFile = File(...)):
    contents = file.file.read()
    with pil.open(io.BytesIO(contents)) as img:
        # Preprocess the image
        img = img.convert(mode='L').crop(b).resize((224,224), resample=pil.BICUBIC)

    return img


def predict(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torchvision.transforms.functional.to_tensor(img)
    model = model_wrappers_dict['ctnet']
    # unsqueeze provides the batch dimension
    tensor = tensor.to(device).unsqueeze(0)
    prediction = model(tensor)
    prediction = torch.argmax(prediction, dim=1).item()

    prediction_dict = {
        0: 'Normal',
        1: 'Pneumonia',
        2: 'COVID - 19'
    }
    return prediction_dict[prediction]