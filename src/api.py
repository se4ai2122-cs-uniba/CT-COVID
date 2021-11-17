import io
import os
import torch
import torchvision
import uvicorn

from http import HTTPStatus
from PIL import Image as pil
from fastapi import FastAPI, Request, UploadFile, File
from covidx.ct.models import CTNet

# Some global variables
MODELS_PATH = 'models'
MODEL_NAME = 'ct_net.pt'
MODEL_WRAPPERS = dict()
PREDICTION_TAGS = {
    0: 'Normal',
    1: 'Pneumonia',
    2: 'COVID - 19'
}

# Define the FastAPI application
app = FastAPI(
    title="CT-COVID",
    description="This API lets you make predictions of diseases analysing CT-scans.",
    version="0.1",
)


@app.on_event("startup")
def _load_models():
    # Get the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))

    # Load the model, map_location makes it work also on CPU
    model = CTNet(num_classes=3, pretrained=False)
    state_filepath = os.path.join(MODELS_PATH, MODEL_NAME)
    model_params = torch.load(state_filepath, map_location='cpu')['model']
    model.load_state_dict(model_params)
    MODEL_WRAPPERS['ctnet'] = model

    # Move the model to device
    model.to(device)

    # Make sure the model is set to evaluation mode
    model.eval()


@app.get("/", tags=["General"])
def _index(request: Request):
    # A OK-status response when connecting to the root
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to CT-COVID classifier!"},
    }
    return response


@app.get("/models", tags=["Models"])
def _get_models_list(request: Request):
    # Get the available models
    available_models = list(model_wrappers_dict.keys())

    # Send an OK-status response with the list of available models
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": available_models,
    }
    return response


@app.post("/predict", tags=["Prediction"])
async def predict(request: Request, xmin: int, ymin: int, xmax: int, ymax: int, file: UploadFile = File(...)):
    # Load and preprocess the image by upload
    bbox = (xmin, ymin, xmax, ymax)
    img = upload_file(bbox, file)

    # Get the device to use and get the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL_WRAPPERS['ctnet']

    # Convert the input image to a tensor, normalize it,
    # move it to device and unsqueeze the batch dimension
    tensor = torchvision.transforms.functional.to_tensor(img)
    tensor = torchvision.transforms.functional.normalize(tensor, (0.5,), (0.5,))
    tensor = tensor.to(device).unsqueeze(0)

    # Obtain the prediction by the model
    with torch.no_grad():  # Disable gradient graph building
        prediction = model(tensor)
        prediction = torch.argmax(prediction, dim=1).item()

    # Send an OK-status response with the prediction
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "prediction": PREDICTION_TAGS[prediction]
    }
    return response


def upload_file(bbox: tuple, file: UploadFile = File(...)):
    # A synchronous utility function used to upload the image file
    contents = file.file.read()
    with pil.open(io.BytesIO(contents)) as img:
        # Preprocess the image using Crop + Resize (with bicubic interpolation)
        img = img.convert(mode='L').crop(bbox).resize((224, 224), resample=pil.BICUBIC)
    return img


if __name__ == "__main__":
    # Run uvicorn when running this script
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True, reload_dirs=["src", "models"])
