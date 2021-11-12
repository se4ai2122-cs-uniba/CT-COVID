import io
import os
import torch
import torchvision
from http import HTTPStatus
from covidx.ct.models import CTNet
from PIL import Image as pil
from fastapi import FastAPI, Request, UploadFile, File
import uvicorn


MODELS_PATH = 'models'
MODEL_NAME = 'ct_net.pt'
model_wrappers_dict = {}

# Define application
app = FastAPI(
    title="CT-COVID",
    description="This API lets you make predictions of diseases analysing CT-scans.",
    version="0.1",
)

# Loads the model
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


# Run the application
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


# Predict the disease
@app.post("/predict")
async def predict(request: Request, xmin: int, ymin: int, xmax: int, ymax: int, file: UploadFile = File(...)):
    b = (xmin, ymin, xmax, ymax)
    img = upload_file(b, file)
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

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "prediction": prediction_dict[prediction]
    }

    return response


def upload_file(b: tuple, file: UploadFile = File(...)):
    contents = file.file.read()
    with pil.open(io.BytesIO(contents)) as img:
        # Preprocess the image
        img = img.convert(mode='L').crop(b).resize((224, 224), resample=pil.BICUBIC)

    return img


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True, reload_dirs=['src', 'models'])
