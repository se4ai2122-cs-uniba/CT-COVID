import io
import os
import torch
import torchvision
import uvicorn

from http import HTTPStatus
from PIL import Image as pil
from fastapi import Query, FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from covidx.utils.plot import save_binary_attention_map
from covidx.ct.models import CTNet

# Some global variables
DEVICE = None
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
def load_models():
    # Get the device to use
    # Set the device to use globally	    # Get the device to use
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(DEVICE))

    # Load the model, map_location makes it work also on CPU
    model = CTNet(num_classes=3, pretrained=False)
    state_filepath = os.path.join(MODELS_PATH, MODEL_NAME)
    model_params = torch.load(state_filepath, map_location='cpu')['model']
    model.load_state_dict(model_params)
    MODEL_WRAPPERS['ctnet'] = model

    # Move the model to device
    model.to(DEVICE)

    # Make sure the model is set to evaluation mode
    model.eval()


@app.get(
    "/", tags=["General"],
    summary="Does nothing. Use this to test the connectivity to the service.",
    responses={
        200: {
            "description": "A HTTP OK-status message with a welcome message.",
            "content": {
                "application/json": {
                    "example": {
                        "message": "OK",
                        "status-code": 200,
                        "data": {"message": "Welcome to the CT-COVID analysis service!"}
                    }
                }
            }
        }
    }
)
def index(request: Request):
    # A OK-status response when connecting to the root
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to the CT-COVID analysis service!"},
    }
    return response


@app.get(
    "/models", tags=["Models"],
    summary="Get the list of available models in the system.",
    responses={
        200: {
            "description": "A list of model names.",
            "content": {
                "application/json": {
                    "example": {
                        "message": "OK",
                        "status-code": 200,
                        "data": {"models": ["model1", "model2", "model3"]}
                    }
                }
            }
        }
    }
)
def get_models_list(request: Request):
    # Get the available models
    available_models = list(MODEL_WRAPPERS.keys())

    # Send an OK-status response with the list of available models
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"models": available_models},
    }
    return response


def iterfile():
    with open("example.png", mode="rb") as file_like:
        yield from file_like


@app.post(
    "/predict", tags=["Prediction"],
    summary="Given the bounding box of the relevant area and the image file of a CT scan, tell if the patient has COVID"
            " and return an heatmap of the pixels on which the model focused in order to provide the prediction.",
    responses={
        200: {
            "description": "A disease prediction, i.e. one of {}, and also an heatmap. The red zone was the most"
                           "relevant for the prediction.".format(list(PREDICTION_TAGS.values())),
            "content": {
                "image/png": {"example": {"prediction": "COVID - 19"}}
            }
        }
    }
)
async def predict(request: Request,
    xmin: int = Query(None, description="The top-left bounding box X-coordinate."),
    ymin: int = Query(None, description="The top-left bounding box Y-coordinate."),
    xmax: int = Query(None, description="The bottom-right bounding box X-coordinate."),
    ymax: int = Query(None, description="The bottom-right bounding box Y-coordinate."),
    file: UploadFile = File(..., description="The CT image to predict.")
    ):
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
        prediction, att1, att2 = model(tensor, attention=True)
        prediction = torch.argmax(prediction, dim=1).item()

    stream = io.BytesIO()
    tensor = torchvision.transforms.functional.normalize(tensor, (-1,), (2,))
    save_binary_attention_map(stream, tensor, att1, att2)
    stream.seek(0)
    # Send a response with the prediction and the attention map
    return StreamingResponse(stream, headers={"prediction": PREDICTION_TAGS[prediction]}, media_type="image/png")


def upload_file(bbox: tuple, file: UploadFile = File(...)):
    """A synchronous utility function used to upload an image file.
        :param bbox: The image bounding box.
        :param file: The FastAPI file uploader object.
        :return: A preprocessed PIL image."""
    # Read the file contents
    contents = file.file.read()

    # Open it as a PIL image and preprocess it
    with pil.open(io.BytesIO(contents)) as img:
        # Preprocess the image using Crop + Resize (with bicubic interpolation)
        img = img.convert(mode='L').crop(bbox).resize((224, 224), resample=pil.BICUBIC)

    return img


if __name__ == "__main__":
    # Run uvicorn when running this script
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True, reload_dirs=["src", "models"])
