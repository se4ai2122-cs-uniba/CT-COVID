import pytest
from fastapi.testclient import TestClient
from http import HTTPStatus
from api import app
from api import PREDICTION_TAGS
from utils_test import get_formatted_params, get_image_bytes
import numpy as np

with TestClient(app) as client:


    @pytest.mark.api
    def test_root():
        response = client.get("/")
        assert response.status_code == HTTPStatus.OK
        assert type(response.json()['message']) == str


    @pytest.mark.api
    def test_docs():
        response = client.get("/docs")
        assert response.status_code == HTTPStatus.OK


    @pytest.mark.api
    def test_get_models():
        response = client.get("/models")
        assert response.status_code == HTTPStatus.OK

        response_models = response.json()['models']
        assert type(response_models) == list
        assert all(map(lambda x: isinstance(x, str), response_models))


    @pytest.mark.api
    def test_predict():
        random_state = np.random.RandomState()
        params = get_formatted_params(random_state)
        image_bytes = get_image_bytes(random_state)
        response = client.post(
            '/predict?' + '&'.join(params),
            files=[('file', ('input-image', image_bytes, 'image/png'))]
        )
        assert response.status_code == HTTPStatus.OK
        assert response.headers['prediction'] in PREDICTION_TAGS.values()
