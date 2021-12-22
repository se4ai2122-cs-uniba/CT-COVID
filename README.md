# CT-COVID [![codecov](https://codecov.io/gh/se4ai2122-cs-uniba/CT-COVID/branch/main/graph/badge.svg?token=62DKBATM5P)](https://codecov.io/gh/se4ai2122-cs-uniba/CT-COVID) ![Pylint Report](https://github.com/se4ai2122-cs-uniba/CT-COVID/actions/workflows/linter.yml/badge.svg)

*Screening CT 3d images for interpretable COVID19 detection.*

# Usage
## API Endpoints
The API is accessible at the following endpoints:
- `/` which gives a welcome message
- `/docs` which provides a documentation of the API
- `/models` which provides a list of available models
- `/predict` used to receive prediction for a given image and his bounding box 

## Request of prediction
The request is made by passing:
- the image to be processed
- the bounding box of the image

An example of request with `curl`:
```bash
curl -X 'POST' \
  'http://localhost:5000/predict?xmin=0&ymin=0&xmax=224&ymax=224' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=filename.png;type=image/png'
```

## Container
### Pull docker container
Our container is hosted on [dockerhub](https://hub.docker.com/r/peppocola/ct-covid):

`docker pull peppocola/ct-covid:v1`

### Run container
It is preferable to run the container with a GPU, but if you don't have one, you can run it with a CPU without problems.
#### Run on CPU
```docker run -p 5000:5000 --rm -it --name=ct-covid peppocola/ct-covid:v1```
#### Run on GPU
```docker run -p 5000:5000 --rm -it --gpus 1 --name=ct-covid peppocola/ct-covid:v1```

## Run locally
### Requirements
If you just want to run the api:
```bash
pip install -r requirements.txt
```

If you want to use tools like `locust`, `pytest` and `great_expectations`:

```bash
pip install -r requirements_dev.txt
```

### Run api
```bash
python src/api.py
```
Those commands will run the api, which will accept requests on port 5000.

## Frontend
```bash
cd src/frontend
npm install
npm start
```

## Prometheus
```bash
docker run -d -p 9090:9090 --add-host host.docker.internal:host-gateway \
    -v "$PWD/prometheus.yml":/etc/prometheus/prometheus.yml \
    --name=prometheus prom/prometheus
```

## Grafana
```bash
docker run -d -p 3000:3000 --add-host host.docker.internal:host-gateway \
    --name=grafana grafana/grafana-enterprise
```

## Locust
```bash
locust -f tests/locust.py --host http://localhost:5000
```

## PyTest
To run pytest without gpu:

```bash
PYTHONPATH=src pytest -m "not gpu" --cov src/covidx tests/
```

To run pytest with gpu:
```bash
PYTHONPATH=src pytest --cov src/covidx tests/
```

## Great Expectations
```bash
cd tests
for checkpoint in ct-train ct-valid ct-test
do
  great_expectations --v3-api checkpoint run $checkpoint
done
```
