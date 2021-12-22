#!/bin/bash
docker run -p 5000:5000 --rm -it --gpus 1 --name=ct-covid peppocola/ct-covid:v1
