#!/bin/bash
docker run --rm --gpus 1 -it -p 5000:5000 ct-covid:cuda
