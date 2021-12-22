#!/bin/bash
PYTHONPATH=src pytest -m "not gpu" --cov src/covidx tests/
