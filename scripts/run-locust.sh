#!/bin/bash
locust -f tests/locust.py --host http://localhost:5000
