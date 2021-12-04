#!/bin/bash
locust -f test/locust.py --host http://localhost:5000
