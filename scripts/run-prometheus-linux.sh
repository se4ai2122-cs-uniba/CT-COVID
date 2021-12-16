#!/bin/bash
cd ..
docker run -d -p 9090:9090 --add-host host.docker.internal:host-gateway -v "$PWD/prometheus.yml":/etc/prometheus/prometheus.yml prom/prometheus
