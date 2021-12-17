#!/bin/bash
cd ..
docker run -d -p 9090:9090 --add-host host.docker.internal:host-gateway --name prometheus -v "$PWD/prometheus.yml":/etc/prometheus/prometheus.yml -v "$PWD/alert_rules.yml":/etc/prometheus/alert_rules.yml prom/prometheus --config.file /etc/prometheus/prometheus.yml

