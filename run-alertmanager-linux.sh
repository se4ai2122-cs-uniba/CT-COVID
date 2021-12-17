#!/bin/bash
cd ..
docker run -d -p 9093:9093 --add-host host.docker.internal:host-gateway --name alertmanager -v "$PWD/alert_rules.yml":/etc/prometheus/alert_rules.yml prom/alertmanager
