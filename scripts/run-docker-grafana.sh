#!/bin/bash
docker run -d -p 3000:3000 --add-host host.docker.internal:host-gateway \
    --name=grafana grafana/grafana-enterprise
