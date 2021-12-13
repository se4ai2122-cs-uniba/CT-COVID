#!/bin/bash
docker run --add-host host.docker.internal:host-gateway --name=grafana -p 3000:3000 grafana/grafana-enterprise
