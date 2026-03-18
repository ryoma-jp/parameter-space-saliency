#! /bin/bash

docker compose run --rm -u $(id -u):$(id -g) \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    pss \
    bash
