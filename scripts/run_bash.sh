#! /bin/bash

docker compose run --rm \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    pss \
    bash
