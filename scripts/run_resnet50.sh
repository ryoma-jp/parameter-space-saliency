#! /bin/bash

docker compose run --rm pss \
    python3 parameter_and_input_saliency.py \
        --model resnet50 \
        --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg \
        --image_target_label 2
