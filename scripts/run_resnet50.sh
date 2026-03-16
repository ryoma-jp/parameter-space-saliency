#! /bin/bash

OUTPUT_ROOT=results/resnet50

rm -rf $OUTPUT_ROOT
docker compose run --rm pss \
    python3 parameter_and_input_saliency.py \
        --model resnet50 \
        --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg \
        --output_root $OUTPUT_ROOT \
        --image_target_label 2 \
        --export_model_pth $OUTPUT_ROOT/resnet50_exported.pth
