#! /bin/bash

MODEL_WEIGHTS_PATH=results/resnet50/resnet50_exported.pth
OUTPUT_ROOT=results/resnet50_custom_model

if [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
    echo "Missing checkpoint: $MODEL_WEIGHTS_PATH"
    echo "Run scripts/run_resnet50.sh first to export it."
    exit 1
fi

rm -rf "$OUTPUT_ROOT"
docker compose run --rm -u $(id -u):$(id -g) \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    -e TORCH_HOME=/work/.cache/torch \
    pss \
    python3 parameter_and_input_saliency.py \
        --model_source custom_module \
        --model_class_path torchvision.models.resnet50 \
        --model_weights_path "$MODEL_WEIGHTS_PATH" \
        --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg \
        --output_root "$OUTPUT_ROOT" \
        --image_target_label 2
