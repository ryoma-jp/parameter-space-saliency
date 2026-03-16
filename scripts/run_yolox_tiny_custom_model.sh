#! /bin/bash

MODEL_WEIGHTS_PATH=externals/YOLOX/weights/yolox_tiny.pth
OUTPUT_ROOT=results/yolox_tiny_custom_model

if [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
    echo "Missing checkpoint: $MODEL_WEIGHTS_PATH"
    echo "Download or place YOLOX Tiny weights at externals/YOLOX/weights/yolox_tiny.pth"
    exit 1
fi

rm -rf "$OUTPUT_ROOT"
docker compose run --rm -u $(id -u):$(id -g) \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    -e TORCH_HOME=/work/.cache/torch \
    -e PYTHONPATH=/work/externals/YOLOX:/work \
    pss \
        bash -lc "pip3 install -q loguru && python3 parameter_and_input_saliency.py \
        --task detection \
        --model_source custom_module \
        --model_class_path model_adapter.yolox_tiny_wrapper.YOLOXTinyWrapper \
        --model_weights_path "$MODEL_WEIGHTS_PATH" \
        --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg \
        --output_root "$OUTPUT_ROOT" \
            --target_type predicted_top1"
