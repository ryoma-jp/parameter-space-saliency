#! /bin/bash

MODEL_WEIGHTS_PATH=externals/YOLOX/weights/yolox_tiny.pth
OUTPUT_ROOT_BASE=results/yolox_tiny_custom_model
MODEL_KWARGS_JSON='{"exp_path":"/work/externals/YOLOX/exps/default/yolox_tiny.py","ckpt_path":"externals/YOLOX/weights/yolox_tiny.pth","device":"cpu"}'
PREPROCESS_CFG_JSON='{"resize":[416,416],"letterbox":true,"pad_value":114,"channel_order":"bgr"}'

if [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
    echo "Missing checkpoint: $MODEL_WEIGHTS_PATH"
    echo "Download or place YOLOX Tiny weights at externals/YOLOX/weights/yolox_tiny.pth"
    exit 1
fi

for method in auto matching; do
    OUTPUT_ROOT="${OUTPUT_ROOT_BASE}_${method}"
    echo "Running input saliency with method=${method}"
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
            --model_import_root /work/externals/YOLOX \
            --model_class_path yolox.models.build.yolox_custom \
            --model_kwargs_json '$MODEL_KWARGS_JSON' \
            --preprocess_cfg_json '$PREPROCESS_CFG_JSON' \
            --image_path raw_images/coco2017/val2017/000000397133.jpg \
            --det_annotations_json raw_images/coco2017/annotations/instances_val2017.json \
            --output_root "$OUTPUT_ROOT" \
            --det_objective_mode gt_all_instances \
            --det_objective_provider yolox_official \
            --input_saliency_method "$method" \
            --target_type true_label"
done
