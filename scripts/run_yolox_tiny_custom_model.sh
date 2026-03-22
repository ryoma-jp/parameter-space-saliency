#! /bin/bash

MODEL_WEIGHTS_PATH=externals/YOLOX/weights/yolox_tiny.pth
OUTPUT_ROOT_BASE=results/yolox_tiny_custom_model
IMAGE_DIR=raw_images/coco2017/val2017
MODEL_KWARGS_JSON='{"exp_path":"/work/externals/YOLOX/exps/default/yolox_tiny.py","ckpt_path":"externals/YOLOX/weights/yolox_tiny.pth","device":"cpu"}'
PREPROCESS_CFG_JSON='{"resize":[416,416],"letterbox":true,"pad_value":114,"channel_order":"bgr"}'
DET_FP_LOC_WEIGHT=1.0
DET_FP_LOC_IOU_THRESHOLD=0.3
DET_FP_LOC_GATE_SHARPNESS=12.0
DET_FP_LOC_SCORE_POWER=1.0

if [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
    echo "Missing checkpoint: $MODEL_WEIGHTS_PATH"
    echo "Download or place YOLOX Tiny weights at externals/YOLOX/weights/yolox_tiny.pth"
    exit 1
fi

#for method in auto matching; do
for method in auto; do
    OUTPUT_ROOT="${OUTPUT_ROOT_BASE}_${method}"
    echo "Running input saliency with method=${method}, det_fp_loc_weight=${DET_FP_LOC_WEIGHT}"
    rm -rf "$OUTPUT_ROOT"

    docker compose run --rm -u $(id -u):$(id -g) \
        -e HOME=/work \
        -e XDG_CACHE_HOME=/work/.cache \
        -e TORCH_HOME=/work/.cache/torch \
        -e PYTHONPATH=/work/externals/YOLOX:/work \
        -e IMAGE_DIR="$IMAGE_DIR" \
        -e OUTPUT_ROOT="$OUTPUT_ROOT" \
        -e MODEL_KWARGS_JSON="$MODEL_KWARGS_JSON" \
        -e PREPROCESS_CFG_JSON="$PREPROCESS_CFG_JSON" \
        -e DET_FP_LOC_WEIGHT="$DET_FP_LOC_WEIGHT" \
        -e DET_FP_LOC_IOU_THRESHOLD="$DET_FP_LOC_IOU_THRESHOLD" \
        -e DET_FP_LOC_GATE_SHARPNESS="$DET_FP_LOC_GATE_SHARPNESS" \
        -e DET_FP_LOC_SCORE_POWER="$DET_FP_LOC_SCORE_POWER" \
        -e METHOD="$method" \
        pss \
            bash -lc 'set -euo pipefail

            if [ ! -d "$IMAGE_DIR" ]; then
                echo "Missing image directory in container: $IMAGE_DIR"
                exit 1
            fi

            mapfile -t image_paths < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)
            total=${#image_paths[@]}
            if [ "$total" -eq 0 ]; then
                echo "No supported image files found in container under: $IMAGE_DIR"
                exit 1
            fi

            for idx in "${!image_paths[@]}"; do
                image_path="${image_paths[$idx]}"
                echo "[$METHOD] ($((idx + 1))/$total) processing: ${image_path}"
                python3 parameter_and_input_saliency.py \
                    --task detection \
                    --model_source custom_module \
                    --model_import_root /work/externals/YOLOX \
                    --model_class_path yolox.models.build.yolox_custom \
                    --model_kwargs_json "$MODEL_KWARGS_JSON" \
                    --preprocess_cfg_json "$PREPROCESS_CFG_JSON" \
                    --image_path "$image_path" \
                    --det_annotations_json raw_images/coco2017/annotations/instances_val2017.json \
                    --output_root "$OUTPUT_ROOT" \
                    --det_objective_mode gt_all_instances \
                    --det_objective_provider yolox_official \
                    --det_fp_loc_weight "$DET_FP_LOC_WEIGHT" \
                    --det_fp_loc_iou_threshold "$DET_FP_LOC_IOU_THRESHOLD" \
                    --det_fp_loc_gate_sharpness "$DET_FP_LOC_GATE_SHARPNESS" \
                    --det_fp_loc_score_power "$DET_FP_LOC_SCORE_POWER" \
                    --input_saliency_method "$METHOD" \
                    --target_type true_label
            done'
done
