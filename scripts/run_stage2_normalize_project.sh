#! /bin/bash

# ---------------------------------------------------------------------------
# Stage2: TP-based normalization + input-space projection
#
# Runs parameter_and_input_saliency.py (--pipeline_stage stage2) for the full run directory.
# Phase 2a: compute TP normalization stats from all stage1 files.
# Phase 2b: normalize all saliencies; optionally compute input-space projection.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Settings (shared with run_stage1_filter_saliency.sh)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS_PATH=externals/YOLOX/weights/yolox_tiny.pth
IMAGE_DIR=${IMAGE_DIR:-raw_images/coco2017/val2017}
MODEL_KWARGS_JSON='{"exp_path":"/work/externals/YOLOX/exps/default/yolox_tiny.py","ckpt_path":"externals/YOLOX/weights/yolox_tiny.pth"}'
DET_ANNOTATIONS_JSON=${DET_ANNOTATIONS_JSON:-raw_images/coco2017/annotations/instances_val2017.json}
PREPROCESS_CFG_JSON='{"resize":[416,416],"letterbox":true,"pad_value":114,"channel_order":"bgr"}'

RUN_DIR=${RUN_DIR:-results/yolox_tiny_pss_paper}

# Projection settings
NO_PROJECTION=${NO_PROJECTION:-0}    # set to 1 to skip input-space projection
TOP_F=${TOP_F:-10}
BOOST_K=${BOOST_K:-100.0}

# Phase to run (all | 2a | 2b)
PHASE=${PHASE:-all}

DET_CONF_THRESHOLD=${DET_CONF_THRESHOLD:-0.3}
DET_NMS_IOU_THRESHOLD=${DET_NMS_IOU_THRESHOLD:-0.45}
DET_MATCH_IOU_THRESHOLD=${DET_MATCH_IOU_THRESHOLD:-0.5}
DET_FN_TAU=${DET_FN_TAU:-0.1}

if [ "$NO_PROJECTION" != "1" ] && [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
    echo "Missing checkpoint: $MODEL_WEIGHTS_PATH"
    echo "(Use NO_PROJECTION=1 to run normalization-only without model weights.)"
    exit 1
fi

# Build stage2 no-projection flag if requested
NO_PROJ_FLAG=""
if [ "$NO_PROJECTION" = "1" ]; then
    NO_PROJ_FLAG="--stage2_no_projection"
fi

docker compose run -T --rm \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    -e TORCH_HOME=/work/.cache/torch \
    -e PYTHONPATH=/work/externals/YOLOX:/work \
    -e RUN_DIR="$RUN_DIR" \
    -e IMAGE_DIR="$IMAGE_DIR" \
    -e MODEL_KWARGS_JSON="$MODEL_KWARGS_JSON" \
    -e PREPROCESS_CFG_JSON="$PREPROCESS_CFG_JSON" \
    -e DET_ANNOTATIONS_JSON="$DET_ANNOTATIONS_JSON" \
    -e DET_CONF_THRESHOLD="$DET_CONF_THRESHOLD" \
    -e DET_NMS_IOU_THRESHOLD="$DET_NMS_IOU_THRESHOLD" \
    -e DET_MATCH_IOU_THRESHOLD="$DET_MATCH_IOU_THRESHOLD" \
    -e DET_FN_TAU="$DET_FN_TAU" \
    -e PHASE="$PHASE" \
    -e NO_PROJ_FLAG="$NO_PROJ_FLAG" \
    -e TOP_F="$TOP_F" \
    -e BOOST_K="$BOOST_K" \
    pss \
        bash -s -- <<'BASH'
        set -euo pipefail
        echo "[Stage2] phase=${PHASE}, run_dir=${RUN_DIR}, no_projection=${NO_PROJ_FLAG:-none}"

        python3 parameter_and_input_saliency.py \
            --pipeline_stage stage2 \
            --run_dir "$RUN_DIR" \
            --pipeline_image_dir "$IMAGE_DIR" \
            --stage2_phase "$PHASE" \
            --model_source custom_module \
            --model_import_root /work/externals/YOLOX \
            --model_class_path yolox.models.build.yolox_custom \
            --model_kwargs_json "$MODEL_KWARGS_JSON" \
            --preprocess_cfg_json "$PREPROCESS_CFG_JSON" \
            --det_annotations_json "$DET_ANNOTATIONS_JSON" \
            --det_conf_threshold "$DET_CONF_THRESHOLD" \
            --det_nms_iou_threshold "$DET_NMS_IOU_THRESHOLD" \
            --det_match_iou_threshold "$DET_MATCH_IOU_THRESHOLD" \
            --det_fn_tau "$DET_FN_TAU" \
            --stage2_top_f "$TOP_F" \
            --stage2_boost_k "$BOOST_K" \
            ${NO_PROJ_FLAG}

        echo "[Stage2] Done."
BASH
