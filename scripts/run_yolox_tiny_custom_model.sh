#! /bin/bash

# ---------------------------------------------------------------------------
# Base model / dataset settings
# ---------------------------------------------------------------------------
# YOLOX Tiny checkpoint to load.
MODEL_WEIGHTS_PATH=externals/YOLOX/weights/yolox_tiny.pth

# Directory of input images processed in batch.
# Supported extensions: jpg / jpeg / png.
IMAGE_DIR=raw_images/coco2017/val2017

# Custom model constructor arguments passed to `yolox.models.build.yolox_custom`.
# Adjust `exp_path` or `ckpt_path` when switching YOLOX variants or checkpoints.
MODEL_KWARGS_JSON='{"exp_path":"/work/externals/YOLOX/exps/default/yolox_tiny.py","ckpt_path":"externals/YOLOX/weights/yolox_tiny.pth","device":"cpu"}'

# Input preprocessing configuration consumed by the adapter.
# `letterbox=true` should stay aligned with the training / inference setup.
PREPROCESS_CFG_JSON='{"resize":[416,416],"letterbox":true,"pad_value":114,"channel_order":"bgr"}'

# Run mode configuration.
# Available values:
#   - test: process only the first TEST_MAX_FILES images after sorting
#   - full: process all discovered images
RUN_MODE=${RUN_MODE:-test}
TEST_MAX_FILES=${TEST_MAX_FILES:-100}

# ---------------------------------------------------------------------------
# Unified hotness objective settings
# ---------------------------------------------------------------------------
# Detection objective mode.
# Available values:
#   - hotness_unified: unified visualization over TP / FN / FP-A / FP-B
#   - gt_all_instances: existing GT instance coverage objective
#   - gt_all_classes: existing GT class coverage objective
#   - legacy_single_class: legacy single-class objective
DET_OBJECTIVE_MODE=${DET_OBJECTIVE_MODE:-hotness_unified}

# Objective provider.
# Available values:
#   - yolox_official: YOLOX official training loss provider
#   - auto: automatically infer from the model
#   - none: disable provider usage and rely only on the adapter-side objective
DET_OBJECTIVE_PROVIDER=${DET_OBJECTIVE_PROVIDER:-yolox_official}

# Weight of FP-A (wrong location / no matching GT) term.
# Set to 0.0 to disable this term.
DET_FP_LOC_WEIGHT=1.0

# IoU threshold below which a prediction is treated as FP-A-like.
# Lower values make FP-A classification stricter; higher values include more cases.
DET_FP_LOC_IOU_THRESHOLD=0.3

# Legacy fp_loc sharpness parameter.
# It is not a primary control in hotness_unified, but is kept for compatibility.
DET_FP_LOC_GATE_SHARPNESS=12.0

# Power applied to FP-A score penalty.
# Larger values emphasize high-confidence FP-A cases more strongly.
DET_FP_LOC_SCORE_POWER=1.0

# Margin for FP-B (class confusion) loss.
# Larger values penalize cases where the wrong class is clearly dominant more strongly.
DET_FP_CLS_MARGIN=${DET_FP_CLS_MARGIN:-0.1}

# Unified hotness weights.
# Since the goal is to avoid making TP hot, starting with TP = 0.0 is recommended.
# Adjust these ratios to emphasize FN, FP-A, and FP-B independently.
DET_HOTNESS_WEIGHT_TP=${DET_HOTNESS_WEIGHT_TP:-0.0}
DET_HOTNESS_WEIGHT_FN=${DET_HOTNESS_WEIGHT_FN:-1.0}
DET_HOTNESS_WEIGHT_FP_A=${DET_HOTNESS_WEIGHT_FP_A:-1.0}
DET_HOTNESS_WEIGHT_FP_B=${DET_HOTNESS_WEIGHT_FP_B:-1.0}

# Gate strength for g(L) = L^alpha.
# Increasing alpha suppresses low-loss contributions more aggressively and strengthens TP suppression.
DET_HOTNESS_GATE_ALPHA=${DET_HOTNESS_GATE_ALPHA:-1.0}

# Blend ratio between gradient map and spatial prior map.
# 1.0: gradient only
# 0.0: spatial prior only
# Starting around 0.4 - 0.7 is usually a reasonable choice.
DET_HOTNESS_LAMBDA=${DET_HOTNESS_LAMBDA:-0.6}

# Behavior for images with no GT annotation.
# Using fp_loc_only makes empty images behave as FP-A-dominant cases.
DET_EMPTY_GT_POLICY=fp_loc_only

# ---------------------------------------------------------------------------
# Output directory policy
# ---------------------------------------------------------------------------
# When OUTPUT_ROOT_BASE is not given from the outside, choose a default root
# based on the objective mode so that old and new experiments are separated.
if [ -z "${OUTPUT_ROOT_BASE:-}" ]; then
    if [ "$DET_OBJECTIVE_MODE" = "hotness_unified" ]; then
        OUTPUT_ROOT_BASE=results/yolox_tiny_custom_model_hotness
    else
        OUTPUT_ROOT_BASE=results/yolox_tiny_custom_model
    fi
fi

if [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
    echo "Missing checkpoint: $MODEL_WEIGHTS_PATH"
    echo "Download or place YOLOX Tiny weights at externals/YOLOX/weights/yolox_tiny.pth"
    exit 1
fi

# ---------------------------------------------------------------------------
# Input saliency mode loop
# ---------------------------------------------------------------------------
# Available values for METHOD / input_saliency_method:
#   - auto: automatically selects direct_loss for detection tasks
#   - direct_loss: directly visualizes dL/dx
#   - matching: existing PSS matching-based saliency
# When using unified hotness for detection, auto or direct_loss is recommended.
#for method in auto matching; do
for method in auto; do
    OUTPUT_ROOT="${OUTPUT_ROOT_BASE}_${method}"
    echo "Running input saliency with method=${method}, det_objective_mode=${DET_OBJECTIVE_MODE}, det_fp_loc_weight=${DET_FP_LOC_WEIGHT}"
    rm -rf "$OUTPUT_ROOT"

    docker compose run --rm \
        -e HOME=/work \
        -e XDG_CACHE_HOME=/work/.cache \
        -e TORCH_HOME=/work/.cache/torch \
        -e PYTHONPATH=/work/externals/YOLOX:/work \
        -e IMAGE_DIR="$IMAGE_DIR" \
        -e OUTPUT_ROOT="$OUTPUT_ROOT" \
        -e RUN_MODE="$RUN_MODE" \
        -e TEST_MAX_FILES="$TEST_MAX_FILES" \
        -e MODEL_KWARGS_JSON="$MODEL_KWARGS_JSON" \
        -e PREPROCESS_CFG_JSON="$PREPROCESS_CFG_JSON" \
        -e DET_OBJECTIVE_MODE="$DET_OBJECTIVE_MODE" \
        -e DET_OBJECTIVE_PROVIDER="$DET_OBJECTIVE_PROVIDER" \
        -e DET_FP_LOC_WEIGHT="$DET_FP_LOC_WEIGHT" \
        -e DET_FP_LOC_IOU_THRESHOLD="$DET_FP_LOC_IOU_THRESHOLD" \
        -e DET_FP_LOC_GATE_SHARPNESS="$DET_FP_LOC_GATE_SHARPNESS" \
        -e DET_FP_LOC_SCORE_POWER="$DET_FP_LOC_SCORE_POWER" \
        -e DET_FP_CLS_MARGIN="$DET_FP_CLS_MARGIN" \
        -e DET_HOTNESS_WEIGHT_TP="$DET_HOTNESS_WEIGHT_TP" \
        -e DET_HOTNESS_WEIGHT_FN="$DET_HOTNESS_WEIGHT_FN" \
        -e DET_HOTNESS_WEIGHT_FP_A="$DET_HOTNESS_WEIGHT_FP_A" \
        -e DET_HOTNESS_WEIGHT_FP_B="$DET_HOTNESS_WEIGHT_FP_B" \
        -e DET_HOTNESS_GATE_ALPHA="$DET_HOTNESS_GATE_ALPHA" \
        -e DET_HOTNESS_LAMBDA="$DET_HOTNESS_LAMBDA" \
        -e DET_EMPTY_GT_POLICY="$DET_EMPTY_GT_POLICY" \
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

            case "$RUN_MODE" in
                test)
                    run_total="$TEST_MAX_FILES"
                    if [ "$run_total" -gt "$total" ]; then
                        run_total="$total"
                    fi
                    echo "Run mode: test (${run_total}/${total} files)"
                    ;;
                full)
                    run_total="$total"
                    echo "Run mode: full (${run_total}/${total} files)"
                    ;;
                *)
                    echo "Unsupported RUN_MODE: $RUN_MODE"
                    echo "Supported values: test, full"
                    exit 1
                    ;;
            esac

            for ((idx = 0; idx < run_total; idx++)); do
                image_path="${image_paths[$idx]}"
                echo "[$METHOD][$RUN_MODE] ($((idx + 1))/$run_total) processing: ${image_path}"
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
                        --det_objective_mode "$DET_OBJECTIVE_MODE" \
                        --det_objective_provider "$DET_OBJECTIVE_PROVIDER" \
                    --det_fp_loc_weight "$DET_FP_LOC_WEIGHT" \
                    --det_fp_loc_iou_threshold "$DET_FP_LOC_IOU_THRESHOLD" \
                    --det_fp_loc_gate_sharpness "$DET_FP_LOC_GATE_SHARPNESS" \
                    --det_fp_loc_score_power "$DET_FP_LOC_SCORE_POWER" \
                        --det_fp_cls_margin "$DET_FP_CLS_MARGIN" \
                        --det_hotness_weight_tp "$DET_HOTNESS_WEIGHT_TP" \
                        --det_hotness_weight_fn "$DET_HOTNESS_WEIGHT_FN" \
                        --det_hotness_weight_fp_a "$DET_HOTNESS_WEIGHT_FP_A" \
                        --det_hotness_weight_fp_b "$DET_HOTNESS_WEIGHT_FP_B" \
                        --det_hotness_gate_alpha "$DET_HOTNESS_GATE_ALPHA" \
                        --det_hotness_lambda "$DET_HOTNESS_LAMBDA" \
                    --det_empty_gt_policy "$DET_EMPTY_GT_POLICY" \
                    --input_saliency_method "$METHOD" \
                    --target_type true_label
            done'
done
