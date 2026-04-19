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
MODEL_KWARGS_JSON='{"exp_path":"/work/externals/YOLOX/exps/default/yolox_tiny.py","ckpt_path":"externals/YOLOX/weights/yolox_tiny.pth"}'

# Detection annotations used by both the objective builder and auto probe selection.
DET_ANNOTATIONS_JSON=raw_images/coco2017/annotations/instances_val2017.json

# Input preprocessing configuration consumed by the adapter.
# `letterbox=true` should stay aligned with the training / inference setup.
PREPROCESS_CFG_JSON='{"resize":[416,416],"letterbox":true,"pad_value":114,"channel_order":"bgr"}'

# Run mode configuration.
# Available values:
#   - test: process only the first TEST_MAX_FILES images after sorting
#   - full: process all discovered images
RUN_MODE=${RUN_MODE:-test}
TEST_MAX_FILES=${TEST_MAX_FILES:-10}

# Resume control.
# 1: keep existing OUTPUT_ROOT and resume unfinished images only.
# 0: start fresh by deleting OUTPUT_ROOT.
RESUME_ENABLED=${RESUME_ENABLED:-1}

# Force reset when RESUME_ENABLED=1.
# 1: clear OUTPUT_ROOT once before run, then behave as resume-enabled.
# 0: keep existing results and resume from checkpoints.
RESUME_RESET=${RESUME_RESET:-0}

# Number of images to process in parallel inside the container.
# Accepted values:
#   - <integer> : use this value directly (e.g. 1 = sequential, 2 = two workers)
# Dynamic auto scheduling has been removed from this script.
PARALLEL_JOBS=${PARALLEL_JOBS:-1}

# Poll interval (seconds) while waiting for a worker slot in fixed parallel mode.
AUTO_LAUNCH_POLL_SEC=${AUTO_LAUNCH_POLL_SEC:-0.2}

# Failure diagnostics for parallel workers.
# 1: print failed image path, exit code, and per-image log tail
# 0: keep warning output minimal
DEBUG_FAILURE_DIAG=${DEBUG_FAILURE_DIAG:-1}

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

# Disable gated component maps when set to 1.
# 0: export gradients/images for tp_gated/fn_gated/fp_a_gated/fp_b_gated (legacy behavior)
# 1: skip gated component gradients and gated PNG exports
DET_DISABLE_GATED_COMPONENTS=${DET_DISABLE_GATED_COMPONENTS:-1}

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
    if [ "$RESUME_ENABLED" = "1" ]; then
        if [ "$RESUME_RESET" = "1" ]; then
            echo "Resume mode: reset requested, clearing OUTPUT_ROOT=${OUTPUT_ROOT}"
            rm -rf "$OUTPUT_ROOT"
        else
            echo "Resume mode: enabled, keeping existing OUTPUT_ROOT=${OUTPUT_ROOT}"
        fi
        mkdir -p "$OUTPUT_ROOT"
    else
        echo "Resume mode: disabled, starting from clean OUTPUT_ROOT=${OUTPUT_ROOT}"
        rm -rf "$OUTPUT_ROOT"
        mkdir -p "$OUTPUT_ROOT"
    fi

    docker compose run -T --rm \
        -e HOME=/work \
        -e XDG_CACHE_HOME=/work/.cache \
        -e TORCH_HOME=/work/.cache/torch \
        -e PYTHONPATH=/work/externals/YOLOX:/work \
        -e IMAGE_DIR="$IMAGE_DIR" \
        -e OUTPUT_ROOT="$OUTPUT_ROOT" \
        -e RUN_MODE="$RUN_MODE" \
        -e TEST_MAX_FILES="$TEST_MAX_FILES" \
        -e RESUME_ENABLED="$RESUME_ENABLED" \
        -e PARALLEL_JOBS="$PARALLEL_JOBS" \
        -e AUTO_LAUNCH_POLL_SEC="$AUTO_LAUNCH_POLL_SEC" \
        -e DEBUG_FAILURE_DIAG="$DEBUG_FAILURE_DIAG" \
        -e MODEL_KWARGS_JSON="$MODEL_KWARGS_JSON" \
        -e PREPROCESS_CFG_JSON="$PREPROCESS_CFG_JSON" \
        -e DET_OBJECTIVE_MODE="$DET_OBJECTIVE_MODE" \
        -e DET_OBJECTIVE_PROVIDER="$DET_OBJECTIVE_PROVIDER" \
        -e DET_ANNOTATIONS_JSON="$DET_ANNOTATIONS_JSON" \
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
        -e DET_DISABLE_GATED_COMPONENTS="$DET_DISABLE_GATED_COMPONENTS" \
        -e DET_EMPTY_GT_POLICY="$DET_EMPTY_GT_POLICY" \
        -e METHOD="$method" \
        pss \
            bash -s -- <<'BASH'
            set -euo pipefail

            active_pids=()

            terminate_all_jobs() {
                local sig="${1:-TERM}"
                for pid in "${active_pids[@]:-}"; do
                    kill -"$sig" "$pid" 2>/dev/null || true
                done
            }

            on_interrupt() {
                echo "[INFO] interrupt received; stopping running jobs..."
                terminate_all_jobs TERM
                terminate_all_jobs KILL
                exit 130
            }

            trap on_interrupt INT TERM

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

            checkpoint_dir="$OUTPUT_ROOT/.resume_state"
            mkdir -p "$checkpoint_dir"

            pending_indices=()
            for ((idx = 0; idx < run_total; idx++)); do
                idx_padded=$(printf "%08d" "$idx")
                marker_file="$checkpoint_dir/${idx_padded}.done"
                if [ "$RESUME_ENABLED" = "1" ] && [ -f "$marker_file" ]; then
                    continue
                fi
                pending_indices+=("$idx")
            done

            pending_total=${#pending_indices[@]}
            skipped_total=$((run_total - pending_total))
            echo "Resume scan: total=${run_total}, pending=${pending_total}, skipped=${skipped_total}"

            csv_file="$OUTPUT_ROOT/timing_runtime.csv"
            if [ ! -f "$csv_file" ]; then
                echo "record_type,timestamp,method,run_mode,output_root,image_index,image_count,image_path,elapsed_ms,total_elapsed_ms,avg_elapsed_ms,det_objective_mode,input_saliency_method" > "$csv_file"
            fi

            csv_escape() {
                local v="$1"
                v="${v//\"/\"\"}"
                printf '"%s"' "$v"
            }

            # ---------------------------------------------------------------------------
            # Helper: run parameter_and_input_saliency.py for a single image.
            # Usage: run_one_image <image_path>
            # ---------------------------------------------------------------------------
            run_one_image() {
                local img="$1"
                local gated_flag=()
                if [ "${DET_DISABLE_GATED_COMPONENTS:-0}" = "1" ]; then
                    gated_flag+=(--det_disable_gated_components)
                fi
                python3 parameter_and_input_saliency.py \
                    --task detection \
                    --model_source custom_module \
                    --model_import_root /work/externals/YOLOX \
                    --model_class_path yolox.models.build.yolox_custom \
                    --model_kwargs_json "$MODEL_KWARGS_JSON" \
                    --preprocess_cfg_json "$PREPROCESS_CFG_JSON" \
                    --image_path "$img" \
                    --det_annotations_json "$DET_ANNOTATIONS_JSON" \
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
                        "${gated_flag[@]}" \
                    --det_empty_gt_policy "$DET_EMPTY_GT_POLICY" \
                    --input_saliency_method "$METHOD" \
                    --target_type true_label
            }

            # Temp directory for per-image CSV records; merged in order after all jobs complete.
            tmp_dir=$(mktemp -d)
            debug_dir="$OUTPUT_ROOT/.debug_logs"
            mkdir -p "$debug_dir"

            # Keep pid-to-image mapping so failed workers can be identified.
            active_job_images=()
            active_job_indices=()

            # Wait for all tracked jobs and clear the list.
            # Returns non-zero when any job fails or is interrupted.
            flush_active_pids() {
                local status=0
                local i pid image_path idx_padded job_log wait_status
                for i in "${!active_pids[@]}"; do
                    pid="${active_pids[$i]}"
                    image_path="${active_job_images[$i]}"
                    idx_padded="${active_job_indices[$i]}"
                    job_log="$debug_dir/${idx_padded}.job.log"
                    if wait "$pid"; then
                        continue
                    else
                        wait_status=$?
                        [ "$status" -eq 0 ] && status="$wait_status"
                        echo "[WARN] background job failed: pid=$pid exit_code=$wait_status image=$image_path log=$job_log"
                        if [ "${DEBUG_FAILURE_DIAG:-1}" = "1" ]; then
                            if [ -f "$job_log" ]; then
                                echo "[WARN] ---- last 40 lines of failed job log (image=$image_path) ----"
                                tail -n 40 "$job_log" || true
                                echo "[WARN] -----------------------------------------------------------"
                            fi
                            if [ "$wait_status" -eq 137 ] || [ "$wait_status" -eq 143 ]; then
                                if command -v nvidia-smi > /dev/null 2>&1; then
                                    echo "[WARN] GPU memory snapshot around failure:"
                                    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
                                fi
                            fi
                        fi
                    fi
                done
                active_pids=()
                active_job_images=()
                active_job_indices=()
                return "$status"
            }

            # Reap finished jobs without blocking running workers.
            # Returns non-zero when any finished job failed.
            reap_finished_pids() {
                local status=0
                local i pid image_path idx_padded job_log wait_status
                local next_pids=()
                local next_images=()
                local next_indices=()

                for i in "${!active_pids[@]}"; do
                    pid="${active_pids[$i]}"
                    image_path="${active_job_images[$i]}"
                    idx_padded="${active_job_indices[$i]}"
                    job_log="$debug_dir/${idx_padded}.job.log"

                    if kill -0 "$pid" 2>/dev/null; then
                        next_pids+=("$pid")
                        next_images+=("$image_path")
                        next_indices+=("$idx_padded")
                        continue
                    fi

                    if wait "$pid"; then
                        continue
                    else
                        wait_status=$?
                        [ "$status" -eq 0 ] && status="$wait_status"
                        echo "[WARN] background job failed: pid=$pid exit_code=$wait_status image=$image_path log=$job_log"
                        if [ "${DEBUG_FAILURE_DIAG:-1}" = "1" ]; then
                            if [ -f "$job_log" ]; then
                                echo "[WARN] ---- last 40 lines of failed job log (image=$image_path) ----"
                                tail -n 40 "$job_log" || true
                                echo "[WARN] -----------------------------------------------------------"
                            fi
                            if [ "$wait_status" -eq 137 ] || [ "$wait_status" -eq 143 ]; then
                                if command -v nvidia-smi > /dev/null 2>&1; then
                                    echo "[WARN] GPU memory snapshot around failure:"
                                    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
                                fi
                            fi
                        fi
                    fi
                done

                active_pids=("${next_pids[@]}")
                active_job_images=("${next_images[@]}")
                active_job_indices=("${next_indices[@]}")
                return "$status"
            }
            if [[ "$PARALLEL_JOBS" == "auto" ]]; then
                echo "[WARN] PARALLEL_JOBS=auto is no longer supported; falling back to 1"
                PARALLEL_JOBS=1
            fi
            if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$PARALLEL_JOBS" -le 0 ]; then
                echo "Invalid PARALLEL_JOBS: $PARALLEL_JOBS"
                echo "Set PARALLEL_JOBS to a positive integer."
                exit 1
            fi

            run_start_ms=$(date +%s%3N)

            for ((pos = 0; pos < pending_total; pos++)); do
                idx="${pending_indices[$pos]}"
                image_path="${image_paths[$idx]}"
                idx_padded=$(printf "%08d" "$idx")

                while [ "${#active_pids[@]}" -ge "$PARALLEL_JOBS" ]; do
                    reap_finished_pids || exit $?
                    if [ "${#active_pids[@]}" -ge "$PARALLEL_JOBS" ]; then
                        sleep "$AUTO_LAUNCH_POLL_SEC"
                    fi
                done

                launch_ts=$(date -Iseconds)
                echo "[$METHOD][$RUN_MODE][$launch_ts] launching job idx=$((idx + 1))/$run_total path=${image_path} active_before=${#active_pids[@]}"

                (
                    job_log="$debug_dir/${idx_padded}.job.log"
                    {
                        echo "[$METHOD][$RUN_MODE] ($((idx + 1))/$run_total) processing: ${image_path}"
                        image_start_ms=$(date +%s%3N)
                        run_one_image "$image_path"
                        image_end_ms=$(date +%s%3N)
                        image_elapsed_ms=$((image_end_ms - image_start_ms))
                        printf "%s\n" "$image_path" > "$checkpoint_dir/${idx_padded}.done"
                        image_ts=$(date -Iseconds)
                        echo "[$METHOD][$RUN_MODE] ($((idx + 1))/$run_total) elapsed: ${image_elapsed_ms} ms"
                        printf "%s,%s,%s,%s,%s,%d,%d,%s,%d,%s,%s,%s,%s\n" \
                            "per_image" \
                            "$(csv_escape "$image_ts")" \
                            "$(csv_escape "$METHOD")" \
                            "$(csv_escape "$RUN_MODE")" \
                            "$(csv_escape "$OUTPUT_ROOT")" \
                            $((idx + 1)) \
                            "$run_total" \
                            "$(csv_escape "$image_path")" \
                            "$image_elapsed_ms" \
                            "" \
                            "" \
                            "$(csv_escape "$DET_OBJECTIVE_MODE")" \
                            "$(csv_escape "$METHOD")" \
                            > "$tmp_dir/${idx_padded}.csv"
                    } > "$job_log" 2>&1
                ) &
                active_pids+=("$!")
                active_job_images+=("$image_path")
                active_job_indices+=("$idx_padded")
            done
            flush_active_pids || exit $?

            # Merge per-image records into the CSV file in index order.
            for ((idx = 0; idx < run_total; idx++)); do
                idx_padded=$(printf "%08d" "$idx")
                tmp_file="$tmp_dir/${idx_padded}.csv"
                [ -f "$tmp_file" ] && cat "$tmp_file" >> "$csv_file"
            done
            rm -rf "$tmp_dir"

            run_end_ms=$(date +%s%3N)
            total_elapsed_ms=$((run_end_ms - run_start_ms))
            if [ "$run_total" -gt 0 ]; then
                avg_elapsed_ms=$((total_elapsed_ms / run_total))
            else
                avg_elapsed_ms=0
            fi
            run_ts=$(date -Iseconds)
            echo "[$METHOD][$RUN_MODE] summary: images=${run_total}, parallel_jobs=${PARALLEL_JOBS}, total_elapsed_ms=${total_elapsed_ms}, avg_elapsed_ms=${avg_elapsed_ms}"
            printf "%s,%s,%s,%s,%s,%s,%d,%s,%s,%d,%d,%s,%s\n" \
                "summary" \
                "$(csv_escape "$run_ts")" \
                "$(csv_escape "$METHOD")" \
                "$(csv_escape "$RUN_MODE")" \
                "$(csv_escape "$OUTPUT_ROOT")" \
                "" \
                "$run_total" \
                "" \
                "" \
                "$total_elapsed_ms" \
                "$avg_elapsed_ms" \
                "$(csv_escape "$DET_OBJECTIVE_MODE")" \
                "$(csv_escape "$METHOD")" \
                >> "$csv_file"
BASH
done
