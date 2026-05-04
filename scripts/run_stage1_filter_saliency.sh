#! /bin/bash

# ---------------------------------------------------------------------------
# Stage1: Per-object filter saliency computation
#
# Calls parameter_and_input_saliency.py (--pipeline_stage stage1) for each image and saves
#   {RUN_DIR}/{image_stem}/stage1_filter_saliency.pth
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Base model / dataset settings (same as run_yolox_tiny_custom_model.sh)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS_PATH=externals/YOLOX/weights/yolox_tiny.pth
IMAGE_DIR=${IMAGE_DIR:-raw_images/coco2017/val2017}
MODEL_KWARGS_JSON='{"exp_path":"/work/externals/YOLOX/exps/default/yolox_tiny.py","ckpt_path":"externals/YOLOX/weights/yolox_tiny.pth"}'
DET_ANNOTATIONS_JSON=${DET_ANNOTATIONS_JSON:-raw_images/coco2017/annotations/instances_val2017.json}
PREPROCESS_CFG_JSON='{"resize":[416,416],"letterbox":true,"pad_value":114,"channel_order":"bgr"}'

RUN_DIR=${RUN_DIR:-results/yolox_tiny_pss_paper}
RUN_MODE=${RUN_MODE:-test}
TEST_MAX_FILES=${TEST_MAX_FILES:-10}
RESUME_ENABLED=${RESUME_ENABLED:-1}
RESUME_RESET=${RESUME_RESET:-0}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}
AUTO_LAUNCH_POLL_SEC=${AUTO_LAUNCH_POLL_SEC:-0.2}
DEBUG_FAILURE_DIAG=${DEBUG_FAILURE_DIAG:-1}

# Detection thresholds
DET_CONF_THRESHOLD=${DET_CONF_THRESHOLD:-0.3}
DET_NMS_IOU_THRESHOLD=${DET_NMS_IOU_THRESHOLD:-0.45}
DET_MATCH_IOU_THRESHOLD=${DET_MATCH_IOU_THRESHOLD:-0.5}
DET_FN_TAU=${DET_FN_TAU:-0.1}

if [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
    echo "Missing checkpoint: $MODEL_WEIGHTS_PATH"
    exit 1
fi

if [ "$RESUME_ENABLED" = "1" ]; then
    if [ "$RESUME_RESET" = "1" ]; then
        echo "Resume mode: reset requested, clearing RUN_DIR=${RUN_DIR}"
        rm -rf "$RUN_DIR"
    else
        echo "Resume mode: enabled, keeping existing RUN_DIR=${RUN_DIR}"
    fi
    mkdir -p "$RUN_DIR"
else
    echo "Resume mode: disabled, starting from clean RUN_DIR=${RUN_DIR}"
    rm -rf "$RUN_DIR"
    mkdir -p "$RUN_DIR"
fi

docker compose run -T --rm \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    -e TORCH_HOME=/work/.cache/torch \
    -e PYTHONPATH=/work/externals/YOLOX:/work \
    -e IMAGE_DIR="$IMAGE_DIR" \
    -e RUN_DIR="$RUN_DIR" \
    -e RUN_MODE="$RUN_MODE" \
    -e TEST_MAX_FILES="$TEST_MAX_FILES" \
    -e RESUME_ENABLED="$RESUME_ENABLED" \
    -e PARALLEL_JOBS="$PARALLEL_JOBS" \
    -e AUTO_LAUNCH_POLL_SEC="$AUTO_LAUNCH_POLL_SEC" \
    -e DEBUG_FAILURE_DIAG="$DEBUG_FAILURE_DIAG" \
    -e MODEL_KWARGS_JSON="$MODEL_KWARGS_JSON" \
    -e PREPROCESS_CFG_JSON="$PREPROCESS_CFG_JSON" \
    -e DET_ANNOTATIONS_JSON="$DET_ANNOTATIONS_JSON" \
    -e DET_CONF_THRESHOLD="$DET_CONF_THRESHOLD" \
    -e DET_NMS_IOU_THRESHOLD="$DET_NMS_IOU_THRESHOLD" \
    -e DET_MATCH_IOU_THRESHOLD="$DET_MATCH_IOU_THRESHOLD" \
    -e DET_FN_TAU="$DET_FN_TAU" \
    pss \
        bash -s -- <<'BASH'
        set -euo pipefail

        active_pids=()
        active_job_images=()
        active_job_indices=()

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
            echo "No supported image files found under: $IMAGE_DIR"
            exit 1
        fi

        case "$RUN_MODE" in
            test)
                run_total="$TEST_MAX_FILES"
                [ "$run_total" -gt "$total" ] && run_total="$total"
                echo "Run mode: test (${run_total}/${total} files)"
                ;;
            full)
                run_total="$total"
                echo "Run mode: full (${run_total}/${total} files)"
                ;;
            *)
                echo "Unsupported RUN_MODE: $RUN_MODE"; exit 1 ;;
        esac

        checkpoint_dir="$RUN_DIR/.resume_state"
        mkdir -p "$checkpoint_dir"

        pending_indices=()
        for ((idx = 0; idx < run_total; idx++)); do
            idx_padded=$(printf "%08d" "$idx")
            marker_file="$checkpoint_dir/${idx_padded}.stage1.done"
            if [ "$RESUME_ENABLED" = "1" ] && [ -f "$marker_file" ]; then
                continue
            fi
            pending_indices+=("$idx")
        done

        pending_total=${#pending_indices[@]}
        skipped_total=$((run_total - pending_total))
        echo "Stage1 scan: total=${run_total}, pending=${pending_total}, skipped=${skipped_total}"

        debug_dir="$RUN_DIR/.debug_logs"
        mkdir -p "$debug_dir"

        flush_active_pids() {
            local status=0
            local i pid image_path idx_padded job_log wait_status
            for i in "${!active_pids[@]}"; do
                pid="${active_pids[$i]}"
                image_path="${active_job_images[$i]}"
                idx_padded="${active_job_indices[$i]}"
                job_log="$debug_dir/${idx_padded}.stage1.log"
                if wait "$pid"; then
                    continue
                else
                    wait_status=$?
                    [ "$status" -eq 0 ] && status="$wait_status"
                    echo "[WARN] job failed: pid=$pid exit=$wait_status image=$image_path"
                    if [ "${DEBUG_FAILURE_DIAG:-1}" = "1" ] && [ -f "$job_log" ]; then
                        echo "[WARN] --- last 40 lines ---"
                        tail -n 40 "$job_log" || true
                        echo "[WARN] ---"
                    fi
                fi
            done
            active_pids=()
            active_job_images=()
            active_job_indices=()
            return "$status"
        }

        reap_finished_pids() {
            local status=0
            local i pid image_path idx_padded job_log wait_status
            local next_pids=() next_images=() next_indices=()
            for i in "${!active_pids[@]}"; do
                pid="${active_pids[$i]}"
                image_path="${active_job_images[$i]}"
                idx_padded="${active_job_indices[$i]}"
                job_log="$debug_dir/${idx_padded}.stage1.log"
                if kill -0 "$pid" 2>/dev/null; then
                    next_pids+=("$pid")
                    next_images+=("$image_path")
                    next_indices+=("$idx_padded")
                else
                    if wait "$pid"; then
                        :
                    else
                        wait_status=$?
                        [ "$status" -eq 0 ] && status="$wait_status"
                        echo "[WARN] job failed: pid=$pid exit=$wait_status image=$image_path"
                        if [ "${DEBUG_FAILURE_DIAG:-1}" = "1" ] && [ -f "$job_log" ]; then
                            echo "[WARN] --- last 40 lines ---"
                            tail -n 40 "$job_log" || true
                            echo "[WARN] ---"
                        fi
                    fi
                fi
            done
            active_pids=("${next_pids[@]}")
            active_job_images=("${next_images[@]}")
            active_job_indices=("${next_indices[@]}")
            return "$status"
        }

        run_one_image() {
            local img="$1"
            python3 parameter_and_input_saliency.py \
                --pipeline_stage stage1 \
                --image_path "$img" \
                --run_dir "$RUN_DIR" \
                --model_source custom_module \
                --model_import_root /work/externals/YOLOX \
                --model_class_path yolox.models.build.yolox_custom \
                --model_kwargs_json "$MODEL_KWARGS_JSON" \
                --preprocess_cfg_json "$PREPROCESS_CFG_JSON" \
                --det_annotations_json "$DET_ANNOTATIONS_JSON" \
                --det_conf_threshold "$DET_CONF_THRESHOLD" \
                --det_nms_iou_threshold "$DET_NMS_IOU_THRESHOLD" \
                --det_match_iou_threshold "$DET_MATCH_IOU_THRESHOLD" \
                --det_fn_tau "$DET_FN_TAU"
        }

        overall_status=0
        for pending_pos in "${!pending_indices[@]}"; do
            idx="${pending_indices[$pending_pos]}"
            img="${image_paths[$idx]}"
            idx_padded=$(printf "%08d" "$idx")
            marker_file="$checkpoint_dir/${idx_padded}.stage1.done"
            job_log="$debug_dir/${idx_padded}.stage1.log"

            # Wait for a free slot
            while [ "${#active_pids[@]}" -ge "$PARALLEL_JOBS" ]; do
                sleep "$AUTO_LAUNCH_POLL_SEC"
                reap_finished_pids || overall_status=$?
            done

            echo "[$((pending_pos+1))/${pending_total}] Stage1: $img"
            run_one_image "$img" >"$job_log" 2>&1 && touch "$marker_file" &
            active_pids+=("$!")
            active_job_images+=("$img")
            active_job_indices+=("$idx_padded")
        done

        flush_active_pids || overall_status=$?

        if [ "$overall_status" -ne 0 ]; then
            echo "[Stage1] Some images failed (exit=$overall_status)"
            exit "$overall_status"
        fi
        echo "[Stage1] Done. Results in: $RUN_DIR"
BASH
