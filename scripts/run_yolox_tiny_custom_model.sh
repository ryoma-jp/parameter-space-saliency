#! /bin/bash

# ---------------------------------------------------------------------------
# End-to-end PSS pipeline: Stage1 -> Stage2 -> Stage3
#
# Stage1: per-object filter saliency computation
# Stage2: TP-based normalization + optional input-space projection
# Stage3: statistical analysis (histograms + summary_stats.csv)
#
# Override any variable from the environment before calling this script,
# e.g.:  RUN_DIR=results/my_run RUN_MODE=full bash scripts/run_compare_npy_yolox_vs_pss_feature.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Common settings (forwarded to all stage scripts)
# ---------------------------------------------------------------------------
export MODEL_WEIGHTS_PATH=${MODEL_WEIGHTS_PATH:-externals/YOLOX/weights/yolox_tiny.pth}
export IMAGE_DIR=${IMAGE_DIR:-raw_images/coco2017/val2017}
export DET_ANNOTATIONS_JSON=${DET_ANNOTATIONS_JSON:-raw_images/coco2017/annotations/instances_val2017.json}
export RUN_DIR=${RUN_DIR:-results/yolox_tiny_pss_paper}
export RUN_MODE=${RUN_MODE:-test}
export TEST_MAX_FILES=${TEST_MAX_FILES:-10}
export RESUME_ENABLED=${RESUME_ENABLED:-0}
export RESUME_RESET=${RESUME_RESET:-0}
export PARALLEL_JOBS=${PARALLEL_JOBS:-1}

# Stage2 projection settings
export NO_PROJECTION=${NO_PROJECTION:-0}
export TOP_F=${TOP_F:-10}
export BOOST_K=${BOOST_K:-100.0}

# Stage3 histogram settings
export HIST_BINS=${HIST_BINS:-50}
export HIST_CLIP=${HIST_CLIP:-10.0}

echo "============================================================"
echo " PSS Pipeline: Stage1 -> Stage2 -> Stage3"
echo " RUN_DIR    : $RUN_DIR"
echo " IMAGE_DIR  : $IMAGE_DIR"
echo " RUN_MODE   : $RUN_MODE"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: per-object filter saliency
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 1: filter saliency ---"
bash scripts/run_stage1_filter_saliency.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Stage 1 failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2: TP normalization + projection
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 2: normalize + project ---"
bash scripts/run_stage2_normalize_project.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Stage 2 failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 3: statistics
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 3: statistics ---"
bash scripts/run_stage3_statistics.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Stage 3 failed"
    exit 1
fi

echo ""
echo "============================================================"
echo " Pipeline complete. Results in: $RUN_DIR"
echo "   Stage3 stats: $RUN_DIR/stage3_stats/"
echo "============================================================"
