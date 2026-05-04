#! /bin/bash

# ---------------------------------------------------------------------------
# Stage3: Statistical analysis of normalized filter saliency
#
# Uses parameter_and_input_saliency.py (--pipeline_stage stage3) and produces
# per-type histograms and summary_stats.csv under {RUN_DIR}/stage3_stats/.
# ---------------------------------------------------------------------------

RUN_DIR=${RUN_DIR:-results/yolox_tiny_pss_paper}
HIST_BINS=${HIST_BINS:-50}
HIST_CLIP=${HIST_CLIP:-10.0}

docker compose run -T --rm \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    -e TORCH_HOME=/work/.cache/torch \
    -e PYTHONPATH=/work/externals/YOLOX:/work \
    -e RUN_DIR="$RUN_DIR" \
    -e HIST_BINS="$HIST_BINS" \
    -e HIST_CLIP="$HIST_CLIP" \
    pss \
        bash -s -- <<'BASH'
        set -euo pipefail
        echo "[Stage3] run_dir=${RUN_DIR}"

        python3 parameter_and_input_saliency.py \
            --pipeline_stage stage3 \
            --run_dir "$RUN_DIR" \
            --stage3_hist_bins "$HIST_BINS" \
            --stage3_hist_clip "$HIST_CLIP"

        echo "[Stage3] Done. Results in: ${RUN_DIR}/stage3_stats/"
BASH
