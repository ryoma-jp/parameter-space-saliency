#! /bin/bash

set -euo pipefail

RESULTS_ROOT=${RESULTS_ROOT:-results/yolox_tiny_custom_model_auto}
IMAGE_DIR=${IMAGE_DIR:-raw_images/coco2017/val2017}
OUTPUT_CSV=${OUTPUT_CSV:-${RESULTS_ROOT}/object_catalog.csv}
OUTPUT_JSON=${OUTPUT_JSON:-${RESULTS_ROOT}/object_catalog.json}

docker compose run --rm -u $(id -u):$(id -g) \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    -e PYTHONPATH=/work/externals/YOLOX:/work \
    -e RESULTS_ROOT="$RESULTS_ROOT" \
    -e IMAGE_DIR="$IMAGE_DIR" \
    -e OUTPUT_CSV="$OUTPUT_CSV" \
    -e OUTPUT_JSON="$OUTPUT_JSON" \
    pss \
    bash -lc 'set -euo pipefail

    if [ ! -d "$RESULTS_ROOT" ]; then
        echo "Missing results root: $RESULTS_ROOT"
        exit 1
    fi

    if [ ! -d "$IMAGE_DIR" ]; then
        echo "Missing image directory: $IMAGE_DIR"
        exit 1
    fi

    python3 tools/detection_feature_analysis/run_catalog.py \
        --results_root "$RESULTS_ROOT" \
        --image_dir "$IMAGE_DIR" \
        --output_csv "$OUTPUT_CSV" \
        --output_json "$OUTPUT_JSON"

    echo "Saved catalog CSV: $OUTPUT_CSV"
    echo "Saved catalog JSON: $OUTPUT_JSON"'
