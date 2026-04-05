#! /bin/bash
# run_catalog.sh
#
# カタログ生成 → 集計CSV生成 → EDA Notebook 生成 を一気通貫で実行する。
#
# 使用例:
#   bash scripts/run_catalog.sh
#
# カタログ生成をスキップして集計・Notebook 生成のみ実行する場合:
#   SKIP_CATALOG=1 bash scripts/run_catalog.sh

set -euo pipefail

RESULTS_ROOT=${RESULTS_ROOT:-results/yolox_tiny_custom_model_auto}
IMAGE_DIR=${IMAGE_DIR:-raw_images/coco2017/val2017}
OUTPUT_CSV=${OUTPUT_CSV:-${RESULTS_ROOT}/object_catalog.csv}
OUTPUT_JSON=${OUTPUT_JSON:-${RESULTS_ROOT}/object_catalog.json}
SKIP_CATALOG=${SKIP_CATALOG:-0}

DOCKER_COMMON_OPTS=(
    --rm
    -e HOME=/work
    -e XDG_CACHE_HOME=/work/.cache
    -e PYTHONPATH=/work/externals/YOLOX:/work
    -e RESULTS_ROOT="$RESULTS_ROOT"
    -e IMAGE_DIR="$IMAGE_DIR"
    -e OUTPUT_CSV="$OUTPUT_CSV"
    -e OUTPUT_JSON="$OUTPUT_JSON"
)

# ---------------------------------------------------------------------------
# Step 1: カタログ生成
# ---------------------------------------------------------------------------
if [ "$SKIP_CATALOG" = "1" ]; then
    echo "=== Step 1: Skipping catalog generation (SKIP_CATALOG=1) ==="
    if [ ! -f "$OUTPUT_JSON" ]; then
        echo "ERROR: catalog JSON not found: $OUTPUT_JSON"
        exit 1
    fi
else
    echo "=== Step 1: Generating object catalog ==="
    docker compose run "${DOCKER_COMMON_OPTS[@]}" pss \
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
fi

# ---------------------------------------------------------------------------
# Step 2: 集計CSV 生成
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Generating aggregated CSVs ==="
docker compose run "${DOCKER_COMMON_OPTS[@]}" pss \
    bash -lc 'set -euo pipefail
    python3 tools/object_analysis/precompute_method1.py'

# ---------------------------------------------------------------------------
# Step 3: EDA Notebook 生成
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Generating EDA notebook ==="
docker compose run "${DOCKER_COMMON_OPTS[@]}" pss \
    bash -lc 'set -euo pipefail
    python3 tools/object_analysis/generate_eda_notebook.py --results_root "$RESULTS_ROOT"'

echo ""
echo "=== All steps completed ==="
echo "  Catalog JSON : $OUTPUT_JSON"
echo "  Aggregates   : $RESULTS_ROOT/aggregates/"
echo "  Notebook     : $RESULTS_ROOT/reports/eda_method1.ipynb"
