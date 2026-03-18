#! /bin/bash

OUTPUT_DIR=results/compare_npy_yolox_vs_pss/backbone_s8
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

docker compose run --rm -u $(id -u):$(id -g) \
    -e HOME=/work \
    -e XDG_CACHE_HOME=/work/.cache \
    pss \
    bash -lc "python3 tools/compare_npy/compare_npy.py \
    externals/YOLOX/YOLOX_outputs/yolox_tiny/eval/vis_res/397133/npy/feat_backbone_s8.npy \
    results/yolox_tiny_custom_model/000000397133/npy/feat_backbone_s8.npy \
    --output \"$OUTPUT_DIR\""
