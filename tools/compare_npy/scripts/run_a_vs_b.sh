#! /bin/bash

OUTPUT_DIR=tools/compare_npy/results_a_vs_b
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

python3 tools/compare_npy/compare_npy.py \
    tools/compare_npy/sample_data/sample_a.npy \
    tools/compare_npy/sample_data/sample_b_match.npy \
    --output "$OUTPUT_DIR"
