#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two .npy files and write the result to a text file."
    )
    parser.add_argument("source_npy", help="Path to the source .npy file")
    parser.add_argument("target_npy", help="Path to the target .npy file")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output directory (result.txt and diff.csv will be written here)",
    )
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def arrays_match(source: np.ndarray, target: np.ndarray) -> bool:
    if source.shape != target.shape:
        return False

    if source.dtype != target.dtype:
        return False

    if np.issubdtype(source.dtype, np.inexact):
        return np.array_equal(source, target, equal_nan=True)

    return np.array_equal(source, target)


def build_result_text(
    source_path: Path,
    target_path: Path,
    source_array: np.ndarray,
    target_array: np.ndarray,
) -> str:
    lines = [
        f"source: {source_path}",
        f"target: {target_path}",
        f"source shape: {source_array.shape}  dtype: {source_array.dtype}",
        f"target shape: {target_array.shape}  dtype: {target_array.dtype}",
    ]
    if arrays_match(source_array, target_array):
        lines.append("Comparison result: Match")
    else:
        lines.append("Comparison result: Mismatch")
        if source_array.shape != target_array.shape or source_array.dtype != target_array.dtype:
            lines.append("(diff.csv not generated: shape or dtype mismatch)")
    return "\n".join(lines) + "\n"


def write_sparse_diff_csv(output_path: Path, source_array: np.ndarray, target_array: np.ndarray) -> None:
    src_flat = source_array.flatten()
    tgt_flat = target_array.flatten()
    mismatch_mask = src_flat != tgt_flat
    if np.issubdtype(source_array.dtype, np.inexact):
        nan_equal = np.isnan(src_flat) & np.isnan(tgt_flat)
        mismatch_mask = mismatch_mask & ~nan_equal
    indices = np.argwhere(mismatch_mask).flatten()
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "source_value", "target_value", "abs_diff"])
        for flat_idx in indices:
            nd_idx = np.unravel_index(flat_idx, source_array.shape)
            s_val = src_flat[flat_idx]
            t_val = tgt_flat[flat_idx]
            abs_diff = abs(float(t_val) - float(s_val))
            writer.writerow([str(nd_idx), s_val, t_val, abs_diff])


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_npy)
    target_path = Path(args.target_npy)

    output_dir = Path(args.output)

    try:
        source_array = load_array(source_path)
        target_array = load_array(target_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_text = build_result_text(source_path, target_path, source_array, target_array)
        (output_dir / "result.txt").write_text(result_text, encoding="utf-8")
        if (
            not arrays_match(source_array, target_array)
            and source_array.shape == target_array.shape
            and source_array.dtype == target_array.dtype
        ):
            write_sparse_diff_csv(output_dir / "diff.csv", source_array, target_array)
    except Exception as exc:  # pragma: no cover - minimal CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())