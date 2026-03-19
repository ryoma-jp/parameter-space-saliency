#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


RELATIVE_ERROR_FLOOR_RATIO = 1e-6
ABSOLUTE_ERROR_FLOOR = 1e-12


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


def can_compute_error_stats(source: np.ndarray, target: np.ndarray) -> bool:
    if source.shape != target.shape:
        return False

    return np.issubdtype(source.dtype, np.number) and np.issubdtype(target.dtype, np.number)


def compute_absolute_error_stats(source: np.ndarray, target: np.ndarray) -> dict[str, float] | None:
    if not can_compute_error_stats(source, target):
        return None

    source_float = source.astype(np.float64, copy=False)
    target_float = target.astype(np.float64, copy=False)
    equal_mask = source == target

    if np.issubdtype(source.dtype, np.inexact) or np.issubdtype(target.dtype, np.inexact):
        equal_mask = equal_mask | (np.isnan(source_float) & np.isnan(target_float))

    abs_error = np.zeros(source.shape, dtype=np.float64)
    mismatch_mask = ~equal_mask
    abs_error[mismatch_mask] = np.abs(target_float[mismatch_mask] - source_float[mismatch_mask])

    finite_errors = abs_error[np.isfinite(abs_error)]
    if finite_errors.size == 0:
        return None

    return {
        "max": float(np.max(finite_errors)),
        "min": float(np.min(finite_errors)),
        "mean": float(np.mean(finite_errors)),
        "std": float(np.std(finite_errors)),
        "median": float(np.median(finite_errors)),
    }


def _compute_relative_error_floor(source_float: np.ndarray) -> float:
    source_abs = np.abs(source_float)
    finite_abs = source_abs[np.isfinite(source_abs)]
    if finite_abs.size == 0:
        return ABSOLUTE_ERROR_FLOOR

    non_zero_abs = finite_abs[finite_abs > 0.0]
    if non_zero_abs.size == 0:
        return ABSOLUTE_ERROR_FLOOR

    source_ref_scale = float(np.median(non_zero_abs))
    return max(RELATIVE_ERROR_FLOOR_RATIO * source_ref_scale, ABSOLUTE_ERROR_FLOOR)


def _build_percentile_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)),
    }


def compute_relative_error_stats(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]] | tuple[None, None]:
    if not can_compute_error_stats(source, target):
        return None, None

    source_float = source.astype(np.float64, copy=False)
    target_float = target.astype(np.float64, copy=False)
    floor = _compute_relative_error_floor(source_float)

    source_abs = np.abs(source_float)
    target_abs = np.abs(target_float)
    # Symmetric denominator suppresses one-sided blow-ups near zero.
    denominator = np.maximum(np.maximum(source_abs, target_abs), floor)

    equal_mask = source == target
    if np.issubdtype(source.dtype, np.inexact) or np.issubdtype(target.dtype, np.inexact):
        equal_mask = equal_mask | (np.isnan(source_float) & np.isnan(target_float))

    abs_error = np.zeros(source.shape, dtype=np.float64)
    mismatch_mask = ~equal_mask
    abs_error[mismatch_mask] = np.abs(target_float[mismatch_mask] - source_float[mismatch_mask])

    relative_error = abs_error / denominator
    finite_relative_error = relative_error[np.isfinite(relative_error)]
    if finite_relative_error.size == 0:
        return None, None

    floor_applied_count = int(np.count_nonzero(source_abs < floor))
    total_count = int(source_abs.size)
    metadata = {
        "floor": float(floor),
        "reference_scale": float(floor / RELATIVE_ERROR_FLOOR_RATIO)
        if RELATIVE_ERROR_FLOOR_RATIO > 0.0
        else 0.0,
        "floor_ratio": float(RELATIVE_ERROR_FLOOR_RATIO),
        "floor_applied_count": float(floor_applied_count),
        "total_count": float(total_count),
        "floor_applied_fraction": float(floor_applied_count / total_count) if total_count > 0 else 0.0,
    }

    stats = {
        "max": float(np.max(finite_relative_error)),
        "min": float(np.min(finite_relative_error)),
        "mean": float(np.mean(finite_relative_error)),
        "std": float(np.std(finite_relative_error)),
        "median": float(np.median(finite_relative_error)),
    }
    stats.update(_build_percentile_stats(finite_relative_error))
    return stats, metadata


def build_result_text(
    source_path: Path,
    target_path: Path,
    source_array: np.ndarray,
    target_array: np.ndarray,
) -> str:
    error_stats = compute_absolute_error_stats(source_array, target_array)
    relative_error_stats, relative_error_meta = compute_relative_error_stats(source_array, target_array)
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

    if error_stats is None:
        lines.append("Absolute error stats: not available")
    else:
        lines.append("Absolute error stats:")
        lines.append(f"  max: {error_stats['max']}")
        lines.append(f"  min: {error_stats['min']}")
        lines.append(f"  mean: {error_stats['mean']}")
        lines.append(f"  std: {error_stats['std']}")
        lines.append(f"  median: {error_stats['median']}")

    if relative_error_stats is None or relative_error_meta is None:
        lines.append("Relative error stats: not available")
    else:
        lines.append("Relative error stats (symmetric: abs_error / max(abs(source), abs(target), floor)):")
        lines.append(f"  max: {relative_error_stats['max']} ({relative_error_stats['max'] * 100:.6f}%)")
        lines.append(f"  min: {relative_error_stats['min']} ({relative_error_stats['min'] * 100:.6f}%)")
        lines.append(f"  mean: {relative_error_stats['mean']} ({relative_error_stats['mean'] * 100:.6f}%)")
        lines.append(f"  std: {relative_error_stats['std']} ({relative_error_stats['std'] * 100:.6f}%)")
        lines.append(f"  median: {relative_error_stats['median']} ({relative_error_stats['median'] * 100:.6f}%)")
        lines.append(f"  p90: {relative_error_stats['p90']} ({relative_error_stats['p90'] * 100:.6f}%)")
        lines.append(f"  p95: {relative_error_stats['p95']} ({relative_error_stats['p95'] * 100:.6f}%)")
        lines.append(f"  p99: {relative_error_stats['p99']} ({relative_error_stats['p99'] * 100:.6f}%)")
        lines.append(f"  p99.9: {relative_error_stats['p99_9']} ({relative_error_stats['p99_9'] * 100:.6f}%)")
        lines.append(
            "Relative error config: "
            "reference=median(|source| where |source|>0), "
            f"floor_ratio={relative_error_meta['floor_ratio']}, "
            f"floor={relative_error_meta['floor']}, "
            f"reference_scale={relative_error_meta['reference_scale']}"
        )
        lines.append(
            "Relative error floor usage: "
            f"{int(relative_error_meta['floor_applied_count'])}/{int(relative_error_meta['total_count'])} "
            f"({relative_error_meta['floor_applied_fraction'] * 100:.6f}%)"
        )

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