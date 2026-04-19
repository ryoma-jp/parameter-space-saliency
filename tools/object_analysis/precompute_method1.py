#!/usr/bin/env python3
"""
precompute_method1.py

object_catalog.json をフラット化し、集計CSVを生成する（方式1: ベース集計）

出力:
  results/yolox_tiny_custom_model_auto/aggregates/objects_flat.csv
  results/yolox_tiny_custom_model_auto/aggregates/class_metrics.csv
  results/yolox_tiny_custom_model_auto/aggregates/size_band_metrics.csv
  results/yolox_tiny_custom_model_auto/analysis/feature_stats.csv

実行:
  docker compose run --rm pss python tools/object_analysis/precompute_method1.py
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = Path("results/yolox_tiny_custom_model_auto")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# COCO準拠のサイズ帯定義
SIZE_THRESHOLDS = [
    (0,      1_024,       "xs"),   # ~32x32 未満
    (1_024,  9_216,       "s"),    # 32x32 ～ 96x96
    (9_216,  65_536,      "m"),    # 96x96 ～ 256x256
    (65_536, float("inf"), "l"),   # 256x256 以上
]

# 画像特徴量カラム（JSON内に存在する列）
IMAGE_FEATURE_COLS = [
    "luminance_mean",
    "luminance_std",
    "luminance_median",
    "rms_contrast",
    "saturation_mean",
    "saturation_std",
    "hue_std",
    "colorfulness",
    "noise_sigma",
    "total_variation",
    "sharpness_laplacian",
    "edge_density",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def assign_size_band(area: float) -> str:
    for lo, hi, name in SIZE_THRESHOLDS:
        if lo <= area < hi:
            return name
    return "l"


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom != 0 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten object_catalog.json and generate aggregate CSVs"
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing object_catalog.json",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    catalog_path = ROOT / args.results_root / "object_catalog.json"
    out_agg = ROOT / args.results_root / "aggregates"
    out_analysis = ROOT / args.results_root / "analysis"

    out_agg.mkdir(parents=True, exist_ok=True)
    out_analysis.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. JSON ロード
    # -----------------------------------------------------------------------
    print(f"Using results_root: {args.results_root}")
    print("Loading object_catalog.json ...")
    with open(catalog_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"  Records : {len(df):,}")
    print(f"  result_type : {df['result_type'].value_counts().to_dict()}")

    # -----------------------------------------------------------------------
    # 2. 派生特徴量の生成
    # -----------------------------------------------------------------------

    # 正解フラグ
    df["is_correct"] = df["result_type"] == "tp"
    df["is_fp"]      = df["result_type"].isin(["fp_loc", "fp_cls"])
    df["is_fn"]      = df["result_type"] == "fn"

    # fn は score が NaN → 0 で補完（予測なしを意味する）
    df["score"] = df["score"].fillna(0.0)

    # サイズ帯
    df["size_band"] = df["box_area_orig_px"].apply(assign_size_band)
    df["size_band"] = pd.Categorical(
        df["size_band"], categories=["xs", "s", "m", "l"], ordered=True
    )

    # 中心座標（画像幅・高さで正規化）
    # box_width_ratio = box_width / image_width から image_width を逆算
    box_w = df["box_orig_x2"] - df["box_orig_x1"]
    box_h = df["box_orig_y2"] - df["box_orig_y1"]
    img_w = np.where(df["box_width_ratio"]  > 0, box_w / df["box_width_ratio"],  np.nan)
    img_h = np.where(df["box_height_ratio"] > 0, box_h / df["box_height_ratio"], np.nan)
    cx = (df["box_orig_x1"] + df["box_orig_x2"]) / 2
    cy = (df["box_orig_y1"] + df["box_orig_y2"]) / 2
    df["center_x_norm"] = cx / img_w
    df["center_y_norm"] = cy / img_h

    # 画像端からの最小距離（0〜0.5）
    df["border_distance"] = pd.concat(
        [
            df["center_x_norm"],
            df["center_y_norm"],
            1 - df["center_x_norm"],
            1 - df["center_y_norm"],
        ],
        axis=1,
    ).min(axis=1)

    # -----------------------------------------------------------------------
    # 3. objects_flat.csv 出力
    # -----------------------------------------------------------------------
    flat_path = out_agg / "objects_flat.csv"
    df.to_csv(flat_path, index=False)
    print(f"\nSaved: {flat_path}  ({len(df):,} rows)")

    # -----------------------------------------------------------------------
    # 4. class_metrics.csv
    # -----------------------------------------------------------------------
    class_rows = []
    for cls_name, g in df.groupby("class_name"):
        tp  = int(g["is_correct"].sum())
        fp  = int(g["is_fp"].sum())
        fn  = int(g["is_fn"].sum())
        prec = safe_div(tp, tp + fp)
        rec  = safe_div(tp, tp + fn)
        f1   = safe_div(2 * prec * rec, prec + rec) if not (
            np.isnan(prec) or np.isnan(rec)
        ) else float("nan")

        mean_score_tp = g.loc[g["is_correct"], "score"].mean()
        mean_score_fp = g.loc[g["is_fp"],      "score"].mean()
        mean_iou_tp   = g.loc[g["is_correct"], "max_gt_iou"].mean()

        class_rows.append(
            dict(
                class_name    = cls_name,
                cls_id        = int(g["cls_id"].iloc[0]),
                tp=tp, fp=fp, fn=fn,
                total         = len(g),
                precision     = round(prec, 4),
                recall        = round(rec,  4),
                f1            = round(f1,   4),
                mean_score_tp = round(mean_score_tp, 4) if not np.isnan(mean_score_tp) else float("nan"),
                mean_score_fp = round(mean_score_fp, 4) if not np.isnan(mean_score_fp) else float("nan"),
                mean_iou_tp   = round(mean_iou_tp,   4) if not np.isnan(mean_iou_tp)   else float("nan"),
            )
        )

    cm_df = (
        pd.DataFrame(class_rows)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    cm_path = out_agg / "class_metrics.csv"
    cm_df.to_csv(cm_path, index=False)
    print(f"Saved: {cm_path}  ({len(cm_df)} classes)")

    # -----------------------------------------------------------------------
    # 5. size_band_metrics.csv
    # -----------------------------------------------------------------------
    sb_rows = []
    for band, g in df.groupby("size_band", observed=False):
        tp    = int(g["is_correct"].sum())
        fp    = int(g["is_fp"].sum())
        fn    = int(g["is_fn"].sum())
        total = len(g)
        correct_rate = safe_div(tp, total)
        mean_iou = g.loc[g["is_correct"], "max_gt_iou"].mean()
        mean_area = g["box_area_orig_px"].mean()
        sb_rows.append(
            dict(
                size_band    = band,
                tp=tp, fp=fp, fn=fn,
                total        = total,
                correct_rate = round(correct_rate, 4),
                mean_area_px = round(mean_area, 1),
                mean_iou_tp  = round(mean_iou, 4) if not np.isnan(mean_iou) else float("nan"),
            )
        )

    sb_df = pd.DataFrame(sb_rows)
    sb_path = out_agg / "size_band_metrics.csv"
    sb_df.to_csv(sb_path, index=False)
    print(f"Saved: {sb_path}")
    print(sb_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # 6. feature_stats.csv（画像特徴量の正解/不正解別統計）
    # -----------------------------------------------------------------------
    feat_rows = []
    for col in IMAGE_FEATURE_COLS:
        if col not in df.columns:
            continue
        for label, g in df.groupby("is_correct"):
            vals = g[col].dropna()
            feat_rows.append(
                dict(
                    feature    = col,
                    is_correct = label,
                    count      = len(vals),
                    mean       = round(vals.mean(),              4),
                    std        = round(vals.std(),               4),
                    median     = round(vals.median(),            4),
                    q25        = round(vals.quantile(0.25),      4),
                    q75        = round(vals.quantile(0.75),      4),
                )
            )

    feat_df = pd.DataFrame(feat_rows)
    feat_path = out_analysis / "feature_stats.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"Saved: {feat_path}")

    print("\n===== Summary =====")
    print(f"  Total objects  : {len(df):,}")
    print(f"  TP             : {df['is_correct'].sum():,}  ({df['is_correct'].mean():.1%})")
    print(f"  FP (loc+cls)   : {df['is_fp'].sum():,}  ({df['is_fp'].mean():.1%})")
    print(f"  FN             : {df['is_fn'].sum():,}  ({df['is_fn'].mean():.1%})")
    print(f"  Classes        : {df['class_name'].nunique()}")
    print("Done.")


if __name__ == "__main__":
    main()
