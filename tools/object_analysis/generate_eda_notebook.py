#!/usr/bin/env python3
"""
generate_eda_notebook.py

方式1 EDA Notebook を nbformat で生成する。

実行:
    docker compose run --rm pss python tools/object_analysis/generate_eda_notebook.py
    docker compose run --rm pss python tools/object_analysis/generate_eda_notebook.py \
            --results_root results/yolox_tiny_custom_model_auto
"""

import argparse
import nbformat as nbf
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate method1 EDA notebook")
parser.add_argument(
        "--results_root",
        default="results/yolox_tiny_custom_model_auto",
        help="Results root directory (default: results/yolox_tiny_custom_model_auto)",
)
args = parser.parse_args()

RESULTS_ROOT = args.results_root

OUT_PATH = Path(f"{RESULTS_ROOT}/reports/eda_method1.ipynb")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# ---------------------------------------------------------------------------
# Title
cells.append(nbf.v4.new_markdown_cell("""# 方式1: 物体検出 正解/不正解 EDA

**前提**: `precompute_method1.py` を実行して集計CSVを生成済みであること。

```bash
docker compose run --rm pss python tools/object_analysis/precompute_method1.py
```"""))

# ---------------------------------------------------------------------------
# Setup
cells.append(nbf.v4.new_code_cell(f"""\
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# Docker コンテナ内ではワークスペースは /work に固定
# ホスト上で直接実行する場合は ROOT を書き換えてください
ROOT     = Path("/work")
AGG_DIR  = ROOT / "{RESULTS_ROOT}/aggregates"
ANAL_DIR = ROOT / "{RESULTS_ROOT}/analysis"
FIG_DIR  = ROOT / "{RESULTS_ROOT}/reports/figures/method1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df      = pd.read_csv(AGG_DIR / "objects_flat.csv")
cm_df   = pd.read_csv(AGG_DIR / "class_metrics.csv")
sb_df   = pd.read_csv(AGG_DIR / "size_band_metrics.csv")
feat_df = pd.read_csv(ANAL_DIR / "feature_stats.csv")

df["size_band"]    = pd.Categorical(df["size_band"],    categories=["xs","s","m","l"], ordered=True)
sb_df["size_band"] = pd.Categorical(sb_df["size_band"], categories=["xs","s","m","l"], ordered=True)

print(f"Total records : {{len(df):,}}")
print(df["result_type"].value_counts())
"""))

# ---------------------------------------------------------------------------
# 1. KPI Summary
cells.append(nbf.v4.new_markdown_cell("## 1. KPI サマリ"))
cells.append(nbf.v4.new_code_cell("""\
tp = df["is_correct"].sum()
fp = df["is_fp"].sum()
fn = df["is_fn"].sum()
total = len(df)

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
kpis = [
    ("Total",  f"{total:,}",               "#4e79a7"),
    ("TP",     f"{tp:,} ({tp/total:.1%})", "#59a14f"),
    ("FP",     f"{fp:,} ({fp/total:.1%})", "#e15759"),
    ("FN",     f"{fn:,} ({fn/total:.1%})", "#f28e2b"),
]
for ax, (title, val, color) in zip(axes, kpis):
    ax.text(0.5, 0.6, val,   ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.text(0.5, 0.2, title, ha="center", va="center", fontsize=12, color="gray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(color); spine.set_linewidth(2)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_kpi.png")
plt.show()
"""))

# ---------------------------------------------------------------------------
# 2. Class TP/FP/FN count + rate
cells.append(nbf.v4.new_markdown_cell("## 2. クラス別 TP / FP / FN（上位20クラス）"))
cells.append(nbf.v4.new_code_cell("""\
top20 = cm_df.nlargest(20, "total").copy()
top20["tp_rate"] = top20["tp"] / top20["total"]
top20["fp_rate"] = top20["fp"] / top20["total"]
top20["fn_rate"] = top20["fn"] / top20["total"]

fig, axes = plt.subplots(2, 1, figsize=(13, 10))
x = np.arange(len(top20)); w = 0.28

# --- 上段: 件数 ---
axes[0].bar(x - w, top20["tp"], width=w, label="TP", color="#59a14f")
axes[0].bar(x,     top20["fp"], width=w, label="FP", color="#e15759")
axes[0].bar(x + w, top20["fn"], width=w, label="FN", color="#f28e2b")
axes[0].set_xticks(x)
axes[0].set_xticklabels(top20["class_name"], rotation=45, ha="right")
axes[0].set_ylabel("Count")
axes[0].set_title("Class-wise TP / FP / FN — Count (top 20 by total)")
axes[0].legend()

# --- 下段: 率（積み上げ横棒） ---
axes[1].bar(x, top20["tp_rate"], width=0.6, label="TP rate", color="#59a14f")
axes[1].bar(x, top20["fp_rate"], width=0.6, label="FP rate", color="#e15759",
            bottom=top20["tp_rate"].values)
axes[1].bar(x, top20["fn_rate"], width=0.6, label="FN rate", color="#f28e2b",
            bottom=(top20["tp_rate"] + top20["fp_rate"]).values)
axes[1].set_xticks(x)
axes[1].set_xticklabels(top20["class_name"], rotation=45, ha="right")
axes[1].set_ylim(0, 1); axes[1].set_ylabel("Rate")
axes[1].set_title("Class-wise TP / FP / FN — Rate (top 20 by total)")
axes[1].legend(loc="upper right")
# 各バーの上にTP率をテキスト表示
for i, row in top20.reset_index(drop=True).iterrows():
    axes[1].text(i, row["tp_rate"] / 2, f"{row['tp_rate']:.0%}",
                 ha="center", va="center", fontsize=7, color="white", fontweight="bold")

plt.tight_layout(); plt.savefig(FIG_DIR / "02_class_tp_fp_fn.png"); plt.show()
"""))

# ---------------------------------------------------------------------------
# 3. F1
cells.append(nbf.v4.new_markdown_cell("## 3. クラス別 F1 スコア"))
cells.append(nbf.v4.new_code_cell("""\
plot_cm = cm_df.dropna(subset=["f1"]).sort_values("f1")
fig, ax = plt.subplots(figsize=(8, max(6, len(plot_cm) * 0.3)))
colors = ["#e15759" if v < 0.4 else "#f28e2b" if v < 0.6 else "#59a14f" for v in plot_cm["f1"]]
ax.barh(plot_cm["class_name"], plot_cm["f1"], color=colors)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("F1 Score"); ax.set_title("Class-wise F1 Score"); ax.set_xlim(0, 1)
plt.tight_layout(); plt.savefig(FIG_DIR / "03_class_f1.png"); plt.show()
"""))

# ---------------------------------------------------------------------------
# 4. Score dist
cells.append(nbf.v4.new_markdown_cell("## 4. Score (confidence) 分布: TP vs FP"))
cells.append(nbf.v4.new_code_cell("""\
tp_scores = df.loc[df["is_correct"], "score"]
fp_scores = df.loc[df["is_fp"],      "score"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(fp_scores, bins=50, alpha=0.6, color="#e15759", label=f"FP (n={len(fp_scores):,})", density=True)
ax.hist(tp_scores, bins=50, alpha=0.6, color="#59a14f", label=f"TP (n={len(tp_scores):,})", density=True)
ax.set_xlabel("Score (confidence)"); ax.set_ylabel("Density")
ax.set_title("Score Distribution: TP vs FP"); ax.legend()
plt.tight_layout(); plt.savefig(FIG_DIR / "04_score_dist.png"); plt.show()
"""))

# ---------------------------------------------------------------------------
# 5. Size band
cells.append(nbf.v4.new_markdown_cell("## 5. Box サイズ帯別 正解率"))
cells.append(nbf.v4.new_code_cell("""\
sb_sorted = sb_df.sort_values("size_band")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(sb_sorted["size_band"].astype(str), sb_sorted["correct_rate"],
            color=["#e15759","#f28e2b","#4e79a7","#59a14f"])
axes[0].set_ylim(0, 1); axes[0].set_ylabel("Correct Rate")
axes[0].set_title("Correct Rate by Size Band")
for _, row in sb_sorted.iterrows():
    axes[0].text(str(row["size_band"]), row["correct_rate"] + 0.01,
                 f"{row['correct_rate']:.2f}", ha="center", fontsize=10)
axes[1].bar(sb_sorted["size_band"].astype(str), sb_sorted["tp"], label="TP", color="#59a14f")
axes[1].bar(sb_sorted["size_band"].astype(str), sb_sorted["fp"], label="FP", color="#e15759",
            bottom=sb_sorted["tp"].values)
axes[1].bar(sb_sorted["size_band"].astype(str), sb_sorted["fn"], label="FN", color="#f28e2b",
            bottom=(sb_sorted["tp"] + sb_sorted["fp"]).values)
axes[1].set_ylabel("Count"); axes[1].set_title("TP / FP / FN by Size Band"); axes[1].legend()
plt.tight_layout(); plt.savefig(FIG_DIR / "05_size_band.png"); plt.show()
print(sb_sorted[["size_band","tp","fp","fn","total","correct_rate","mean_area_px","mean_iou_tp"]].to_string(index=False))
"""))

# ---------------------------------------------------------------------------
# 6. Area dist
cells.append(nbf.v4.new_markdown_cell("## 6. Box 面積 × 正解 / 不正解（対数スケール）"))
cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(8, 4))
for label, color, subset in [
    ("TP", "#59a14f", df[df["is_correct"]]),
    ("FP", "#e15759", df[df["is_fp"]]),
    ("FN", "#f28e2b", df[df["is_fn"]]),
]:
    ax.hist(np.log10(subset["box_area_orig_px"].clip(lower=1)),
            bins=50, alpha=0.5, label=f"{label} (n={len(subset):,})", density=True)
ax.set_xlabel("log10(Box Area [px²])"); ax.set_ylabel("Density")
ax.set_title("Box Area Distribution by Result Type")
for thresh, name in [(1024, "xs|s"), (9216, "s|m"), (65536, "m|l")]:
    ax.axvline(np.log10(thresh), color="gray", linestyle=":", linewidth=0.8)
    ax.text(np.log10(thresh) + 0.02, ax.get_ylim()[1] * 0.95, name, fontsize=8, color="gray")
ax.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "06_area_dist.png"); plt.show()
"""))

# ---------------------------------------------------------------------------
# 7. Center heatmap
cells.append(nbf.v4.new_markdown_cell("## 7. 中心位置ヒートマップ（正解 vs 不正解）"))
cells.append(nbf.v4.new_code_cell("""\
bins  = np.linspace(0, 1, 11)
valid = df.dropna(subset=["center_x_norm", "center_y_norm"])
valid = valid[(valid["center_x_norm"].between(0,1)) & (valid["center_y_norm"].between(0,1))]
correct   = valid[valid["is_correct"]]
incorrect = valid[~valid["is_correct"]]
h_c,  _, _ = np.histogram2d(correct["center_x_norm"],   correct["center_y_norm"],   bins=bins)
h_ic, _, _ = np.histogram2d(incorrect["center_x_norm"], incorrect["center_y_norm"], bins=bins)
total_map  = h_c + h_ic
rate_map   = np.where(total_map > 0, h_c / total_map, np.nan)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
kw = dict(origin="lower", extent=[0,1,0,1], vmin=0)
im0 = axes[0].imshow(h_c.T,    **kw, cmap="Greens"); axes[0].set_title("TP count");    fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(h_ic.T,   **kw, cmap="Reds");   axes[1].set_title("FP+FN count"); fig.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(rate_map.T, origin="lower", extent=[0,1,0,1], vmin=0, vmax=1, cmap="RdYlGn")
axes[2].set_title("Correct Rate"); fig.colorbar(im2, ax=axes[2])
for ax in axes:
    ax.set_xlabel("center_x_norm"); ax.set_ylabel("center_y_norm")
plt.suptitle("Object Center Position Heatmap", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR / "07_center_heatmap.png", bbox_inches="tight"); plt.show()
"""))

# ---------------------------------------------------------------------------
# 8. Feature means + correct rate per feature bin
cells.append(nbf.v4.new_markdown_cell("## 8. 画像特徴量: 正解 vs 不正解 の平均比較"))
cells.append(nbf.v4.new_code_cell("""\
pivot = feat_df.pivot_table(index="feature", columns="is_correct", values="mean")
pivot.columns = ["Incorrect (FP+FN)", "Correct (TP)"]
pivot["diff"] = pivot["Correct (TP)"] - pivot["Incorrect (FP+FN)"]
pivot = pivot.sort_values("diff")
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(pivot)); w = 0.35
ax.barh(x - w/2, pivot["Incorrect (FP+FN)"], height=w, label="Incorrect", color="#e15759", alpha=0.8)
ax.barh(x + w/2, pivot["Correct (TP)"],       height=w, label="Correct",   color="#59a14f", alpha=0.8)
ax.set_yticks(x); ax.set_yticklabels(pivot.index)
ax.set_xlabel("Mean Value"); ax.set_title("Image Feature Means: Correct vs Incorrect"); ax.legend()
plt.tight_layout(); plt.savefig(FIG_DIR / "08_feature_means.png"); plt.show()
print(pivot[["Correct (TP)", "Incorrect (FP+FN)", "diff"]].to_string())
"""))

# 8b. Correct rate per feature quantile bin
cells.append(nbf.v4.new_markdown_cell(
    "### 8b. 各画像特徴量の値域別 正解率\n"
    "各特徴量を10分位に区切り、区間ごとの正解率（TP / (TP+FP+FN)）をプロットする。"
))
cells.append(nbf.v4.new_code_cell("""\
FEATURE_COLS = [
    "luminance_mean", "luminance_std", "rms_contrast",
    "saturation_mean", "colorfulness", "sharpness_laplacian",
    "edge_density", "noise_sigma", "total_variation",
]
N_BINS = 10
ncols = 3; nrows = (len(FEATURE_COLS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
axes = axes.flatten()

for ax, col in zip(axes, FEATURE_COLS):
    tmp = df[[col, "is_correct"]].dropna(subset=[col])
    tmp["bin"] = pd.qcut(tmp[col], q=N_BINS, duplicates="drop")
    grp = tmp.groupby("bin", observed=False)
    correct_rate = grp["is_correct"].mean()
    counts       = grp["is_correct"].count()
    bin_labels   = [f"{iv.mid:.2g}" for iv in correct_rate.index]

    bars = ax.bar(range(len(correct_rate)), correct_rate.values,
                  color=["#59a14f" if v >= 0.5 else "#e15759" for v in correct_rate.values],
                  alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(range(len(counts)), counts.values, color="gray", linewidth=1,
             linestyle="--", marker="o", markersize=3, alpha=0.6)
    ax2.set_ylabel("Count", fontsize=7, color="gray")
    ax2.tick_params(axis="y", labelsize=6, labelcolor="gray")

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")
    ax.set_title(col, fontsize=9)
    ax.set_ylabel("Correct Rate", fontsize=7)

for ax in axes[len(FEATURE_COLS):]:
    ax.set_visible(False)

plt.suptitle("Correct Rate by Feature Value Bin (10-quantile)", y=1.01)
plt.tight_layout(); plt.savefig(FIG_DIR / "08b_feature_correct_rate.png", bbox_inches="tight"); plt.show()
"""))

# ---------------------------------------------------------------------------
# 9. Feature KDE
cells.append(nbf.v4.new_markdown_cell("## 9. 各特徴量の分布（正解 vs 不正解）"))
cells.append(nbf.v4.new_code_cell("""\
feature_cols = [
    "luminance_mean", "luminance_std", "rms_contrast",
    "saturation_mean", "colorfulness", "sharpness_laplacian",
    "edge_density", "noise_sigma", "total_variation",
]
ncols = 3; nrows = (len(feature_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
axes = axes.flatten()
for ax, col in zip(axes, feature_cols):
    for label, color, mask in [
        ("Incorrect", "#e15759", ~df["is_correct"]),
        ("Correct",   "#59a14f",  df["is_correct"]),
    ]:
        ax.hist(df.loc[mask, col].dropna(), bins=60, alpha=0.5, density=True, color=color, label=label)
    ax.set_title(col, fontsize=9); ax.legend(fontsize=7)
for ax in axes[len(feature_cols):]: ax.set_visible(False)
plt.suptitle("Feature Distribution: Correct vs Incorrect", y=1.01)
plt.tight_layout(); plt.savefig(FIG_DIR / "09_feature_kde.png", bbox_inches="tight"); plt.show()
"""))

# ---------------------------------------------------------------------------
# 10. Class x Size heatmap: color=count proportion, annot=correct rate
cells.append(nbf.v4.new_markdown_cell(
    "## 10. クラス × サイズ帯 正解率ヒートマップ（上位15クラス）\n\n"
    "- **数値**: 正解率（TP / (TP+FP+FN)）\n"
    "- **色**: そのクラス内でのサイズ帯の物体数割合（濃い = 物体が多い）"
))
cells.append(nbf.v4.new_code_cell("""\
top15_cls = cm_df.nlargest(15, "total")["class_name"].tolist()
sub = df[df["class_name"].isin(top15_cls)].copy()

correct_rate_data = (
    sub.groupby(["class_name", "size_band"], observed=False)["is_correct"]
    .mean().unstack("size_band")
)
count_data = (
    sub.groupby(["class_name", "size_band"], observed=False)
    .size().unstack("size_band")
).fillna(0)

# 各クラス内でのサイズ帯の物体数割合（行方向で正規化）
row_sum = count_data.sum(axis=1).replace(0, np.nan)
count_ratio = count_data.div(row_sum, axis=0)  # 色に使う

# アノテーション: 正解率（件数が5未満はハイフン）
anno = correct_rate_data.copy().astype(object)
for cls in anno.index:
    for band in anno.columns:
        n = count_data.loc[cls, band]
        r = correct_rate_data.loc[cls, band]
        anno.loc[cls, band] = f"{r:.2f}" if n >= 5 else "-"

fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(
    count_ratio,
    annot=anno, fmt="",
    cmap="Blues", vmin=0, vmax=1,
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Size band proportion within class"}
)
ax.set_title("Class x Size Band  (color=count proportion, number=correct rate, '-'=n<5)")
ax.set_xlabel("Size Band"); ax.set_ylabel("Class")
plt.tight_layout(); plt.savefig(FIG_DIR / "10_class_size_heatmap.png"); plt.show()
"""))

# 10b. 一覧表
cells.append(nbf.v4.new_markdown_cell("### 10b. クラス × サイズ帯 一覧表"))
cells.append(nbf.v4.new_code_cell("""\
# 正解率 + 件数を合わせた一覧表を構築
rows = []
for cls in top15_cls:
    for band in ["xs", "s", "m", "l"]:
        g = sub[(sub["class_name"] == cls) & (sub["size_band"] == band)]
        n = len(g)
        tp_ = int(g["is_correct"].sum())
        fp_ = int(g["is_fp"].sum())
        fn_ = int(g["is_fn"].sum())
        rate = tp_ / n if n > 0 else float("nan")
        rows.append(dict(
            class_name=cls, size_band=band,
            total=n, tp=tp_, fp=fp_, fn=fn_,
            correct_rate=round(rate, 3) if n > 0 else float("nan"),
        ))

table_df = pd.DataFrame(rows)
table_df["size_band"] = pd.Categorical(table_df["size_band"], categories=["xs","s","m","l"], ordered=True)
table_df = table_df.sort_values(["class_name", "size_band"]).reset_index(drop=True)

# ピボット表示（class x size_band, 各セルに 'TP率 (n=件数)' を表示）
pivot_disp = table_df.pivot_table(
    index="class_name", columns="size_band",
    values=["correct_rate", "total"], aggfunc="first"
)
# 見やすいテキスト形式に変換
result_rows = []
for cls in table_df["class_name"].unique():
    row = {"class_name": cls}
    for band in ["xs", "s", "m", "l"]:
        sub_row = table_df[(table_df["class_name"]==cls) & (table_df["size_band"]==band)]
        if len(sub_row) == 0 or sub_row["total"].values[0] == 0:
            row[band] = "-"
        else:
            r = sub_row["correct_rate"].values[0]
            n = sub_row["total"].values[0]
            tp_n = sub_row["tp"].values[0]
            row[band] = f"{r:.2f} (n={n}, TP={tp_n})"
    result_rows.append(row)

disp_df = pd.DataFrame(result_rows).set_index("class_name")
print(disp_df.to_string())
disp_df
"""))

nb.cells = cells
nbf.write(nb, str(OUT_PATH))
print(f"Notebook written to: {OUT_PATH}")
