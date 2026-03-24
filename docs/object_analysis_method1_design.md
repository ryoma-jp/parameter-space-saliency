# 方式1: ベース集計（CSV + Notebook分析）設計書

## 目的

`object_catalog.json` に記録された物体ごとの正解/不正解・Box情報・画像特徴量を
フラット化してCSVに集約し、Jupyter Notebookで基本的な統計・可視化を行う。  
「どのクラス・サイズ・特徴量の傾向が正解しやすいか/不正解しやすいか」を最短で把握する。

---

## 入力データ想定

| ファイル | 内容 |
|---|---|
| `results/yolox_tiny_custom_model_auto/object_catalog.json` | 評価データ全物体のレコード |

### JSONの実スキーマ（確認済み: 43,116件）

| フィールド | 説明 |
|---|---|
| `image_id` | 画像ID |
| `result_type` | `"tp"` / `"fp_loc"` / `"fp_cls"` / `"fn"` |
| `cls_id`, `class_name` | クラスID・クラス名（80クラス） |
| `score` | 予測スコア（fn は NaN） |
| `box_orig_x1/y1/x2/y2` | 元画像座標系のBox |
| `box_input_x1/y1/x2/y2` | モデル入力座標系のBox |
| `box_area_orig_px` | Box面積（元画像ピクセル） |
| `box_relative_area` | 画像面積に対する相対面積 |
| `box_aspect_ratio` | 幅/高さ比率 |
| `box_width_ratio`, `box_height_ratio` | 画像サイズ比 |
| `luminance_mean/std/median` | 輝度統計 |
| `rms_contrast` | RMSコントラスト |
| `saturation_mean/std` | 彩度統計 |
| `hue_std` | 色相標準偏差 |
| `colorfulness` | カラフル度 |
| `noise_sigma` | ノイズ推定値 |
| `total_variation` | 総変動量 |
| `sharpness_laplacian` | ラプラシアンによる鮮明度 |
| `edge_density` | エッジ密度 |
| `max_gt_iou` | GTとの最大IoU |

**件数実績**: tp=17,401 / fp_loc=6,084 / fp_cls=961 / fn=18,670

---

## 出力物

| ファイル | 内容 |
|---|---|
| `results/yolox_tiny_custom_model_auto/aggregates/objects_flat.csv` | 物体ごとのフラット集計（1行=1物体） |
| `results/yolox_tiny_custom_model_auto/aggregates/class_metrics.csv` | クラス別 TP/FP/FN/precision/recall/F1 |
| `results/yolox_tiny_custom_model_auto/aggregates/size_band_metrics.csv` | Box面積帯別 正解率 |
| `results/yolox_tiny_custom_model_auto/analysis/feature_stats.csv` | 特徴量の統計量（正解/不正解別） |
| `results/yolox_tiny_custom_model_auto/reports/eda_method1.ipynb` | 可視化Notebook |

---

## ディレクトリ構成

```
tools/
  object_analysis/
    precompute_method1.py    # JSON→CSV変換・集計スクリプト
results/yolox_tiny_custom_model_auto/
  aggregates/
    objects_flat.csv
    class_metrics.csv
    size_band_metrics.csv
  analysis/
    feature_stats.csv
  reports/
    eda_method1.ipynb
```

---

## 処理フロー

```
object_catalog.json
  │
  ▼
[1] JSONパース・フラット化
    - 画像レベル属性＋物体レベル属性を1行に展開
    - gt_box/pred_box から派生特徴量を計算
    │
    ▼
[2] 派生特徴量の生成
    - box_area: (x2-x1)*(y2-y1)  ← gt_box基準
    - aspect_ratio: (x2-x1)/(y2-y1)
    - center_x_norm, center_y_norm: 画像幅/高さで正規化
    - border_distance: 画像端からの最小距離（正規化済み）
    - size_band: "xs"(<32²) | "s"(<96²) | "m"(<256²) | "l"
    - feature_norm: L2ノルム
    - feature_mean, feature_std: ベクトルの平均・標準偏差
    │
    ▼
[3] CSVエクスポート
    objects_flat.csv ← 全物体フラット
    │
    ▼
[4] クラス別・サイズ帯別集計
    class_metrics.csv
    size_band_metrics.csv
    │
    ▼
[5] 特徴量統計（正解/不正解別）
    feature_stats.csv
    │
    ▼
[6] Notebook可視化
```

---

## 主要集計指標

### クラス別（`class_metrics.csv`）

| カラム | 定義 |
|---|---|
| `class_name` | クラス名 |
| `tp`, `fp`, `fn` | 件数 |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1` | 調和平均 |
| `mean_confidence_tp` | TP時の平均confidence |
| `mean_confidence_fp` | FP時の平均confidence |
| `mean_iou_tp` | TP時の平均IoU |

### サイズ帯別（`size_band_metrics.csv`）

| カラム | 定義 |
|---|---|
| `size_band` | xs / s / m / l |
| `correct_rate` | TP / (TP + FP + FN) |
| `mean_iou` | 平均IoU |
| `count` | 件数 |

### 特徴量統計（`feature_stats.csv`）

| カラム | 定義 |
|---|---|
| `feature` | 特徴量名（12種） |
| `is_correct` | True(TP) / False(FP+FN) |
| `count`, `mean`, `std` | 基本統計 |
| `median`, `q25`, `q75` | 四分位統計 |

---

## Notebook可視化内容

1. 物体数・TP/FP/FN比率（棒グラフ）
2. クラス別 F1スコア（水平棒グラフ）
3. confidence分布（TP vs FP, KDE）
4. Box面積 × 正解率（散布図 + 帯集計の折れ線）
5. 縦横比 × 正解率（散布図）
6. 中心位置ヒートマップ（正解 vs 不正解）
7. 特徴量ノルム分布（正解 vs 不正解, KDE）

---

## 実装ステップ

| # | 作業 | 担当ファイル |
|---|---|---|
| 1 | JSONスキーマ確認・フィールド定義確定 | `precompute_method1.py` |
| 2 | フラット化・派生特徴量生成 | `precompute_method1.py` |
| 3 | CSV出力 | `precompute_method1.py` |
| 4 | クラス別/サイズ別集計 | `precompute_method1.py` |
| 5 | 特徴量統計集計 | `precompute_method1.py` |
| 6 | Notebook作成（可視化）| `eda_method1.ipynb` |

---

## 依存ライブラリ

```
pandas
numpy
matplotlib
seaborn
nbformat    # Notebook生成
nbconvert   # Notebook実行
```

---

## リスクと対策

| リスク | 対策 |
|---|---|
| `feature_vector` が全エントリに存在しない | 欠損チェックを入れ、欠損行はfeature統計から除外 |
| FNエントリにpred_boxが存在しない | pred_box欠損時は派生特徴量をNaNで埋める |
| クラス不均衡による指標の偏り | 件数を必ずセットで表示する |

---

## 受け入れ条件（完了判定）

- [x] `objects_flat.csv` が生成され、全物体のレコードが含まれる
- [x] `class_metrics.csv` に precision/recall/F1 が含まれる
- [x] `size_band_metrics.csv` に4帯の正解率が含まれる
- [x] Notebookを実行してエラーなく全グラフが出力される（`figures/method1/` に10枚生成済み）

## 実行コマンド

```bash
# 集計CSV生成
docker compose run --rm pss python tools/object_analysis/precompute_method1.py

# Notebook生成（初回のみ）
docker compose run --rm pss python tools/object_analysis/generate_eda_notebook.py

# Notebook実行（全グラフ出力）
docker compose run --rm pss bash -c "
  cd /work && jupyter nbconvert --to notebook --execute \\
  results/yolox_tiny_custom_model_auto/reports/eda_method1.ipynb \\
  --output-dir /work/results/yolox_tiny_custom_model_auto/reports/ \\
  --output eda_method1_executed
"
