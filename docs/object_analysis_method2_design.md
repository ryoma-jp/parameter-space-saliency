# 方式2: 誤り原因分解ダッシュボード（Streamlit）設計書

## 目的

方式1で生成した集計CSVを入力として、Streamlitによるインタラクティブな
Webダッシュボードを提供する。  
「どの条件・クラス・サイズ・特徴量クラスタで誤りが増えるか」を  
対話的なフィルタリングとグラフで即時確認できるようにする。

---

## 前提

- 方式1の `precompute_method1.py` が実行済みであること
- `results/yolox_tiny_custom_model_auto/aggregates/objects_flat.csv` が存在すること

---

## 入力データ

| ファイル | 内容 |
|---|---|
| `results/yolox_tiny_custom_model_auto/aggregates/objects_flat.csv` | 物体ごとのフラット集計（方式1出力） |
| `results/yolox_tiny_custom_model_auto/aggregates/class_metrics.csv` | クラス別指標（方式1出力） |
| `raw_images/` | 元画像（サムネイル表示用・任意） |

---

## 出力物

| ファイル | 内容 |
|---|---|
| `tools/object_analysis/app.py` | Streamlitアプリ本体 |
| `tools/object_analysis/precompute_method2.py` | Parquet事前変換スクリプト |
| `results/yolox_tiny_custom_model_auto/aggregates/objects_flat.parquet` | 高速読み込み用Parquet |

---

## ディレクトリ構成

```
tools/
  object_analysis/
    app.py                    # Streamlitアプリ本体
    precompute_method2.py     # CSV→Parquet変換
    components/
      sidebar_filters.py      # サイドバーフィルタUI
      chart_heatmap.py        # ヒートマップ描画
      chart_distribution.py   # 分布グラフ描画
      chart_ranking.py        # FP/FNランキング描画
results/yolox_tiny_custom_model_auto/
  aggregates/
    objects_flat.parquet
```

---

## 画面構成

### サイドバー（フィルタ）

| コントロール | 型 | フィルタ対象 |
|---|---|---|
| クラス選択 | multiselect | `class_name` |
| 正誤フィルタ | radio | `match_type`: ALL / TP / FP / FN |
| Confidence帯 | slider | `confidence` 下限/上限 |
| Box面積帯 | multiselect | `size_band`: xs / s / m / l |
| IoU閾値 | slider | `iou` 下限（TP判定基準の確認用） |

### メインエリア

#### タブ1: 概要ダッシュボード

| ウィジェット | 内容 |
|---|---|
| KPIカード（上部4枚） | 総物体数 / TP率 / FP率 / FN率 |
| 棒グラフ | クラス別 TP/FP/FN 件数（スタック） |
| 折れ線グラフ | confidence帯別 正解率（TP / (TP+FP+FN)） |
| 折れ線グラフ | Box面積帯別 正解率 |

#### タブ2: ヒートマップ分析

| ウィジェット | 内容 |
|---|---|
| ヒートマップ | クラス（行） × サイズ帯（列）の正解率 |
| ヒートマップ | クラス（行） × Confidence帯（列）の正解率 |
| ヒートマップ | 中心X帯 × 中心Y帯 の正解率（画像位置依存性） |

#### タブ3: 要因ランキング

| ウィジェット | 内容 |
|---|---|
| 水平棒グラフ | FP上位クラス（件数） |
| 水平棒グラフ | FN上位クラス（件数） |
| 散布図 | confidence × IoU（TP/FP色分け） |
| KDE | 特徴量ノルム分布（正解 vs 不正解） |

#### タブ4: サンプルビューア（任意）

| ウィジェット | 内容 |
|---|---|
| グリッド表示 | フィルタ条件に合致するサンプル物体の画像＋Box |
| テーブル | 選択物体の詳細属性（class, confidence, IoU, 面積 等） |

---

## データフロー

```
objects_flat.csv / class_metrics.csv
  │
  ▼
[precompute_method2.py]
  CSV → Parquet 変換（型最適化, カテゴリ列のエンコード）
  │
  ▼
objects_flat.parquet
  │
  ▼
[app.py 起動時]
  @st.cache_data でParquetをメモリロード
  │
  ▼
[サイドバー操作]
  フィルタ条件 → pandas .query() でDataFrame絞り込み
  │
  ▼
[各タブ]
  絞り込み済みDataFrame → plotly / matplotlib でグラフ生成
  st.plotly_chart / st.pyplot で表示
```

---

## 実装ステップ

| # | 作業 | 担当ファイル |
|---|---|---|
| 1 | Parquet変換スクリプト作成 | `precompute_method2.py` |
| 2 | サイドバーフィルタUI実装 | `components/sidebar_filters.py` |
| 3 | タブ1: 概要ダッシュボード実装 | `app.py` |
| 4 | タブ2: ヒートマップ実装 | `components/chart_heatmap.py` |
| 5 | タブ3: 要因ランキング実装 | `components/chart_ranking.py` |
| 6 | タブ4: サンプルビューア実装（任意） | `app.py` |
| 7 | Docker起動確認・ポート設定 | `docker-compose.yaml` への追記 |

---

## 起動方法（Docker環境）

```bash
# Parquet 事前生成
docker compose run --rm yolox python tools/object_analysis/precompute_method2.py

# Streamlit 起動
docker compose run --rm --service-ports yolox \
  streamlit run tools/object_analysis/app.py --server.port 8501
```

ブラウザで `http://localhost:8501` にアクセスする。

---

## 依存ライブラリ

```
streamlit
pandas
pyarrow          # Parquet読み書き
plotly
matplotlib
seaborn
Pillow           # サムネイル表示用（任意）
```

---

## リスクと対策

| リスク | 対策 |
|---|---|
| 特徴量ベクトルが高次元でParquetが重い | feature_vectorはParquetに含めず、統計量（norm/mean/std）のみ保持 |
| サンプル画像パスが環境依存 | タブ4は任意扱いとし、パスが存在しない場合はスキップ表示 |
| フィルタ後にデータが0件 | 全コンポーネントで件数=0時に警告メッセージを表示 |
| Docker内でブラウザが開けない | `--server.headless true` で起動し、ホストからポート転送 |

---

## 受け入れ条件（完了判定）

- [ ] `objects_flat.parquet` が生成される
- [ ] `streamlit run app.py` でエラーなく起動する
- [ ] サイドバーのすべてのフィルタが機能し、グラフがリアルタイムで更新される
- [ ] タブ1〜3のすべてのグラフが表示される
- [ ] データ0件時に適切なメッセージが表示される
