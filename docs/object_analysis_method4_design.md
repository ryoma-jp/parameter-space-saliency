# 方式4: 正解しやすさ予測モデル（XGBoost + SHAP）設計書

## 目的

物体ごとの属性（Box幾何・推論情報・画像特徴量統計）を入力として、  
「その物体を正しく検出できるか（正解/不正解）」を予測する二値分類モデルを学習する。  
SHAPによるグローバル/ローカル説明で、**どの要素が正誤に最も寄与するか**を定量化する。

---

## 前提

- 方式1の `precompute_method1.py` が実行済みであること
- `results/yolox_tiny_custom_model_auto/aggregates/objects_flat.csv` が存在すること

---

## 入力データ

| ファイル | 内容 |
|---|---|
| `results/yolox_tiny_custom_model_auto/aggregates/objects_flat.csv` | 物体ごとのフラット集計（方式1出力） |

---

## 出力物

| ファイル | 内容 |
|---|---|
| `tools/object_analysis/train_difficulty_model.py` | 学習・評価・SHAP分析スクリプト |
| `results/yolox_tiny_custom_model_auto/analysis/difficulty_model.json` | モデル評価指標（accuracy/AUC-ROC/F1等） |
| `results/yolox_tiny_custom_model_auto/analysis/feature_importance.csv` | XGBoost特徴量重要度 |
| `results/yolox_tiny_custom_model_auto/analysis/shap_values.csv` | 全サンプルのSHAP値 |
| `results/yolox_tiny_custom_model_auto/analysis/top_rules.md` | 上位寄与ルールのサマリ |
| `figures/shap_global_bar.png` | SHAPグローバル寄与（棒グラフ） |
| `figures/shap_beeswarm.png` | SHAPビースウォームプロット |
| `figures/shap_dependence_top3.png` | 上位3特徴量のSHAP依存プロット |
| `results/yolox_tiny_custom_model_auto/reports/method4_analysis.ipynb` | 結果確認Notebook |

---

## ディレクトリ構成

```
tools/
  object_analysis/
    train_difficulty_model.py   # 学習・評価・SHAP分析
results/yolox_tiny_custom_model_auto/
  analysis/
    difficulty_model.json
    feature_importance.csv
    shap_values.csv
    top_rules.md
  reports/
    method4_analysis.ipynb
figures/
  shap_global_bar.png
  shap_beeswarm.png
  shap_dependence_top3.png
```

---

## 使用特徴量（入力X）

### Box幾何特徴

| カラム名 | 定義 | 備考 |
|---|---|---|
| `box_area` | gt_boxの面積（ピクセル） | FNはgt_box基準 |
| `aspect_ratio` | 幅/高さ比率 | log変換を試みる |
| `center_x_norm` | 中心X座標（画像幅で正規化） | 0〜1 |
| `center_y_norm` | 中心Y座標（画像高さで正規化） | 0〜1 |
| `border_distance` | 画像端からの最小距離（正規化） | 0〜0.5 |

### 推論情報

| カラム名 | 定義 | 備考 |
|---|---|---|
| `confidence` | 予測スコア | FNは0またはNaN（→0で補完） |
| `iou` | 予測BoxとGT BoxのIoU | FNは0で補完 |

### 画像特徴量統計

| カラム名 | 定義 |
|---|---|
| `feature_norm` | 特徴ベクトルのL2ノルム |
| `feature_mean` | 特徴ベクトルの平均値 |
| `feature_std` | 特徴ベクトルの標準偏差 |

### カテゴリ特徴

| カラム名 | 定義 | エンコード |
|---|---|---|
| `class_id` | クラスID | そのまま整数 |
| `size_band` | xs/s/m/l | 順序エンコード（0/1/2/3） |

---

## ターゲット変数（Y）

| カラム名 | 定義 |
|---|---|
| `is_correct` | TP=1, FP/FN=0 |

**注意**: `match_type`・`iou`（生値）はデータリーク候補。  
`iou` はモデルの「結果」であるため、特徴量から除外するオプションを設ける。

---

## 処理フロー

```
objects_flat.csv
  │
  ▼
[1] データ読み込み・前処理
    - FNの confidence/iou を 0 で補完
    - カテゴリ特徴のエンコード
    - データリーク列（match_type等）の除外
    │
    ▼
[2] Train/Test 分割
    - stratified split（is_correct比率保持）
    - test_size=0.2, random_state=42
    │
    ▼
[3] XGBoost 学習
    - 目的関数: binary:logistic
    - 評価指標: AUC-ROC
    - クラス不均衡対応: scale_pos_weight
    - Optuna による主要ハイパーパラメータ探索（任意）
    │
    ▼
[4] モデル評価
    - accuracy, AUC-ROC, F1（macro/binary）
    - confusion matrix
    │
    ▼
[5] SHAP分析
    - TreeExplainer によるSHAP値計算
    - グローバル特徴量重要度（mean |SHAP|）
    - ビースウォームプロット
    - 上位3特徴量の依存プロット
    │
    ▼
[6] ルールサマリ生成
    - SHAP値トップN特徴量の正負方向と値域から
      「この条件だと失敗しやすい」を自然言語で記述
    - top_rules.md に出力
    │
    ▼
[7] Notebook で結果確認
```

---

## モデル設定

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=<neg/pos比>,  # クラス不均衡対応
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42
)
```

---

## SHAP出力イメージ

### グローバル寄与（`shap_global_bar.png`）

```
特徴量          mean |SHAP|
confidence      ████████████  0.42
box_area        ██████████    0.35
feature_norm    ███████       0.24
center_y_norm   █████         0.18
aspect_ratio    ████          0.14
...
```

### ルールサマリ（`top_rules.md`）例

```markdown
## 不正解になりやすい条件（上位3ルール）

1. **confidence が低い（< 0.4）**: SHAP値が大きく負に振れる。FPの大部分を占める。
2. **box_area が小さい（< 1024 px²）**: xsサイズの物体はFNが多い。
3. **feature_norm が低い（< 10.0）**: 特徴量の活性化が弱い物体は不正解傾向。
```

---

## 実装ステップ

| # | 作業 | 担当ファイル |
|---|---|---|
| 1 | 前処理・特徴量エンジニアリング | `train_difficulty_model.py` |
| 2 | Train/Test分割 | `train_difficulty_model.py` |
| 3 | XGBoost学習・評価 | `train_difficulty_model.py` |
| 4 | SHAP計算・グラフ出力 | `train_difficulty_model.py` |
| 5 | ルールサマリ生成 | `train_difficulty_model.py` |
| 6 | Notebook作成（結果確認） | `method4_analysis.ipynb` |

---

## 実行方法（Docker環境）

```bash
docker compose run --rm yolox \
  python tools/object_analysis/train_difficulty_model.py
```

---

## 依存ライブラリ

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
optuna          # ハイパーパラメータ探索（任意）
```

---

## リスクと対策

| リスク | 対策 |
|---|---|
| データリーク（iou, match_type を特徴量に含める） | `EXCLUDE_COLS` リストを明示し、デフォルトで除外 |
| クラス不均衡（TP >> FP+FN、またはその逆） | `scale_pos_weight` で調整 + AUC-ROCで評価 |
| サンプル数不足でSHAPが不安定 | サンプル数をログに出力し、< 200件の場合は警告 |
| 特徴量ベクトルが高次元 | 統計量（norm/mean/std）のみ使用し次元爆発を防ぐ |

---

## 受け入れ条件（完了判定）

- [ ] スクリプトがエラーなく実行完了する
- [ ] `difficulty_model.json` に AUC-ROC が記録される
- [ ] `shap_global_bar.png` が生成される
- [ ] `top_rules.md` に上位3以上のルールが記述される
- [ ] Notebookをそのまま実行してエラーなく全グラフが出力される
