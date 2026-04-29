# FP/FNをHOTにし、TPをHOTにしない可視化: PSS準拠の統合設計

## 目的

本設計の目的は以下を同時に満たすことである。

- FP/FNをHOTにする
- TPをHOTにしない

ここでのHOTは、主にフィルタ単位サリエンシーで寄与が強いことを指し、この寄与度を入力空間へ投影して可視化する。

---

## 前提の原理

本設計は、PSS（Parameter Space Saliency Maps, arXiv:2108.01335）の原理に従う。

- 一次指標は入力勾配 $\partial L/\partial x$ ではなく、パラメータ勾配 $\partial L/\partial \theta$
- まず重み単位サリエンシーを計算する
- 次に重みをフィルタ単位へ統合して Filter Saliency を得る
- 最終的な目的「FP/FN HOT・TP non-HOT」は、この Filter Saliency の合成重みで制御する

---

## 対象の4分類

本設計では、検出結果を以下の4タイプへ分離して扱う。

1. TP
- 正しい位置かつ正しいクラス

2. FN
- GTがあるのに未検出

3. FP-A
- GTがない領域に物体を検出（位置型FP）

4. FP-B
- GTがある領域に物体を検出するがクラス誤分類（クラス混同型FP）

---

## 統合スコアの基本方針

単純な勾配だけでは目的を満たしにくいため、物体単位LossとLossゲートを導入し、重み単位からフィルタ単位へ統合する。

### 1. 物体単位Loss

各タイプ・各物体 $k$ に対して個別Lossを定義する。

- TP: $L_k^{tp}$
- FN: $L_k^{fn}$
- FP-A: $L_k^{fpA}$
- FP-B: $L_k^{fpB}$

要件:

- 望ましい状態では小さく、問題状態では大きくなる
- 画像平均Lossではなく、物体単位Lossを基本とする

### 2. Loss重み付き重みサリエンシー（PSSの主計算）

パラメータ $\theta_j$ に対する物体 $k$ の重みサリエンシーを次で定義する。

$$
S_{k,j}^{param} = g(L_k)\,h\!\left(\frac{\partial L_k}{\partial \theta_j}\right)
$$

- $g(L_k)$: Loss大きさのゲート関数（単調増加）
- $h(\cdot)$: 勾配変換（推奨初期値は $h(z)=|z|$）

意図:

- TPで「Loss小・勾配大」の場合でも $g(L_k)$ が小さく、寄与を抑制
- FN/FPで「Loss大・勾配中程度」でも $g(L_k)$ が寄与を増幅

### 3. Filter Saliency への統合

フィルタ $f$ に属するパラメータ集合を $\Theta_f$ とすると、物体 $k$ のFilter Saliencyを

$$
S_{k,f}^{filter} = \operatorname{Agg}_{j\in\Theta_f} \; S_{k,j}^{param}
$$

で定義する。

実装推奨:

- Conv重みではチャネル・空間次元で平均（mean of abs-grad）
- `signed`解析時のみ符号付き平均を使う

---

## タイプ別Lossの設計

### TP

- Loss: 低いほど良い（通常ほぼ0）
- 期待挙動: 非HOT

### FN

- Loss: GT物体の未検出度を表す（大きいほど悪い）
- 期待挙動: HOT

### FP-A（位置型FP）

- Loss: GT非対応高信頼予測を罰する項
- 期待挙動: HOT

### FP-B（クラス混同型FP）

- Loss: GT重なり予測で誤クラス優勢を罰する項
- 期待挙動: HOT

例（概念）:

- $L_k^{fpB}=\max(0, s_{wrong,k}-s_{gt,k}+m)$

---

## 最終Filter Saliency合成

タイプ別・物体別Filter Saliencyを重み付き合成する。

$$
S_f^{final} =
w_{fn}\sum_k S_{k,f}^{fn}
+ w_{fpA}\sum_k S_{k,f}^{fpA}
+ w_{fpB}\sum_k S_{k,f}^{fpB}
+ w_{tp}\sum_k S_{k,f}^{tp}
$$

推奨初期値:

- $w_{tp}=0$（または極小）
- $w_{fn}>0,\; w_{fpA}>0,\; w_{fpB}>0$

これにより、設計仕様として TP非HOT を明示的に担保できる。

---

## 入力空間可視化との関係（補助出力）

入力空間ヒートマップが必要な場合は、Filter Saliencyを入力空間へ投影して作成する。

$$
S^{img}(x) = \sum_f S_f^{final}\,R_f(x)
$$

- $R_f(x)$: フィルタ $f$ に対応する入力空間への寄与マップ（活性ベースまたは勾配ベース）

注意:

- 本設計の主原理は $\partial L/\partial\theta$ であり、$\partial L/\partial x$ は補助的な投影手段

---

## 目的適合性の確認表

| タイプ | Loss大きさ | 重み/フィルタサリエンシー（本設計） |
|---|---:|---:|
| TP | 小 | 非HOT（抑制） |
| FN | 大 | HOT |
| FP-A | 大 | HOT |
| FP-B | 大 | HOT |

---

## 実装上の指針

- 画像単位平均より先に、必ず物体単位Lossを定義する
- サリエンシーは $\partial L/\partial\theta$ を基本とし、$g(L)$ で重み付けする
- Filter統合（aggregation）を固定し、比較時は同条件を維持する
- 最終合成で $w_{tp}=0$ または極小を維持する

---

## まとめ

本統合様式は、TP/FN/FP-A/FP-Bを分離しつつ同一フレームで扱える。
その上で、PSS原理に従って「物体単位Loss → 重みサリエンシー（$\partial L/\partial\theta$）→ Filter統合」を行うことで、
「FP/FNをHOTにする、TPをHOTにしない」を設計レベルで実現する。

---

## 設計詳細（実装用）

### A. 物体単位Lossの具体化

以下は実装時の基準形である。係数はデータセットに応じて調整する。

1. TP

$$
L_k^{tp} = -\log(s_{gt,k}+\epsilon)
$$

- $s_{gt,k}$ は物体 $k$ に対応する正解クラススコア
- 既に正しく検出できているTPでは小さい

2. FN

$$
L_k^{fn} = -\log(\text{covered\_score}_k+\epsilon)
$$

- covered_score はGT物体 $k$ に対応する検出カバレッジ
- 未検出ほど小さく、Lossは大きい

3. FP-A（位置型FP）

$$
L_k^{fpA} = (s_{pred,k})^p, \quad k \in \{\mathrm{IoU}(pred,GT)<\tau_{loc}\}
$$

- GT非対応の高信頼予測を直接罰する

4. FP-B（クラス混同型FP）

$$
L_k^{fpB} = \max(0, s_{wrong,k} - s_{gt,k} + m)
$$

- GTと重なる予測のうち、誤クラス優勢を罰する
- $m$ はマージン（通常 0.05-0.2）

### B. Lossゲート関数 $g(L)$

TP抑制とFP/FN強調の要である。初期導入は以下の2択が扱いやすい。

1. べき乗ゲート

$$
g(L)=L^{\alpha}, \quad \alpha \in [0.5,2.0]
$$

2. シグモイドゲート

$$
g(L)=\sigma(\beta(L-c))
$$

- $c$: 強調開始しきい値
- $\beta$: 立ち上がりの鋭さ

運用指針:

- TPが残る場合は $c$ を上げる、または $\alpha$ を上げる
- FP/FNが消える場合は $c$ を下げる、または $\alpha$ を下げる

### C. Filter統合（Aggregation）

実装比較の再現性を保つため、統合方法を固定する。

1. `filter_wise`（推奨）

- Conv重み（4D）に対し、入力チャネル・カーネル次元で平均
- 既定は絶対値平均（abs-mean）

2. `parameter_wise`

- すべての重みをフラット化して直接比較
- デバッグ用途で有効

### D. 推奨初期ハイパーパラメータ

- $w_{tp}=0.0$
- $w_{fn}=1.0$, $w_{fpA}=1.0$, $w_{fpB}=1.0$
- $\tau_{loc}=0.3$（FP-A判定）
- $m=0.1$（FP-Bマージン）
- $\epsilon=1e-8$
- $g(L)=L^{1.0}$ から開始
- aggregation: `filter_wise`
- signed: `false`（abs-grad）

### E. 推論時の処理フロー（擬似コード）

```text
for each image:
	detect predictions and match with GT
	split instances into TP, FN, FP-A, FP-B

	for each instance k in each type t:
		compute object-wise loss L_k^t
		compute parameter gradient dL_k^t/dtheta
		S_param[k, j] = g(L_k^t) * abs(dL_k^t/dtheta_j)
		S_filter[k, f] = aggregate_over_filter(S_param[k, :])

	S_final_filter[f] = w_fn  * sum_k S_filter_fn[k, f]
						 + w_fpA * sum_k S_filter_fpA[k, f]
						 + w_fpB * sum_k S_filter_fpB[k, f]
						 + w_tp  * sum_k S_filter_tp[k, f]

	(optional) project S_final_filter to image-space for visualization
```

### F. 評価プロトコル（目的適合性の定量化）

目的は「FP/FN HOT, TP non-HOT」であるため、まずFilter空間で評価し、必要に応じて入力空間指標を併記する。

1. Type-Hotness Ratio in Filter Space (FHR)

$$
\mathrm{FHR}_t = \frac{1}{N_t}\sum_{k\in t}\frac{1}{|\mathcal{F}|}\sum_{f\in\mathcal{F}}\mathbb{1}[S_{k,f}^{filter}>q_f]
$$

- $q_f$: フィルタサリエンシーの分位しきい値（例: 上位10%）

期待値:

- FHR_FN, FHR_FP-A, FHR_FP-B は高い
- FHR_TP は低い

2. Contrast Score

$$
C_{err/tp}=\frac{\operatorname{mean}(S^{filter}\mid FN,FP)}{\operatorname{mean}(S^{filter}\mid TP)+\epsilon}
$$

- $C_{err/tp}>1$ を維持する

3. 検証手順

- 同一画像集合でハイパーパラメータを固定比較
- 乱数seed固定
- TP抑制失敗時はゲートと $w_{tp}$ を優先調整

### G. よくある失敗と対処

1. TPが依然HOT

- 原因: $w_{tp}$ が非ゼロ、または $g(L)$ の閾値が低すぎる
- 対処: $w_{tp}=0$ を固定、$c$ または $\alpha$ を増加

2. FNが弱い

- 原因: FN用Lossの設計が弱く、$L_k^{fn}$ が十分に立たない
- 対処: covered_score定義を見直し、$w_{fn}$ またはゲート強度を増加

3. FP-Bが弱い

- 原因: 位置FP項のみでクラス混同を直接罰していない
- 対処: $L_k^{fpB}$ を有効化し、$w_{fpB}$ を増加

4. 全体がノイジー

- 原因: 勾配スケールの画像間差、またはaggregation条件の不一致
- 対処: abs-grad基準を固定し、比較時のaggregation/signed設定を統一
