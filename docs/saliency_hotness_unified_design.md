# FP/FNをHOTにし、TPをHOTにしない可視化: 統合設計

## 目的

本設計の目的は以下を同時に満たすことである。

- FP/FNをHOTにする
- TPをHOTにしない

ここでのHOTは、最終サリエンシーマップにおいて強く可視化されることを指す。

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

単純な勾配のみでは目的を満たしにくいため、物体単位LossとLoss重み付けを導入する。

### 1. 物体単位Loss

各タイプ・各物体kに対して個別Lossを定義する。

- TP: L_k^tp
- FN: L_k^fn
- FP-A: L_k^fpA
- FP-B: L_k^fpB

要件:

- 望ましい状態では小さく、問題状態では大きくなるよう設計する
- 画像平均Lossではなく、物体単位Lossを基本とする

### 2. Loss重み付きサリエンシー

物体kの入力空間スコアを次で定義する。

$$
S_k(x) = g(L_k)\,\Big(\lambda\,\|\nabla_x L_k\| + (1-\lambda)\,P_k(x)\Big)
$$

- g(L_k): Loss大きさのゲート関数（単調増加）
- ||∇_x L_k||: 勾配ベース感度
- P_k(x): 空間事前分布（タイプ別マスク）
- \lambda in [0,1]: 勾配と空間事前の混合率

推奨例:

- g(L) = L^alpha（alpha > 0）または sigmoid(beta*(L-c))

意図:

- TPで「Loss小・勾配大」の場合でも g(L) が小さく、HOT抑制
- FN/FPで「Loss大・勾配小」の場合でも P_k(x) で局在を補強し、HOT化

---

## タイプ別Lossと空間事前の設計

### TP

- Loss: 低いほど良い（通常ほぼ0）
- 空間事前 P_k(x): 任意（使っても最終寄与は小さくする）
- 期待挙動: 非HOT

### FN

- Loss: GT物体の未検出度を表す（大きいほど悪い）
- 空間事前 P_k(x): GT box中心（必要なら周辺へ少し拡張）
- 期待挙動: GT領域をHOT

備考:

- 勾配のみだと拡散しやすいため、GT由来の空間事前が有効

### FP-A（位置型FP）

- Loss: GT非対応高信頼予測を罰する項
- 空間事前 P_k(x): 誤検出box領域
- 期待挙動: 誤検出領域をHOT

備考:

- 既存の fp_loc 系Lossと整合

### FP-B（クラス混同型FP）

- Loss: GT重なり予測で誤クラス優勢を罰する項
- 空間事前 P_k(x): 対応GT box領域
- 期待挙動: クラス混同を生んだ領域をHOT

例（概念）:

- L_k^fpB = max(0, s_wrong - s_gt + m) のようなマージン型

---

## 最終マップ合成

タイプ別物体スコアを重み付き合成する。

$$
S_final(x) =
w_fn\sum_k S_k^{fn}(x)
+ w_fpA\sum_k S_k^{fpA}(x)
+ w_fpB\sum_k S_k^{fpB}(x)
+ w_tp\sum_k S_k^{tp}(x)
$$

推奨初期値:

- w_tp = 0（または極小）
- w_fn > 0, w_fpA > 0, w_fpB > 0

これにより、設計仕様として TP非HOT を明示的に担保できる。

---

## 目的適合性の確認表

| タイプ | Loss大きさ | 勾配のみ | 本設計（Loss重み+空間事前） |
|---|---:|---:|---:|
| TP | 小 | 大きくなる場合あり | 非HOT（抑制） |
| FN | 大 | 小/拡散になりうる | HOT（GT局在） |
| FP-A | 大 | 比較的局在しやすい | HOT（誤検出局在） |
| FP-B | 大 | 拡散しうる | HOT（GT重なり局在） |

---

## 実装上の指針

- 画像単位平均より先に、必ず物体単位Lossを定義する
- 可視化スコアは勾配単体ではなく g(L) で重み付けする
- FN/FP-Bでは空間事前 P_k(x) を使って局在性を補強する
- 最終合成で w_tp を0または極小に固定する

---

## まとめ

本統合様式は、TP/FN/FP-A/FP-Bを分離しつつ同一フレームで扱える。
その上で、Loss大きさと空間局在を同時に使うことで、
「FP/FNをHOTにする、TPをHOTにしない」という目的を設計レベルで実現する。

---

## 設計詳細（実装用）

### A. 物体単位Lossの具体化

以下は実装時の基準形である。係数はデータセットに応じて調整する。

1. TP

$$
L_k^{tp} = -\log(s_{gt,k}+\epsilon)
$$

- $s_{gt,k}$ は物体kに対応する正解クラススコア
- 既に正しく検出できているTPでは小さくなる

2. FN

$$
L_k^{fn} = -\log(\text{covered\_score}_k+\epsilon)
$$

- covered_scoreはGT物体kに対応する検出カバレッジ
- 未検出ほど小さく、結果としてLossは大きい

3. FP-A（位置型FP）

$$
L_k^{fpA} = (s_{pred,k})^{p}, \quad k \in \{\text{IoU}(pred,GT)<\tau_{loc}\}
$$

- GT非対応の高信頼予測を直接罰する
- 既存のfp_loc系Lossと同型

4. FP-B（クラス混同型FP）

$$
L_k^{fpB} = \max(0, s_{wrong,k} - s_{gt,k} + m)
$$

- GTと重なる予測のうち、誤クラス優勢を罰する
- $m$ はマージン（通常 0.05-0.2 程度）

### B. Lossゲート関数 g(L)

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

### C. 空間事前 P_k(x) の設計

勾配が拡散するケースへの対策として、タイプ別の空間マスクを導入する。

1. TP/FN/FP-B

- 対応GT boxを中心とする2Dガウシアンまたはbox一様マスク
- 境界影響を減らすため、boxを1.05-1.2倍へわずかに拡張可能

2. FP-A

- 誤検出boxを中心に同様のマスクを適用

正規化:

$$
\sum_x P_k(x)=1
$$

とし、スケール依存を抑える。

### D. 推奨初期ハイパーパラメータ

- $\lambda = 0.6$（勾配寄り、ただし事前も有効化）
- $w_{tp}=0.0$
- $w_{fn}=1.0$, $w_{fpA}=1.0$, $w_{fpB}=1.0$
- $\tau_{loc}=0.3$（FP-A判定）
- $m=0.1$（FP-Bマージン）
- $\epsilon=1e-8$
- $g(L)=L^{1.0}$ から開始

### E. 推論時の処理フロー（擬似コード）

```text
for each image:
	detect predictions and match with GT
	split instances into TP, FN, FP-A, FP-B

	for each instance k in each type t:
		compute object-wise loss L_k^t
		compute gradient map G_k^t = ||dL_k^t / dx||
		build prior map P_k^t(x)
		S_k^t(x) = g(L_k^t) * (lambda * G_k^t + (1-lambda) * P_k^t(x))

	S_final(x) = w_fn * sum(S_k^fn)
						 + w_fpA * sum(S_k^fpA)
						 + w_fpB * sum(S_k^fpB)
						 + w_tp * sum(S_k^tp)

	normalize S_final for visualization
```

### F. 評価プロトコル（目的適合性の定量化）

目的は「FP/FN HOT, TP non-HOT」であるため、IoU指標だけでなくHOT率を測る。

1. Type-Hotness Ratio (THR)

$$
	ext{THR}_t = \frac{1}{N_t}\sum_{k \in t}\frac{1}{|B_k|}\sum_{x \in B_k}\mathbb{1}[S_{final}(x) > q]
$$

- $B_k$: 物体kの対応領域（GT boxまたはpred box）
- $q$: 画像内分位点しきい値（例: 上位10%）

期待値:

- THR_FN, THR_FP-A, THR_FP-B は高い
- THR_TP は低い

2. Contrast Score

$$
C_{err/tp} = \frac{\text{mean}(S|FN,FP)}{\text{mean}(S|TP)+\epsilon}
$$

- $C_{err/tp} > 1$ を維持する

3. 検証手順

- 同一画像集合でハイパーパラメータを固定比較
- 乱数seed固定
- TP抑制失敗時はゲート優先で再調整

### G. よくある失敗と対処

1. TPが依然HOT

- 原因: $w_{tp}$ が非ゼロ、または $g(L)$ の閾値が低すぎる
- 対処: $w_{tp}=0$ を固定、$c$ または $\alpha$ を増加

2. FNが出ない

- 原因: 勾配拡散で局在消失
- 対処: $\lambda$ を下げ、$P_k(x)$ 比率を上げる

3. FP-Bが弱い

- 原因: fp_locのみでクラス混同を直接罰していない
- 対処: $L_k^{fpB}$ を有効化し、$w_{fpB}$ を増加

4. 全体がノイジー

- 原因: 勾配スケールの画像間差
- 対処: 画像内正規化（分位クリップ+min-max）を統一
