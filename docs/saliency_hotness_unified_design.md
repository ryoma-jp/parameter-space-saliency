# FP/FNをHOTにし、TPをHOTにしない可視化: PSS論文準拠の物体検知拡張設計

## 目的

本設計の目的は次を同時に満たすこと。

- FP/FNをHOTにする
- TPをHOTにしない

ここでHOTは、フィルタ単位サリエンシー（Filter Saliency）が異常に高い状態を指す。

---

## 論文確認結果（PSSの原式）

arXiv:2108.01335 の主定義は以下。

1. パラメータ単位サリエンシー

$$
s(x,y)_i := \left|\nabla_{\theta_i}\mathcal{L}_{\theta}(x,y)\right|
$$

2. フィルタ単位集約（filter-wise average）

$$
\bar{s}(x,y)_f := \frac{1}{|\alpha_f|}\sum_{i\in\alpha_f} s(x,y)_i
$$

3. データセット基準の標準化

$$
\hat{s}(x,y) := \frac{\bar{s}(x,y)-\mu}{\sigma}
$$

- $\mu,\sigma$ は比較基準集合 $D$ 上の filter-wise saliency の平均と標準偏差
- 論文では「Lossゲート $g(L)$ の乗算」は導入していない

したがって、本拡張でも主計算は「勾配絶対値→フィルタ平均→標準化」を維持する。

---

## 物体検知向け拡張の基本方針

### 1. 推論結果を4タイプに分類

- TP: 正しい位置かつ正しいクラス
- FN: GTがあるのに未検出
- FP-A: GTがない領域への誤検知（位置型FP）
- FP-B: GTと重なるがクラス誤り（クラス混同型FP）

### 2. 物体単位Total LossでPSSを計算

画像全体Lossではなく、各物体 $k$ ごとに detector の通常損失項から Total Loss を定義する。

$$
L_k^{total} = L_{k}^{cls} + \lambda_{obj}L_{k}^{obj} + \lambda_{box}L_{k}^{box} + \lambda_{aux}L_{k}^{aux}
$$

- 各項は採用検出器の既存定義に従う（検出器ごとに項の有無・係数が異なる）
- 新規の独自罰則を主計算に混ぜず、まず論文準拠の勾配定義を優先する
- $\lambda_{aux}L_k^{aux}$ は補助的な正則化損失の汎用プレースホルダ（検出器に存在しない場合は0とする）

> **例: YOLOX の Total Loss**（`yolox/models/yolo_head.py` L.403–404）
>
> $$L^{total} = 5.0\,L_{iou} + L_{obj} + L_{cls} + L_{l1}$$
>
> | 項 | 対応 |
> |---|---|
> | $\lambda_{box}L^{box}$ | $5.0\,L_{iou}$（IoU Loss） |
> | $\lambda_{obj}L^{obj}$ | $L_{obj}$（BCE objectness, $\lambda=1$） |
> | $L^{cls}$ | $L_{cls}$（BCE classification） |
> | $\lambda_{aux}L^{aux}$ | $L_{l1}$（L1 box regression, ウォームアップ後に有効） |

物体 $k$ の parameter-wise saliency を

$$
s_k(i) := \left|\frac{\partial L_k^{total}}{\partial \theta_i}\right|
$$

として計算する。

> **FN 物体の Loss 計算について**
> FN は予測 bbox が存在しないため、検出器の通常 forward パスでは $L_k^{total}$ が直接得られない。
> **SimOTA 再実行方式（free anchor 限定）** を採用する。
>
> #### SimOTA 再実行方式
>
> 通常の SimOTA 実行後に得られる `fg_mask`（割り当て済みanchor）を除外したうえで、FN GT $k$ に対して SimOTA の cost matrix と dynamic-k 選択を再実行し、割り当てられた free anchor に対して Loss を計算する。
>
> ```text
> # 通常の SimOTA 実行済み（fg_mask が確定している前提）
> for each FN GT k:
>     candidate_anchors = all_anchors excluding fg_mask
>     compute pair_wise_ious(GT_k, candidate_anchors)
>     compute cost_matrix(GT_k, candidate_anchors)  # same formula as SimOTA
>     dynamic_k = estimate_dynamic_k(pair_wise_ious)
>     if dynamic_k == 0 or max_iou < tau_fn:
>         skip GT k (mark as "unresolvable FN", log it)
>     else:
>         fn_anchors_k = top-dynamic_k anchors by cost (lowest cost)
>         # 複数 FN GT が同一 free anchor を選んだ場合は cost 最小の GT のみに帰属
>         L_k_fn = sum of detector-native loss over fn_anchors_k
> ```
>
> **この方式の根拠**: SimOTA の cost matrix は「このanchorがこのGTを検出するコスト」そのものであり、競合がなければ学習時に割り当てられたはずの anchor を選ぶことになる。その anchor への勾配は「GT $k$ を正しく検出するためにパラメータがどう変わるべきか」を正しく指す。
>
> **成立しないケースと対処**:
>
> | 状況 | 対処 |
> |---|---|
> | geometry constraint 内に free anchor がゼロ（密集・完全オクルージョン） | 計算対象から除外してログに記録 |
> | free anchor との最大 IoU $< \tau_{fn}$（極小・画像端物体） | 同上 |
>
> $\tau_{fn}$ は初期値 $0.1$、比較実験では固定する。

### 3. 物体単位Filter Saliency

フィルタ $f$ の添字集合を $\alpha_f$ として、物体 $k$ の filter-wise saliency を

$$
\bar{s}_{k,f} := \frac{1}{|\alpha_f|}\sum_{i\in\alpha_f} s_k(i)
$$

で定義する。

---

## 標準化（TP基準）

本拡張での最重要修正点は、標準化統計をTP集合から推定すること。

TP物体集合を $\mathcal{T}_{tp}$ とし、各フィルタ $f$ について

$$
\mu^{tp}_f = \frac{1}{|\mathcal{T}_{tp}|}\sum_{k\in\mathcal{T}_{tp}} \bar{s}_{k,f}
$$

$$
\sigma^{tp}_f = \mathrm{Std}_{k\in\mathcal{T}_{tp}}\left[\bar{s}_{k,f}\right]
$$

と置き、全タイプに対して

$$
\hat{s}_{k,f} = \frac{\bar{s}_{k,f}-\mu^{tp}_f}{\sigma^{tp}_f+\epsilon}
$$

を用いる。

これにより、TPを「正常ベースライン」とした異常度として HOT を定義できる。

---

## 入力空間への投影（論文準拠）

### 論文の定式（Section 2.2）

標準化サリエンシー $\hat{s}(x,y)$ を出発点として、次の2ステップで入力空間マップを生成する。

**ステップ1: Boosted Saliency Profile の構成**

注目するフィルタ集合 $F$（上位 $|F|$ フィルタ）を選び、そのエントリのみを係数 $k$ で強調した参照プロファイルを作る。

$$
\bigl(s'_F\bigr)_f =
\begin{cases}
\hat{s}_f, & f \notin F,\\
k\,\hat{s}_f, & f \in F
\end{cases}
\quad (k > 1,\; \text{論文デフォルト } k=100)
$$

**ステップ2: 入力空間マップの計算**

「現在の $\hat{s}(x,y)$ を $s'_F$ に近づける方向の入力勾配の絶対値」を入力空間サリエンシーマップとする。

$$
M_F = \left|\nabla_x\, D_C\!\bigl(\hat{s}(x,y),\, s'_F\bigr)\right|
$$

- $D_C(\cdot,\cdot)$: コサイン距離
- $M_F$: フィルタ集合 $F$ を誤動作させている入力ピクセルのマップ

### 物体検知への適用方針

各物体 $k$ の標準化サリエンシー $\hat{s}_{k}$（全フィルタのベクトル）に対して上記を適用する。

1. 物体 $k$ ごとに $\hat{s}_k$ を計算する（TP基準で標準化済み）
2. FN / FP-A / FP-B 物体について、それぞれ上位 $|F|$ フィルタを選ぶ
3. $s'_F$ を構成し、$M_{F,k}=\left|\nabla_x D_C(\hat{s}_k, s'_{F,k})\right|$ を計算する
4. 複数物体のマップを重ねる場合は $M_F^{agg}=\sum_k M_{F,k}$ で合算する

TP 物体は $\hat{s}$ が標準化後に低いため $s'_F$ との乖離が小さく、自然に目立たない。 FN/FP 物体は標準化後に高い $\hat{s}$ をもち、入力マップにも強く現れる。

### 実装上の注意

- $F$ の選び方: 物体 $k$ ごとに $\hat{s}_{k}$ の上位 $|F|$ フィルタ（論文では $|F|=10$）
- $k$ の値: 論文デフォルト $100$、可視化のコントラストに応じて調整可
- 主指標はあくまで $\hat{s}_{k,f}$（パラメータ空間）、$M_F$ は補助的な入力空間可視化
- **二階微分について**: $\nabla_x D_C(\hat{s}(x,y), s'_F)$ は $\hat{s}(x,y) = \bar{s}(x,y)/\sigma^{tp}+\mathrm{const}$ が $x \to \mathcal{L} \to \nabla_\theta \mathcal{L}$ を経由して $x$ に依存するため、入力勾配の計算に `create_graph=True` が必要（論文実装も同様）

---

## 推論時フロー（更新版）

```text
for each image:
    run detection and match prediction/GT
    split objects into TP, FN, FP-A, FP-B

    # --- TP / FP-A / FP-B: use detector-native loss directly ---
    for each object k in TP, FP-A, FP-B:
        compute detector-native object total loss L_k_total
        compute parameter gradient abs: s_k(i)=|dL_k_total/dtheta_i|
        aggregate to filter-wise saliency: s_bar[k,f]

    # --- FN: SimOTA re-run on free anchors ---
    for each FN GT k:
        candidate_anchors = all_anchors excluding fg_mask
        rerun SimOTA cost matrix and dynamic-k on candidate_anchors
        if dynamic_k == 0 or max_iou < tau_fn:
            skip (log as unresolvable FN)
            continue
        resolve conflicts: each free anchor assigned to at most one FN GT (lowest cost wins)
        L_k_fn = detector-native loss over fn_anchors_k
        compute s_k(i)=|dL_k_fn/dtheta_i|
        aggregate to filter-wise saliency: s_bar[k,f]

accumulate TP objects from calibration split
compute mu_tp[f], sigma_tp[f] from TP filter saliency only

for each object k (all types):
    z-score normalize by TP stats: s_hat[k,f]
    # s_hat[k,f] ≈ 0 for TP by construction; large positive for FN/FP

# --- input-space projection (optional, per object k) ---
# requires second-order grad: retain computation graph through loss -> grad_theta -> s_bar
for each object k in FN/FP-A/FP-B:
    select top-|F| filters by s_hat[k,:]        # e.g. |F|=10
    build s_prime_F: s_hat[k,f] * k if f in F, else s_hat[k,f]  # k=100
    M_F_k = |grad_x cosine_distance(s_hat[k,:], s_prime_F)|  # s_hat as fn of x
M_F_agg = sum over k of M_F_k                   # per-image input saliency map
```

---

## 実装指針（本改訂で固定すること）

- PSS主計算は論文どおり $|\partial L/\partial\theta|$ と filter-wise mean を使う
- 独自ゲート関数 $g(L)$ は主計算から外す
- 物体単位Total Lossは detector の既存損失項で構成する
- 標準化の $\mu,\sigma$ は TP物体の Filter Saliency から計算する
- 比較実験では、同一の TP基準統計（同一split・同一seed）を使い回す

---

## 目的適合性の確認

TP基準の標準化後、以下を満たすことを確認する。

- TP群の平均 $\hat{s}$ が低い
- FN / FP-A / FP-B 群の平均 $\hat{s}$ が TP より高い
- フィルタ上位分位の占有率が TP < FN/FP

これにより、「FP/FNをHOT、TPをnon-HOT」という要件を、PSS論文準拠の枠組みで実現する。
