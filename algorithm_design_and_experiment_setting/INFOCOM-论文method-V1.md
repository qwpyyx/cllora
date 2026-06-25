# INFOCOM 2027 -- Method V1：Signal-Noise-Aware Sparse FedLoRA Upload

> 本文档给出一版与最新版 motivation 对齐的 method 叙事。  
> 核心目标是让方法自然回应 motivation 中的两个现象：
>
> 1. Tensor-level Top-K 破坏 LoRA A/B 乘法结构，因此上传原子必须是完整 A/B pair；
> 2. 即使使用 A/B pair induced effective update，magnitude-only Top-K 仍会导致跨客户端选择高度集中，因此需要 signal-noise-aware quota allocation 和 client-level complementary assignment。

---

## 0. Method Overview

本文提出一种 server-assisted sparse upload scheduler for federated LoRA fine-tuning。

在每一轮中，参与 clients 先在本地训练得到 LoRA factor updates。由于通信预算有限，server 无法接收所有 LoRA modules 的完整 A/B updates。本文的核心问题是：

> Given a limited uplink budget, which client-module LoRA updates should be uploaded so that the sparse aggregate preserves the useful global adaptation signal?

motivation 已经表明，简单的 client-side Top-K 选择有两个问题：

1. 如果把 LoRA A/B tensors 当作独立上传单元，会破坏 effective update 的结构；
2. 如果按 effective-update magnitude 独立 Top-K，则所有 clients 会倾向于上传同一批 modules，造成严重重复和低覆盖。

因此，本文不让每个 client 独立做 Top-K。相反，server 使用轻量 effective-update statistics 来进行两层调度：

1. **P1: module-level quota allocation**  
   server 估计每个 module 的 shared signal strength 和 heterogeneity noise，决定该 module 应该由多少个 clients 上传，即 quota $k_m^*$。

2. **P2: client-level assignment**  
   给定 quota $k_m^*$，server 再决定具体哪些 clients 上传哪些 modules，即 binary assignment $z_{i,m}^*$。

这两个阶段共同解决 Fig.2 暴露的问题：通信预算不应被 magnitude-only ranking 重复分配到同一批 modules，而应根据 signal-noise marginal gain 判断每一次重复上传是否仍然值得。

---

## 1. Notation and Effective-Update Atom

第 $t$ 轮有 $K$ 个参与 clients。为简化符号，下文省略轮次 $t$。

| Symbol | Meaning |
|---|---|
| $K$ | 当前轮参与 clients 数 |
| $M$ | LoRA modules 数 |
| $i\in\{1,\dots,K\}$ | client index |
| $m\in\{1,\dots,M\}$ | module index |
| $A_m\in\mathbb{R}^{r\times d_{\mathrm{in},m}}$ | 当前全局 LoRA A factor |
| $B_m\in\mathbb{R}^{d_{\mathrm{out},m}\times r}$ | 当前全局 LoRA B factor |
| $U_{A,i,m},U_{B,i,m}$ | client $i$ 在 module $m$ 上的 local factor updates |
| $\Phi_{i,m}$ | client $i$ 在 module $m$ 上诱导的 effective update |
| $z_{i,m}\in\{0,1\}$ | client $i$ 是否上传 module $m$ 的完整 A/B pair |
| $k_m=\sum_i z_{i,m}$ | module $m$ 被多少 clients 上传 |
| $c_m$ | 上传 module $m$ 的通信成本 |
| $C_i$ | client $i$ 的通信预算 |
| $B_i$ | equal-cost setting 下 client $i$ 的上传 slot 数 |
| $s$ | sketch dimension |

client $i$ 在 module $m$ 上的完整 LoRA update 是：

$$
U_{i,m}=(U_{A,i,m},U_{B,i,m}).
$$

它诱导的 effective weight update 为：

$$
\Phi_{i,m}
=
(B_m+U_{B,i,m})(A_m+U_{A,i,m})-B_mA_m.
$$

展开得到：

$$
\Phi_{i,m}
=
B_mU_{A,i,m}
+U_{B,i,m}A_m
+U_{B,i,m}U_{A,i,m}.
$$

本文把完整 A/B pair induced effective update $\Phi_{i,m}$ 作为 sparse upload 的基本原子。也就是说，$z_{i,m}=1$ 表示上传 module $m$ 的完整 A/B pair；$z_{i,m}=0$ 表示该 client-module update 不上传。

这一设计首先保证 **structural consistency**：server 不会只收到 $U_A$ 或 $U_B$ 的一半更新，从而避免 motivation 中 Fig.1 所示的 tensor-level Top-K failure。

进一步地，本文后续所有 utility、signal、noise 和 assignment score 都定义在 $\Phi_{i,m}$ 所在的 effective-update space 中，而不是定义在 raw factor norm

$$
\|U_{A,i,m}\|_F^2+\|U_{B,i,m}\|_F^2
$$

上。这样避免了 LoRA factor reparameterization 带来的 representation-dependent saliency。

---

## 2. Why Independent Top-K Is Not Enough

即使上传原子改成完整 A/B pair，仍然不能简单地让每个 client 独立选择 effective-update norm 最大的 modules：

$$
\|\Phi_{i,m}\|_F^2.
$$

motivation 中的 Fig.2 表明，effective magnitude-only Top-K 会让不同 clients 选择几乎完全相同的 modules。其原因是，在 federated setting 中，每个 client 的 effective update 可以写成：

$$
\Phi_{i,m}=\mu_m+\xi_{i,m},
$$

其中 $\mu_m$ 是跨 clients 共享的 update direction，$\xi_{i,m}$ 是 client-specific deviation。

于是：

$$
\mathbb{E}\|\Phi_{i,m}\|_F^2
=
\underbrace{\|\mu_m\|_F^2}_{a_m}
+
\underbrace{\mathbb{E}\|\xi_{i,m}\|_F^2}_{b_m}.
$$

Magnitude-only Top-K 把 $a_m$ 和 $b_m$ 都当作正面贡献，但二者在全局聚合中的含义不同：

- $a_m$ 表示 shared descent signal，应被优先保留；
- $b_m$ 表示 heterogeneity noise 或 client-specific deviation，不应被无条件奖励；
- 同一个 module 被多个 clients 重复上传时，边际收益递减。

因此，低预算 FedLoRA sparse upload 的关键不是独立 ranking，而是 **allocation**：

> For each module, how many client uploads are useful, and which clients should provide them?

这正是 P1/P2 的作用。

---

## 3. Ideal Objective: Dense Effective-Update Preservation

如果没有通信限制，dense FedAvg effective update 为：

$$
\Delta_{\mathrm{dense}}
=
\frac{1}{K}\sum_{i=1}^{K}\sum_{m=1}^{M}\Phi_{i,m}.
$$

如果 sparse upload 选择 $z_{i,m}$，则 sparse aggregate 为：

$$
\Delta_{\mathrm{sp}}
=
\frac{1}{K}\sum_{i=1}^{K}\sum_{m=1}^{M}z_{i,m}\Phi_{i,m}.
$$

注意这里保持 dense FedAvg 的 $1/K$ scaling，而不是对已上传 clients 重新归一化。未上传的 client-module update 被视为 zero contribution。这样定义的 sparse aggregate 直接表示 sparse upload 对 dense full-upload FedAvg update 的保留程度。

理想原始问题是：

$$
(P)\quad
\min_{\{z_{i,m}\}}
\left\|
\Delta_{\mathrm{dense}}-\Delta_{\mathrm{sp}}
\right\|_F^2
$$

subject to:

$$
\sum_{m=1}^{M}z_{i,m}c_m\le C_i,\qquad i=1,\dots,K,
$$

$$
z_{i,m}\in\{0,1\}.
$$

这个目标是合理的：通信受限时，我们希望 sparse update 尽可能保留 full-upload FedLoRA 的 effective model update。

但是 $P$ 不能直接求解：

1. 它是带 per-client budget 的 binary combinatorial optimization；
2. 目标展开后包含大量 pairwise inner products $\langle\Phi_{i,m},\Phi_{j,n}\rangle_F$，是 quadratic selection problem；
3. 精确知道 $\Delta_{\mathrm{dense}}$ 需要接收所有 $\Phi_{i,m}$，这本身等价于 full upload；
4. $\Phi_{i,m}$ 是高维 effective update matrix，不能直接上传和比较。

因此，$P$ 是本文的理想目标，而实际算法需要一个可估计、可分解、可调度的 statistical relaxation。

---

## 4. Signal-Noise Statistical Relaxation

对每个 module $m$，假设 client effective update 可分解为：

$$
\Phi_{i,m}=\mu_m+\xi_{i,m},
$$

其中：

- $\mu_m=\mathbb{E}_i[\Phi_{i,m}]$ 表示跨 clients 共享的 module-level update direction；
- $\xi_{i,m}$ 表示 client-specific deviation，包括 non-IID bias、本地随机性和局部训练偏差。

采用标准零均值假设：

$$
\mathbb{E}[\xi_{i,m}]=0,
$$

并定义：

$$
a_m=\|\mu_m\|_F^2,
\qquad
b_m=\mathbb{E}\|\xi_{i,m}\|_F^2.
$$

其中 $a_m$ 是 shared signal strength，$b_m$ 是 heterogeneity noise strength。

这里的 "noise" 不是说 $\xi_{i,m}$ 对本地任务没有价值，而是说它相对于全局 shared update direction 是 client-specific deviation。在低预算 global aggregation 中，我们希望优先保留跨 clients 一致的 shared descent signal，同时避免把高方差的局部偏差当作全局重要性。

基于这个模型，本文将 $P$ 放松为 shared-signal preservation：

$$
(P_0)\quad
\min_{\{z_{i,m}\}}
\sum_{m=1}^{M}
\left\|
\mu_m-
\frac{1}{K}\sum_{i=1}^{K}z_{i,m}\Phi_{i,m}
\right\|_F^2
$$

subject to the same communication constraints.

$P_0$ 不是任意换目标，而是对 dense effective-update preservation 的 statistical relaxation：它把 dense aggregate 中稳定一致的 shared component 作为低预算下优先保留的对象，并将 client-specific deviation 作为需要控制的方差项。

---

## 5. P1: Module-Level Signal-Noise Allocation

P0 同时决定 module-level quota 和 client-level assignment，仍然难以直接求解。因此，本文先忽略具体由哪些 clients 上传，只决定每个 module 应该被上传多少次。

定义：

$$
k_m=\sum_{i=1}^{K}z_{i,m}.
$$

如果 module $m$ 被 $k_m$ 个 clients 上传，selected clients 为：

$$
\mathcal{S}_m=\{i:z_{i,m}=1\},
\qquad |\mathcal{S}_m|=k_m.
$$

保持 $1/K$ scaling，则 module-level sparse aggregate 为：

$$
\widehat{\Delta}_m
=
\frac{1}{K}\sum_{i\in\mathcal{S}_m}\Phi_{i,m}.
$$

代入 $\Phi_{i,m}=\mu_m+\xi_{i,m}$：

$$
\widehat{\Delta}_m
=
\frac{k_m}{K}\mu_m
+
\frac{1}{K}\sum_{i\in\mathcal{S}_m}\xi_{i,m}.
$$

因此 residual 为：

$$
\mu_m-\widehat{\Delta}_m
=
\left(1-\frac{k_m}{K}\right)\mu_m
-
\frac{1}{K}\sum_{i\in\mathcal{S}_m}\xi_{i,m}.
$$

对平方 Frobenius norm 取期望，并假设不同 clients 的 noise cross terms 期望为 0，得到：

$$
J_m(k_m)
=
\mathbb{E}
\left\|
\mu_m-\widehat{\Delta}_m
\right\|_F^2
=
\left(1-\frac{k_m}{K}\right)^2a_m
+
\frac{k_m}{K^2}b_m.
$$

第一项是 missing shared signal error，随 $k_m$ 增大而下降；第二项是 accumulated heterogeneity noise，随 $k_m$ 增大而上升。

于是 module-level allocation problem 为：

$$
(P_1)\quad
\min_{\{k_m\}}
\sum_{m=1}^{M}J_m(k_m)
$$

subject to:

$$
\sum_{m=1}^{M}c_mk_m\le C_{\mathrm{total}},
\qquad
0\le k_m\le K,
\qquad
k_m\in\mathbb{Z},
$$

where $C_{\mathrm{total}}=\sum_i C_i$.

### 5.1 Marginal Gain

定义第 $k+1$ 次上传 module $m$ 的边际收益为：

$$
\Delta_m(k)=J_m(k)-J_m(k+1),
\qquad k=0,\dots,K-1.
$$

由 $J_m(k)$ 可得：

$$
\Delta_m(k)
=
\frac{(2(K-k)-1)a_m-b_m}{K^2}.
$$

该式直接解释 Fig.2 中 magnitude-only concentration 的问题：

1. $a_m$ 是正贡献，表示 shared signal；
2. $b_m$ 是负贡献，表示 heterogeneity noise；
3. 随着 $k$ 增大，$\Delta_m(k)$ 严格递减：

$$
\Delta_m(k+1)-\Delta_m(k)
=
-\frac{2a_m}{K^2}.
$$

也就是说，同一个 module 被更多 clients 上传后，继续上传该 module 的新增收益会变小；当

$$
(2(K-k)-1)a_m\le b_m
$$

时，继续上传该 module 的边际收益甚至不再为正。

这正是 magnitude-only Top-K 看不到的信息：它无法判断某个 module 已经被多个 clients 上传后，下一次重复上传是否仍然值得。

### 5.2 Equal-Cost Solution

在主文的 equal-cost setting 下，所有 A/B pair 的通信成本相同。预算等价于最多上传 $B$ 个 A/B pairs：

$$
\sum_m k_m\le B.
$$

此时 P1 等价于从所有 candidate marginal gains

$$
\{\Delta_m(0),\Delta_m(1),\dots,\Delta_m(K-1):m=1,\dots,M\}
$$

中选择最大的 $B$ 个正值。由于 $\Delta_m(k)$ 对 $k$ 单调递减，选择 top marginal gains 会自动满足 prefix constraint：如果选择了第 $k+1$ 次上传，那么前 $k$ 次上传的 marginal gain 更大，也会被优先选择。

因此 equal-cost P1 的解法是：

1. 对每个 module $m$ 计算 $\Delta_m(k)$；
2. 选择最大的 $B$ 个正 marginal gains；
3. 对每个 module 统计被选中的 marginal gains 数量，得到 quota $k_m^*$。

如果预算必须用满，则选择前 $B$ 个；如果预算只是上限，则只选择正收益的前 $B$ 个。

### 5.3 Unequal-Cost Extension

若不同 modules 的通信成本 $c_m$ 不同，可考虑 continuous relaxation：

$$
k_m\in[0,K].
$$

P1 relaxation 为 convex optimization，因为：

$$
J_m''(k_m)=\frac{2a_m}{K^2}\ge0.
$$

KKT 条件给出：

$$
k_m(\lambda)
=
\left[
K-
\frac{b_m+\lambda c_mK^2}{2a_m}
\right]_0^K,
$$

其中 $[x]_0^K=\min(\max(x,0),K)$，$\lambda$ 由预算约束决定：

$$
\sum_m c_mk_m(\lambda)=C_{\mathrm{total}}.
$$

该解具有直观解释：$a_m$ 越大，quota 越多；$b_m$ 越大，quota 越少；$c_m$ 越大，quota 越少。主文可重点采用 equal-cost setting，将 unequal-cost 放入扩展或 appendix。

---

## 6. P2: Client-Level Complementary Assignment

P1 只决定每个 module 应该被上传多少次：

$$
k_m^*.
$$

但它没有决定具体由哪些 clients 上传。因此需要 P2 来恢复 binary assignment $z_{i,m}$。

完整 P2 为：

$$
(P_2)\quad
\min_{\{z_{i,m}\}}
\sum_{m=1}^{M}
\left\|
\mu_m-
\frac{1}{K}\sum_{i=1}^{K}z_{i,m}\Phi_{i,m}
\right\|_F^2
$$

subject to:

$$
\sum_{i=1}^{K}z_{i,m}=k_m^*,
\qquad m=1,\dots,M,
$$

$$
\sum_{m=1}^{M}z_{i,m}\le B_i,
\qquad i=1,\dots,K,
$$

$$
z_{i,m}\in\{0,1\}.
$$

P2 的作用是 complementary assignment：在满足 P1 quota 的前提下，为每个 module 选择一组更能代表 shared signal、且彼此不过度冗余的 clients。

### 6.1 Residual Expansion

对单个 module $m$，记 selected clients 为 $\mathcal{S}$。残差为：

$$
R_m(\mathcal{S})
=
\left\|
\mu_m-
\frac{1}{K}\sum_{i\in\mathcal{S}}\Phi_{i,m}
\right\|_F^2.
$$

展开平方：

$$
R_m(\mathcal{S})
=
\|\mu_m\|_F^2
-
\frac{2}{K}\sum_{i\in\mathcal{S}}
\langle\mu_m,\Phi_{i,m}\rangle_F
+
\frac{1}{K^2}
\left\|
\sum_{i\in\mathcal{S}}\Phi_{i,m}
\right\|_F^2.
$$

继续展开最后一项：

$$
\left\|
\sum_{i\in\mathcal{S}}\Phi_{i,m}
\right\|_F^2
=
\sum_{i\in\mathcal{S}}\|\Phi_{i,m}\|_F^2
+
2\sum_{i<j,\ i,j\in\mathcal{S}}
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F.
$$

于是：

$$
R_m(\mathcal{S})
=
\|\mu_m\|_F^2
-
\sum_{i\in\mathcal{S}}s_{i,m}
+
I_m(\mathcal{S}),
$$

其中 individual score 为：

$$
s_{i,m}
=
\frac{2}{K}\langle\mu_m,\Phi_{i,m}\rangle_F
-
\frac{1}{K^2}\|\Phi_{i,m}\|_F^2,
$$

pairwise interaction penalty 为：

$$
I_m(\mathcal{S})
=
\frac{2}{K^2}
\sum_{i<j,\ i,j\in\mathcal{S}}
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F.
$$

由于 $\|\mu_m\|_F^2$ 与选择无关，minimizing residual 等价于 maximizing：

$$
F_m(\mathcal{S})
=
\sum_{i\in\mathcal{S}}s_{i,m}
-
I_m(\mathcal{S}).
$$

这说明完整 P2 本质上是 quadratic assignment：既要选择 individual gain 高的 client-module pairs，也要避免 selected clients 在同一 module 上过度同向、重复。

### 6.2 Linearized P2-L

如果忽略 interaction term $I_m(\mathcal{S})$，得到 linear surrogate：

$$
(P_{2\text{-L}})\quad
\max_{\{z_{i,m}\}}
\sum_{i=1}^{K}\sum_{m=1}^{M}z_{i,m}s_{i,m}
$$

subject to:

$$
\sum_i z_{i,m}=k_m^*,
\qquad
\sum_m z_{i,m}\le B_i,
\qquad
z_{i,m}\in\{0,1\}.
$$

这个 score 不是 heuristic Top-K，而是从 P2 residual 展开得到的一阶 linear term。

### 6.3 Gap-Aware P2-L

Plain P2-L 忽略了 selected clients 之间的 redundancy。为近似弥补 interaction gap，定义 positive interaction exposure：

$$
d_{i,m}^+
=
\sum_{j\ne i}
\max\left(
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F,
0
\right).
$$

对于任意 selected set $\mathcal{S}$，其 positive interaction 部分满足：

$$
I_m^+(\mathcal{S})
\le
\frac{1}{K^2}
\sum_{i\in\mathcal{S}}d_{i,m}^+.
$$

因此构造 separable correction：

$$
\tilde{s}_{i,m}
=
s_{i,m}
-
\frac{\eta}{K^2}d_{i,m}^+,
$$

其中 $\eta\in[0,1]$ 控制 redundancy correction strength。最终 gap-aware P2-L 为：

$$
\max_{\{z_{i,m}\}}
\sum_{i=1}^{K}\sum_{m=1}^{M}z_{i,m}\tilde{s}_{i,m}
$$

with the same quota and budget constraints.

---

## 7. Leave-One-Out Shared Direction

P2 score 中需要 $\langle\mu_m,\Phi_{i,m}\rangle_F$。实际估计时，如果直接使用所有 clients 的平均：

$$
\hat\mu_m=\frac{1}{K}\sum_{j=1}^{K}\Phi_{j,m},
$$

则 $\langle\hat\mu_m,\Phi_{i,m}\rangle_F$ 包含 self-inner-product $\|\Phi_{i,m}\|_F^2$，会把 alignment score 拉回 magnitude-based selection。

因此本文使用 leave-one-out shared direction：

$$
\hat\mu_{-i,m}
=
\frac{1}{K-1}\sum_{j\ne i}\Phi_{j,m}.
$$

对应 alignment 估计为：

$$
\langle\mu_m,\Phi_{i,m}\rangle_F
\approx
\langle\hat\mu_{-i,m},\Phi_{i,m}\rangle_F.
$$

这使 P2 更关注 client $i$ 的 update 是否与其他 clients 的 shared direction 一致，而不是奖励它自己的 magnitude。

---

## 8. Sketch-Based Effective-Update Statistics

P1 和 P2 需要估计：

$$
\|\Phi_{i,m}\|_F^2,
\qquad
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F,
\qquad
\langle\mu_m,\Phi_{i,m}\rangle_F.
$$

直接上传完整 $\Phi_{i,m}$ 不可行，因此本文使用 shared normalized Rademacher bilinear sketch。

### 8.1 Bilinear Sketch

对每个 module $m$，server 和 clients 共享随机种子，生成 $s$ 组随机向量：

$$
\nu_{\ell,m}\in\mathbb{R}^{d_{\mathrm{out},m}},
\qquad
v_{\ell,m}\in\mathbb{R}^{d_{\mathrm{in},m}},
\qquad
\ell=1,\dots,s.
$$

其中：

$$
\nu_{\ell,m}[p]\in\left\{\pm\frac{1}{\sqrt{d_{\mathrm{out},m}}}\right\},
\qquad
v_{\ell,m}[q]\in\left\{\pm\frac{1}{\sqrt{d_{\mathrm{in},m}}}\right\}.
$$

client $i$ 对 module $m$ 计算：

$$
r_{i,m}[\ell]
=
\sqrt{d_{\mathrm{out},m}d_{\mathrm{in},m}}
\cdot
\nu_{\ell,m}^{\top}\Phi_{i,m}v_{\ell,m}.
$$

于是 $r_{i,m}\in\mathbb{R}^s$。

对于任意两个 matrices $X,Y$，该 sketch 满足：

$$
\mathbb{E}
\left[
r_X[\ell]r_Y[\ell]
\right]
=
\langle X,Y\rangle_F.
$$

因此：

$$
\widehat{\langle X,Y\rangle_F}
=
\frac{1}{s}\langle r_X,r_Y\rangle.
$$

### 8.2 Low-Rank Computation

client 不需要显式构造 $\Phi_{i,m}$。由于：

$$
\Phi_{i,m}
=
B_mU_{A,i,m}
+U_{B,i,m}A_m
+U_{B,i,m}U_{A,i,m},
$$

对任意 sketch vectors $\nu,v$：

$$
\nu^\top\Phi_{i,m}v
=
(\nu^\top B_m)(U_{A,i,m}v)
+
(\nu^\top U_{B,i,m})(A_mv)
+
(\nu^\top U_{B,i,m})(U_{A,i,m}v).
$$

这只需要低秩乘法，单个 module 的 sketch 计算复杂度为：

$$
O\left(sr(d_{\mathrm{in},m}+d_{\mathrm{out},m})\right).
$$

---

## 9. Estimating P1 and P2 Quantities

每个 client 对每个 module 上传轻量统计量：

$$
q_{i,m}=\|\Phi_{i,m}\|_F^2,
\qquad
r_{i,m}\in\mathbb{R}^{s}.
$$

调度阶段上传的是 $\{q_{i,m},r_{i,m}\}_{m=1}^{M}$，不是完整 LoRA A/B updates，也不是完整 effective matrices。

### 9.1 Shared Signal Strength

server 使用 pairwise inner product 估计：

$$
\hat a_m
=
\frac{
\left\|\sum_{i=1}^{K}r_{i,m}\right\|^2
-
\sum_{i=1}^{K}\|r_{i,m}\|^2
}{
sK(K-1)
}.
$$

实际中做非负截断：

$$
\hat a_m\leftarrow\max(\hat a_m,0).
$$

### 9.2 Noise Strength

平均 total magnitude 为：

$$
\hat q_m
=
\frac{1}{K}\sum_{i=1}^{K}q_{i,m}.
$$

估计 heterogeneity noise：

$$
\hat b_m
=
\hat q_m-\hat a_m.
$$

并做截断：

$$
\hat b_m\leftarrow\max(\hat b_m,\epsilon).
$$

### 9.3 P2 Score in Sketch Space

sketch-space leave-one-out direction 为：

$$
\hat r_{\mu,-i,m}
=
\frac{1}{K-1}\sum_{j\ne i}r_{j,m}.
$$

alignment 估计为：

$$
\widehat{\langle\mu_m,\Phi_{i,m}\rangle_F}
=
\frac{1}{s}
\langle\hat r_{\mu,-i,m},r_{i,m}\rangle.
$$

positive interaction exposure 估计为：

$$
\hat d_{i,m}^+
=
\sum_{j\ne i}
\max\left(
\frac{1}{s}\langle r_{i,m},r_{j,m}\rangle,
0
\right).
$$

最终 gap-aware score 为：

$$
\tilde{s}_{i,m}
=
\frac{2}{Ks}
\langle\hat r_{\mu,-i,m},r_{i,m}\rangle
-
\frac{1}{K^2}q_{i,m}
-
\frac{\eta}{K^2}\hat d_{i,m}^+.
$$

该 score 同时包含：

1. 与其他 clients shared direction 的 alignment；
2. 自身 update norm 对 residual 的 penalty；
3. 与其他 clients 的 positive redundancy correction。

---

## 10. Solving P2-L by Min-Cost Flow

在 equal-cost setting 下，P2-L 是标准 bipartite b-matching，可写成 min-cost flow：

- source 到每个 client 节点，capacity 为 $B_i$；
- client $i$ 到 module $m$，capacity 为 1，cost 为 $-\tilde{s}_{i,m}$；
- module $m$ 到 sink，demand/capacity 为 $k_m^*$。

求最小费用最大流等价于最大化：

$$
\sum_{i,m}z_{i,m}\tilde{s}_{i,m}.
$$

如果由于 heterogeneous budgets 或 pruning 导致 quota 不可行，可以使用 slack edges 允许：

$$
\sum_i z_{i,m}\le k_m^*,
$$

并对 unmet quota 加 penalty。主文可先呈现 feasible equal-cost version，复杂实现细节放入 appendix。

---

## 11. Full Algorithm

**Algorithm: Signal-Noise-Aware Sparse FedLoRA Upload**

输入：当前全局 LoRA factors $\{A_m,B_m\}_{m=1}^{M}$，参与 clients $\mathcal{K}$，client budgets $\{B_i\}$，sketch dimension $s$，correction strength $\eta$。

1. **Local training**：每个 client $i$ 在本地数据上训练，得到 $\{U_{A,i,m},U_{B,i,m}\}_{m=1}^{M}$。
2. **Effective-update statistics**：每个 client 对每个 module 计算 $q_{i,m}=\|\Phi_{i,m}\|_F^2$ 和 sketch $r_{i,m}$。
3. **Lightweight statistics upload**：client 上传 $\{q_{i,m},r_{i,m}\}_{m=1}^{M}$。
4. **Signal-noise estimation**：server 估计 $\hat a_m$ 和 $\hat b_m$。
5. **P1 allocation**：server 根据 marginal gain

$$
\hat\Delta_m(k)
=
\frac{(2(K-k)-1)\hat a_m-\hat b_m}{K^2}
$$

求解 module quota $k_m^*$。

6. **P2 score computation**：server 计算 leave-one-out alignment、positive redundancy exposure 和 $\tilde{s}_{i,m}$。
7. **Gap-aware assignment**：server 用 min-cost flow 求解 $z_{i,m}^*$。
8. **Assignment broadcast**：server 向每个 client 下发应上传的 module index set

$$
\mathcal{S}_i=\{m:z_{i,m}^*=1\}.
$$

9. **Sparse A/B upload**：client 只上传被选中的完整 A/B pairs。
10. **Aggregation**：server 按 $1/K$ scaling 聚合 selected A/B updates，未上传 entries 视为 zero contribution。

---

## 12. Online and Pipelined Scheduling

严格 online 版本需要一个轻量 scheduling phase：

1. client 本地训练；
2. client 上传 statistics；
3. server 求解 P1/P2；
4. server 下发 assignment；
5. client 上传 selected A/B pairs；
6. server 聚合。

这会增加一次控制交互，但上传的 statistics 很小。

为了减少额外 RTT，可以采用 pipelined scheduling：server 使用历史 EMA statistics 预测当前轮的 quota 和 assignment，当前轮收到新 statistics 后更新下一轮调度。主文可以先描述 online version，因为它最清晰；系统讨论中说明 pipelined version 可降低控制延迟。

---

## 13. Communication and Computation Cost

### 13.1 Communication

每个 client 每个 module 上传：

$$
q_{i,m}\in\mathbb{R},
\qquad
r_{i,m}\in\mathbb{R}^{s}.
$$

因此调度统计上传量为：

$$
O(M(s+1))
$$

per client，所有 clients 合计为：

$$
O(KM(s+1)).
$$

若使用 fp16，字节数约为：

$$
2KM(s+1)\ \mathrm{bytes}.
$$

由于 $s$ 是小常数，例如 $s=16$，该调度开销远小于 selected LoRA A/B payload：

$$
O\left(
\sum_{i,m}z_{i,m}r(d_{\mathrm{in},m}+d_{\mathrm{out},m})
\right).
$$

server 下发的 assignment 只是 module indices，下行开销可忽略。

### 13.2 Client-Side Computation

利用低秩结构，client 对所有 modules 计算 sketches 的复杂度为：

$$
O\left(
s\sum_{m=1}^{M}r(d_{\mathrm{in},m}+d_{\mathrm{out},m})
\right)
=
O(sd_{\mathrm{LoRA}}),
$$

其中：

$$
d_{\mathrm{LoRA}}
=
\sum_m r(d_{\mathrm{in},m}+d_{\mathrm{out},m}).
$$

精确 norm $q_{i,m}$ 的低秩计算引入额外 Gram operations，量级为：

$$
O(rd_{\mathrm{LoRA}}).
$$

因此 per-client per-round overhead 为：

$$
O((s+r)d_{\mathrm{LoRA}}),
$$

在 $s,r$ 为小常数时是 linear in trainable LoRA dimension。

### 13.3 Server-Side Computation

server 统计量估计为：

$$
O(MKs)
$$

for $\hat a_m,\hat b_m$，以及：

$$
O(MK^2s)
$$

for pairwise sketch inner products in gap-aware P2.

P1 equal-cost allocation 可通过排序所有 marginal gains 实现：

$$
O(MK\log(MK)),
$$

或用 priority queue：

$$
O(B\log M).
$$

P2 min-cost flow graph 的节点数为：

$$
V=K+M+2,
$$

边数为：

$$
E=KM+K+M.
$$

若总流量为 $B=\sum_m k_m^*$，successive shortest augmenting path 的复杂度可写为：

$$
O(BE\log V)
=
O\left(B(KM+K+M)\log(K+M)\right).
$$

由于 $K$、$M$ 是当前轮参与 clients 和 LoRA target modules 的调度规模，server-side scheduling 相比本地 LLM fine-tuning 通常较小。

---

## 14. Method Summary

本文方法与 motivation 的对应关系如下：

1. **Fig.1: Tensor-level Top-K fails**  
   Method 采用完整 A/B pair induced effective update $\Phi_{i,m}$ 作为上传原子，保证 structural consistency。

2. **Bridge: Factor norm is representation-dependent**  
   Method 将 saliency、signal、noise 和 assignment score 都定义在 effective-update space，而不是 raw factor space。

3. **Fig.2: Magnitude-only selection concentrates uploads**  
   Method 不做 independent per-client Top-K，而是先通过 P1 决定每个 module 的 quota $k_m^*$，显式考虑 shared signal、heterogeneity noise 和 diminishing returns。

4. **Need complementary client assignment**  
   Method 再通过 P2 在 client level 实现 quota，并用 leave-one-out alignment 和 gap-aware redundancy correction 选择更互补的 clients。

最终，本文方法不是简单的 diversity heuristic，也不是 magnitude Top-K 的小改动，而是从 dense effective-update preservation 出发，经由 signal-noise statistical relaxation 推导出的 server-assisted sparse upload scheduler。

