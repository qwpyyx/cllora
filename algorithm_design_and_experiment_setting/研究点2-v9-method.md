# 研究点2-v9：Method 专项文档（更新版）

> 本文档专门整理低预算 FedLoRA sparse upload 的方法设计。  
> 本版在 v9 method 的基础上进一步补充了方法推导中的关键解释，包括：原始问题 $P$ 为什么难、为什么以 dense FedAvg aggregate 作为参考、为什么使用 Frobenius norm、为什么将 $\mu_m$ 称为 shared descent signal、$P_1$ 边际收益及 equal-cost / unequal-cost 解法的证明步骤，以及 $P_2$ 从 residual 展开到 gap-aware P2-L 的完整推导。

---

## 0. 方法一句话概括

本文方法可以概括为：

> **Server-assisted signal-noise-aware sparse FedLoRA upload.**

在通信受限的 federated LoRA fine-tuning 中，每一轮有多个 clients 产生 module-wise LoRA updates，但 server 无法收集所有 updates。因此，核心系统问题不是如何压缩一个孤立 tensor，而是在有限 uplink budget 下，如何调度哪些 client-module LoRA updates 应该占用通信资源。

本文将这一问题分为两个层次：

1. **P1: module-level allocation**。服务器根据每个 LoRA module 的 shared signal strength、heterogeneity noise 和 communication cost，决定该 module 应该被多少 clients 上传，即求 $k_m^*$。
2. **P2: client-level assignment**。给定 $k_m^*$，服务器使用 gap-aware linear assignment 决定具体哪些 clients 上传哪些 modules，即求 $z_{i,m}^*$。

为了避免在调度阶段上传完整 effective update matrix，每个 client 只上传每个 module 的一个 norm scalar 和一个低维 shared normalized Rademacher bilinear sketch。服务器基于这些轻量统计量估计 $a_m,b_m,\mu_m$ 以及 P2 的 redundancy interaction。

---

## 1. 符号定义

第 $t$ 轮有 $K$ 个参与 clients。为简化符号，下文省略轮次 $t$。

| 符号 | 含义 |
|---|---|
| $K$ | 当前轮参与 clients 数量 |
| $M$ | LoRA modules 数量 |
| $i\in\{1,\dots,K\}$ | client index |
| $m\in\{1,\dots,M\}$ | LoRA module index |
| $A_m\in\mathbb{R}^{r\times d_{\mathrm{in},m}}$ | 当前全局 LoRA A factor |
| $B_m\in\mathbb{R}^{d_{\mathrm{out},m}\times r}$ | 当前全局 LoRA B factor |
| $U_{A,i,m},U_{B,i,m}$ | client $i$ 在 module $m$ 上的 local factor updates |
| $\Phi_{i,m}$ | client $i$ 在 module $m$ 上诱导的 effective update |
| $z_{i,m}\in\{0,1\}$ | 是否上传 client $i$ 的 module $m$ A/B pair |
| $k_m=\sum_i z_{i,m}$ | module $m$ 被多少 clients 上传 |
| $c_m$ | module $m$ 的上传成本 |
| $C_i$ | client $i$ 的通信预算 |
| $B=\sum_m k_m^*$ | 当前轮总上传 A/B pairs 数量 |
| $s$ | sketch dimension |
| $d_{\mathrm{LoRA}}=\sum_m r(d_{\mathrm{in},m}+d_{\mathrm{out},m})$ | 总 LoRA trainable dimension |

主文推荐采用 equal-cost A/B pair setting，即 $c_m=c$，并定义每个 client 的上传 slot 数：

$$
B_i=\left\lfloor \frac{C_i}{c}\right\rfloor.
$$

该设定适用于相同类型的 LoRA target projections，例如只选择 q/v projections，此时 A/B pair 的参数量通常相同。该设定使 P2-L 可以写成标准 bipartite b-matching / min-cost flow。若 module size 不同，可在扩展部分讨论 generalized assignment 或 unit-slot discretization。

---

## 2. Effective-update atom

client $i$ 在 module $m$ 上的完整 A/B update 为：

$$
U_{i,m}=(U_{A,i,m},U_{B,i,m}).
$$

它诱导的 effective update 是：

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
+
U_{B,i,m}A_m
+
U_{B,i,m}U_{A,i,m}.
$$

本文以完整 A/B pair induced effective update $\Phi_{i,m}$ 作为 sparse upload 的基本 atom。也就是说，$z_{i,m}=1$ 表示上传 module $m$ 的完整 A/B pair；$z_{i,m}=0$ 表示该 module 在该 client 上不上传。

这一步同时保证两件事：

1. **structural consistency**：不会单独上传 $U_A$ 或 $U_B$；
2. **representation consistency**：后续 utility 定义在 $\Phi_{i,m}$ 的 effective-update space 中，而不是 factor norm 上。

---

## 3. 原始问题 $P$: dense effective-update preservation

### 3.1 Dense aggregate 与 sparse aggregate

如果没有 sparse upload，dense full upload 的 FedAvg effective update 为：

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

注意，这里保持 dense FedAvg 的 $1/K$ scaling，而不是对被选中的 clients 重新除以 $k_m$。也就是说，未上传的 module 被视为 zero contribution。

这一点非常重要，因为本文关心的是 sparse update 对 dense FedAvg update 的保留程度。如果改成 $1/k_m$ normalization，则目标变成 selected-client average，P1 中 signal 和 noise 项的推导都会改变。

最原始的问题定义为：

$$
(P)\quad
\min_{\{z_{i,m}\}}
\left\|
\Delta_{\mathrm{dense}}-\Delta_{\mathrm{sp}}
\right\|_F^2
$$

subject to:

$$
\sum_{m=1}^{M}z_{i,m}c_m\le C_i,
\qquad i=1,\dots,K,
$$

$$
z_{i,m}\in\{0,1\}.
$$

$P$ 表示：在每个 client 通信预算下，让 sparse upload 尽量保留 dense full-upload FedLoRA 的 effective update。

### 3.2 为什么要让 sparse aggregate 接近 dense upload aggregate？

dense FedAvg 是没有通信限制时最自然的参考更新。也就是说，如果 uplink budget 足够，server 本来应该得到：

$$
\Delta_{\mathrm{dense}}
=
\frac{1}{K}\sum_{i,m}\Phi_{i,m}.
$$

这个更新定义了当前通信轮中 full-upload FedLoRA 的优化方向。通信受限时，我们不是重新定义一个完全不同的优化目标，而是希望用可上传的 sparse aggregate $\Delta_{\mathrm{sp}}$ 尽可能保留这个 full-upload 更新的方向和幅度。

这个思路与通信压缩 FL 中常见的原则一致：compressed / sparse update should approximate the uncompressed aggregate update。本文的区别在于，FedLoRA 不能直接做普通 tensor compression，因为 LoRA 的有效模型变化不是单独的 $U_A$ 或 $U_B$，而是由完整 A/B pair 共同诱导的 $\Phi_{i,m}$。因此，本文逼近的是 **effective-update aggregate**，而不是 raw factor tensor aggregate。

从优化角度看，如果全局 loss 在当前模型附近是 smooth 的，那么 sparse update 与 dense update 的差：

$$
e=\Delta_{\mathrm{sp}}-\Delta_{\mathrm{dense}}
$$

越小，使用 sparse update 造成的优化偏差越小。因此，以 dense aggregate preservation 作为原始目标是一个 task-agnostic 且 communication-aware 的 surrogate。

### 3.3 为什么原始问题 $P$ 难以解决？

$P$ 是合理的原始目标，但难以直接求解，原因包括以下四点。

**第一，$P$ 是组合优化问题。** 变量 $z_{i,m}\in\{0,1\}$，总共有 $KM$ 个二元选择变量，并且每个 client 都有 budget constraint：

$$
\sum_m z_{i,m}c_m\le C_i.
$$

这已经是 knapsack / assignment 类型的组合优化。

**第二，目标函数是 quadratic 的。** 展开目标：

$$
\left\|
\Delta_{\mathrm{dense}}-\Delta_{\mathrm{sp}}
\right\|_F^2
=
\left\|
\frac{1}{K}\sum_{i,m}(1-z_{i,m})\Phi_{i,m}
\right\|_F^2.
$$

继续展开会出现大量交叉项：

$$
\langle \Phi_{i,m},\Phi_{j,n}\rangle_F.
$$

因此它不是简单地给每个 client-module pair 一个 independent score，然后 Top-K 选择，而是 binary quadratic optimization。

**第三，dense aggregate 本身不能在 sparse setting 下直接获得。** 如果 server 想精确知道 $\Delta_{\mathrm{dense}}$，就需要接收所有 $\Phi_{i,m}$，这等价于 full upload，已经违背 sparse upload 的目的。因此 $P$ 只能作为原始理想目标，不能直接作为实际算法。

**第四，$\Phi_{i,m}$ 是高维 effective update，而不是低秩 LoRA factor 本身。** $\Phi_{i,m}\in\mathbb{R}^{d_{\mathrm{out},m}\times d_{\mathrm{in},m}}$。直接构造、上传和比较所有 $\Phi_{i,m}$ 成本过高。因此后续需要利用 LoRA 低秩结构与 bilinear sketch 进行估计。

### 3.4 为什么用 Frobenius norm？二范数是否可以？

这里需要区分两种“二范数”。

如果把矩阵展开成向量，则 vectorized $\ell_2$ norm 与 Frobenius norm 完全等价：

$$
\|X\|_F=\|\mathrm{vec}(X)\|_2.
$$

因此，写成：

$$
\left\|
\mathrm{vec}(\Delta_{\mathrm{dense}})-\mathrm{vec}(\Delta_{\mathrm{sp}})
\right\|_2^2
$$

与写成：

$$
\left\|
\Delta_{\mathrm{dense}}-\Delta_{\mathrm{sp}}
\right\|_F^2
$$

本质相同。

但如果“二范数”指的是 matrix spectral norm：

$$
\|X\|_2=\sigma_{\max}(X),
$$

则不适合本文推导。原因是 spectral norm 只关注最大奇异方向，不能衡量整个 effective update matrix 的总体误差；同时它不能自然展开成 Frobenius inner product 形式，而本文的 P1/P2 推导、signal-noise decomposition 和 random projection sketch 都依赖：

$$
\langle X,Y\rangle_F=\mathrm{Tr}(X^\top Y).
$$

因此，本文使用 Frobenius norm，本质上是在 effective-update parameter space 中使用 Euclidean distance。

---

## 4. Signal-noise statistical model

### 4.1 分解假设

对每个 module $m$，假设 client effective update 可分解为：

$$
\Phi_{i,m}=\mu_m+\xi_{i,m}.
$$

其中：

- $\mu_m$ 表示 module $m$ 上跨 clients 共享的 descent signal；
- $\xi_{i,m}$ 表示 client-specific deviation，包括 non-IID bias、本地随机噪声和局部训练偏差。

采用标准零均值噪声假设：

$$
\mathbb{E}[\xi_{i,m}]=0,
$$

$$
\mathbb{E}\|\xi_{i,m}\|_F^2=\sigma_m^2.
$$

定义：

$$
a_m=\|\mu_m\|_F^2,
\qquad
b_m=\sigma_m^2.
$$

其中 $a_m$ 是 shared signal strength，$b_m$ 是 heterogeneity noise strength。

### 4.2 为什么 $\mu_m$ 叫 shared descent signal？

$\mu_m$ 可以理解为：

$$
\mu_m=\mathbb{E}_i[\Phi_{i,m}],
$$

也就是 module $m$ 上不同 clients 共同支持的平均 effective update direction。

在 FedAvg 中，local update 来自本地优化。对于 client $i$，$\Phi_{i,m}$ 表示它在 module $m$ 上希望施加到模型的更新方向。跨 clients 平均之后，$\mu_m$ 就代表当前一轮中大多数 clients 共同支持的更新方向。在标准 smooth optimization 语境下，如果 local updates 是由 gradient descent 产生的，那么跨 client 平均 update 近似对应全局目标的下降方向。因此，我们称 $\mu_m$ 为 **shared descent signal**。

这里需要强调，“noise” 不是指 $\xi_{i,m}$ 对本地任务毫无价值。它只是相对于全局 shared update direction 的 client-specific deviation。换句话说，$\xi_{i,m}$ 可能包含某个 client 或某类数据的局部信息，但在当前目标中，我们希望优先保留对全局 FedAvg update 稳定一致的 shared signal。

这个假设不是为了精确描述所有 client updates，而是为了推导一个可解释的 allocation rule：在通信预算受限时，哪些 modules 更值得上传。

---

## 5. Statistical proxy $P_0$: shared-signal preservation

基于上述模型，本文不再只追求保留 raw dense update，而是追求保留 shared module-level signal。对 module $m$，sparse aggregate 是：

$$
\widehat{\Delta}_m
=
\frac{1}{K}\sum_{i=1}^{K}z_{i,m}\Phi_{i,m}.
$$

目标是让它接近 $\mu_m$：

$$
(P_0)\quad
\min_{\{z_{i,m}\}}
\sum_{m=1}^{M}
\left\|
\mu_m-
\frac{1}{K}\sum_{i=1}^{K}z_{i,m}\Phi_{i,m}
\right\|_F^2
$$

subject to:

$$
\sum_{m=1}^{M}z_{i,m}c_m\le C_i,
\qquad i=1,\dots,K,
$$

$$
z_{i,m}\in\{0,1\}.
$$

$P_0$ 仍然很难直接求解，因为它同时决定 module-level quota 和 client-level assignment。因此，本文将它分成两个阶段：

1. 先通过 module-level statistical relaxation 得到每个 module 的 quota $k_m^*$；
2. 再通过 client-level assignment 恢复具体的 $z_{i,m}$。

需要强调的是，$P_1$ 不是 $P_0$ 的精确等价分解，而是 $P_0$ 的 module-level statistical relaxation。它先忽略具体 client assignment，只决定每个 module 应上传多少次。随后 $P_2$ 在 per-client budget 下实现这些 quota。

---

## 6. P1: module-level signal-noise allocation

### 6.1 从 $P_0$ 到 module-level error

定义：

$$
k_m=\sum_{i=1}^{K}z_{i,m}.
$$

如果 module $m$ 被 $k_m$ 个 clients 上传，记 selected clients 为：

$$
\mathcal{S}_m=\{i:z_{i,m}=1\},
\qquad |\mathcal{S}_m|=k_m.
$$

保持 $1/K$ scaling，则 sparse aggregate 为：

$$
\widehat{\Delta}_m
=
\frac{1}{K}\sum_{i\in\mathcal{S}_m}\Phi_{i,m}.
$$

代入 signal-noise decomposition：

$$
\Phi_{i,m}=\mu_m+\xi_{i,m},
$$

得到：

$$
\widehat{\Delta}_m
=
\frac{1}{K}\sum_{i\in\mathcal{S}_m}(\mu_m+\xi_{i,m})
=
\frac{k_m}{K}\mu_m
+
\frac{1}{K}\sum_{i\in\mathcal{S}_m}\xi_{i,m}.
$$

于是 residual 为：

$$
\mu_m-\widehat{\Delta}_m
=
\left(1-\frac{k_m}{K}\right)\mu_m
-
\frac{1}{K}\sum_{i\in\mathcal{S}_m}\xi_{i,m}.
$$

对平方 Frobenius norm 取期望：

$$
\mathbb{E}
\left\|
\mu_m-\widehat{\Delta}_m
\right\|_F^2.
$$

由于 $\mathbb{E}[\xi_{i,m}]=0$，并假设不同 clients 的 noise 交叉项期望为 0，cross term 消失，得到：

$$
\mathbb{E}
\left\|
\mu_m-\widehat{\Delta}_m
\right\|_F^2
=
\left(1-\frac{k_m}{K}\right)^2\|\mu_m\|_F^2
+
\frac{k_m}{K^2}\sigma_m^2.
$$

记 $a_m=\|\mu_m\|_F^2$，$b_m=\sigma_m^2$，定义：

$$
J_m(k_m)
=
\left(1-\frac{k_m}{K}\right)^2a_m
+
\frac{k_m}{K^2}b_m.
$$

这个式子非常关键：第一项是 missing shared signal error，随 $k_m$ 增大而下降；第二项是 accumulated heterogeneity noise，随 $k_m$ 增大而上升。

因此，module-level allocation problem 为：

$$
(P_1)\quad
\min_{\{k_m\}}
\sum_{m=1}^{M}J_m(k_m)
$$

subject to:

$$
\sum_{m=1}^{M}c_mk_m\le C_{\mathrm{total}},
$$

$$
0\le k_m\le K,
\qquad
k_m\in\mathbb{Z},
$$

其中：

$$
C_{\mathrm{total}}=\sum_{i=1}^{K}C_i.
$$

### 6.2 边际收益 $\Delta_m(k)$ 的推导

定义第 $k+1$ 次上传 module $m$ 的边际收益为误差下降量：

$$
\Delta_m(k)=J_m(k)-J_m(k+1),
\qquad k=0,\dots,K-1.
$$

首先写出：

$$
J_m(k)
=
\left(1-\frac{k}{K}\right)^2a_m
+
\frac{k}{K^2}b_m,
$$

$$
J_m(k+1)
=
\left(1-\frac{k+1}{K}\right)^2a_m
+
\frac{k+1}{K^2}b_m.
$$

两式相减：

$$
\Delta_m(k)
=
a_m\left[
\left(1-\frac{k}{K}\right)^2
-
\left(1-\frac{k+1}{K}\right)^2
\right]
-
\frac{1}{K^2}b_m.
$$

因为：

$$
1-\frac{k}{K}=\frac{K-k}{K},
$$

所以：

$$
\left(1-\frac{k}{K}\right)^2
-
\left(1-\frac{k+1}{K}\right)^2
=
\frac{(K-k)^2-(K-k-1)^2}{K^2}.
$$

利用：

$$
x^2-(x-1)^2=2x-1,
$$

令 $x=K-k$，得到：

$$
(K-k)^2-(K-k-1)^2=2(K-k)-1.
$$

因此：

$$
\Delta_m(k)
=
\frac{(2(K-k)-1)a_m-b_m}{K^2}.
$$

并且：

$$
\Delta_m(k+1)-\Delta_m(k)
=
-\frac{2a_m}{K^2}.
$$

当 $a_m>0$ 时，边际收益严格递减。这说明同一个 module 上传越多 clients，继续上传一个 client 的新增收益越小。

### 6.3 Equal-cost 解法及证明

在 equal-cost setting 下，所有 A/B pair 的 cost 相同。此时预算等价于最多上传 $B$ 个 A/B pairs：

$$
\sum_m k_m\le B.
$$

P1 是：

$$
\min_{\{k_m\}}
\sum_m J_m(k_m)
$$

subject to:

$$
\sum_m k_m\le B,
\qquad
0\le k_m\le K,
\qquad
k_m\in\mathbb{Z}.
$$

等价地，最大化相对于 $k_m=0$ 的误差下降：

$$
\max_{\{k_m\}}
\sum_m [J_m(0)-J_m(k_m)].
$$

而：

$$
J_m(0)-J_m(k_m)
=
\sum_{\ell=0}^{k_m-1}\Delta_m(\ell).
$$

因此，P1 等价于从所有 candidate marginal gains：

$$
\{\Delta_m(0),\Delta_m(1),\dots,\Delta_m(K-1):m=1,\dots,M\}
$$

中选择最多 $B$ 个收益最大的增量。

这里有一个 prefix constraint：如果选择了 $\Delta_m(k)$，必须也选择 $\Delta_m(0),\dots,\Delta_m(k-1)$，因为第 $k+1$ 次上传只有在前 $k$ 次上传已经发生时才有意义。

由于边际收益满足：

$$
\Delta_m(0)\ge\Delta_m(1)\ge\cdots\ge\Delta_m(K-1),
$$

所以选择所有 marginal gains 中最大的 $B$ 个正值时，prefix constraint 自动满足。原因是，如果 $\Delta_m(k)$ 被选中，那么所有更早的 $\Delta_m(\ell)$，$\ell<k$，都不小于它，因此也一定排在它之前，不会被遗漏。

因此 equal-cost 最优解为：

1. 计算所有 $\Delta_m(k)$；
2. 选择最大的 $B$ 个正 marginal gains；
3. 对每个 module 统计被选中的 marginal gains 数量，得到 $k_m^*$。

如果预算必须用满，则选前 $B$ 个；如果预算只是上限，则只选择正收益的前 $B$ 个。

复杂度可以通过排序写成：

$$
O(MK\log(MK)).
$$

也可以用 priority queue 逐个分配 $B$ 个名额：

$$
O(B\log M).
$$

### 6.4 Unequal-cost continuous relaxation 及证明

如果不同 modules 的 cost $c_m$ 不同，整数版本类似 knapsack，直接精确求解较复杂。因此可以考虑 continuous relaxation：

$$
k_m\in[0,K].
$$

P1 relaxation 为：

$$
\min_{\{k_m\}}
\sum_m J_m(k_m)
$$

subject to:

$$
\sum_m c_mk_m\le C_{\mathrm{total}},
\qquad
0\le k_m\le K.
$$

因为：

$$
J_m(k_m)
=
\left(1-\frac{k_m}{K}\right)^2a_m
+
\frac{k_m}{K^2}b_m,
$$

其二阶导数为：

$$
J_m''(k_m)=\frac{2a_m}{K^2}\ge 0.
$$

所以 continuous relaxation 是 convex optimization。

构造 Lagrangian：

$$
\mathcal{L}
=
\sum_m J_m(k_m)
+
\lambda\left(\sum_m c_mk_m-C_{\mathrm{total}}\right),
$$

其中 $\lambda\ge0$ 是预算约束的 Lagrange multiplier。

计算一阶导数：

$$
J_m'(k_m)
=
-\frac{2a_m}{K}\left(1-\frac{k_m}{K}\right)
+
\frac{b_m}{K^2}
=
\frac{-2a_m(K-k_m)+b_m}{K^2}.
$$

KKT stationarity 条件为：

$$
J_m'(k_m)+\lambda c_m=0.
$$

代入得到：

$$
\frac{-2a_m(K-k_m)+b_m}{K^2}
+\lambda c_m=0.
$$

两边乘以 $K^2$：

$$
-2a_m(K-k_m)+b_m+\lambda c_mK^2=0.
$$

因此：

$$
2a_m(K-k_m)=b_m+\lambda c_mK^2.
$$

解得：

$$
k_m(\lambda)
=
K-
\frac{b_m+\lambda c_mK^2}{2a_m}.
$$

再考虑边界约束 $0\le k_m\le K$，得到 clipped solution：

$$
k_m(\lambda)
=
\left[
K-
\frac{b_m+\lambda c_mK^2}{2a_m}
\right]_0^K.
$$

其中 $[x]_0^K=\min(\max(x,0),K)$。$\lambda$ 由预算约束决定：

$$
\sum_m c_mk_m(\lambda)=C_{\mathrm{total}}
$$

当预算约束 active 时成立；如果 unconstrained optimum 已经不超过预算，则 $\lambda=0$。实际可通过二分搜索找到 $\lambda$。

该解具有清楚的解释：

- $a_m$ 越大，module shared signal 越强，分配越多；
- $b_m$ 越大，heterogeneity noise 越强，分配越少；
- $c_m$ 越大，通信成本越高，分配越少；
- $\lambda$ 是通信资源价格，预算越紧，$\lambda$ 越大。

主文可以将 unequal-cost 放在讨论或扩展中，核心算法采用 equal-cost setting 更简洁。

---

## 7. P2: client-level assignment

### 7.1 P2 为什么需要？

P1 只决定每个 module 应该被上传多少次，即：

$$
k_m^*.
$$

但 P1 没有决定具体由哪些 clients 上传。因此，需要 P2 来恢复 client-level assignment。

对 module $m$，定义 selected clients 为：

$$
\mathcal{S}_m=\{i:z_{i,m}=1\},
\qquad |\mathcal{S}_m|=k_m^*.
$$

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

这里使用 equal-cost slot 形式。如果使用 cost form，则第二个约束为 $\sum_m z_{i,m}c_m\le C_i$。

### 7.2 单个 module residual 的含义

先看单个 module $m$。为简化符号，记：

$$
\mathcal{S}=\mathcal{S}_m.
$$

这只是表示：当前分析的是 module $m$，它被选中的 client 集合叫 $\mathcal{S}_m$，后续简写为 $\mathcal{S}$。

如果选择集合 $\mathcal{S}$ 中的 clients 上传 module $m$，则 server 实际聚合到的 module-level sparse update 是：

$$
\frac{1}{K}\sum_{i\in\mathcal{S}}\Phi_{i,m}.
$$

注意这里仍然除以 $K$，而不是除以 $|\mathcal{S}|$，因为我们保持 dense FedAvg 的 scaling，未上传 clients 被视为 zero contribution。

因此，该选择造成的 residual 是：

$$
\mu_m-
\frac{1}{K}\sum_{i\in\mathcal{S}}\Phi_{i,m}.
$$

它表示：选择 $\mathcal{S}$ 这些 clients 后，得到的 sparse aggregate 与理想 shared signal $\mu_m$ 之间还差多少。将这个矩阵差距转化成标量误差，得到：

$$
R_m(\mathcal{S})
=
\left\|
\mu_m-
\frac{1}{K}\sum_{i\in\mathcal{S}}\Phi_{i,m}
\right\|_F^2.
$$

如果 $R_m(\mathcal{S})$ 小，说明这组 clients 的上传能很好地代表 module $m$ 的 shared signal；如果 $R_m(\mathcal{S})$ 大，说明这组 clients 的更新方向不够一致，或包含较强 client-specific deviation。

因此 P2 的目标就是：在满足 quota 和 client budgets 的条件下，选择 $\mathcal{S}_m$，让所有 modules 的 $R_m(\mathcal{S}_m)$ 尽可能小。

### 7.3 从 residual 展开到 gain

对单个 module，残差为：

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

代回去：

$$
R_m(\mathcal{S})
=
\|\mu_m\|_F^2
-
\sum_{i\in\mathcal{S}}
\left[
\frac{2}{K}\langle\mu_m,\Phi_{i,m}\rangle_F
-
\frac{1}{K^2}\|\Phi_{i,m}\|_F^2
\right]
+
\frac{2}{K^2}
\sum_{i<j,\ i,j\in\mathcal{S}}
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F.
$$

定义 individual score：

$$
s_{i,m}
=
\frac{2}{K}\langle\mu_m,\Phi_{i,m}\rangle_F
-
\frac{1}{K^2}\|\Phi_{i,m}\|_F^2,
$$

定义 interaction term：

$$
I_m(\mathcal{S})
=
\frac{2}{K^2}
\sum_{i<j,\ i,j\in\mathcal{S}}
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F.
$$

则：

$$
R_m(\mathcal{S})
=
\|\mu_m\|_F^2
-
\sum_{i\in\mathcal{S}}s_{i,m}
+
I_m(\mathcal{S}).
$$

由于 $\|\mu_m\|_F^2$ 与选择 $\mathcal{S}$ 无关，minimizing residual 等价于 maximizing gain：

$$
F_m(\mathcal{S})
=
\sum_{i\in\mathcal{S}}s_{i,m}-I_m(\mathcal{S}).
$$

这个展开说明：P2 的真实收益由两部分构成：

1. 每个 client-module 的 individual gain $s_{i,m}$；
2. selected clients 之间的 pairwise interaction penalty $I_m(\mathcal{S})$。

因此，完整 P2 不是普通 Top-K，而是 quadratic assignment。

### 7.4 Plain P2-L: linearized surrogate

如果忽略 interaction term $I_m(\mathcal{S})$，则得到 linear surrogate：

$$
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

这就是 plain P2-L。它不是随便定义的 heuristic score，而是完整 residual 展开后的一阶 linear term。

但是 plain P2-L 仍然忽略了 selected clients 之间的 redundancy，尤其是当多个 selected clients 在同一 module 上高度同向时，positive interaction 会降低总收益。

### 7.5 Gap-aware P2-L

完整 P2 gain 是：

$$
F_m(\mathcal{S})
=
\sum_{i\in\mathcal{S}}s_{i,m}
-
I_m(\mathcal{S}).
$$

其中：

$$
I_m(\mathcal{S})
=
\frac{2}{K^2}
\sum_{i<j,\ i,j\in\mathcal{S}}
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F.
$$

如果两个 selected clients 的 updates 同向：

$$
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F>0,
$$

则它们会产生 positive interaction penalty，说明二者存在一定冗余。为近似弥补 plain P2-L 与完整 P2 的差距，定义 positive interaction exposure：

$$
d_{i,m}^+
=
\sum_{j\ne i}
\max\left(
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F,
0
\right).
$$

对于任意 selected set $\mathcal{S}$，其 positive interaction 部分为：

$$
I_m^+(\mathcal{S})
=
\frac{2}{K^2}
\sum_{i<j,\ i,j\in\mathcal{S}}
\max\left(
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F,
0
\right).
$$

而：

$$
I_m^+(\mathcal{S})
\le
\frac{1}{K^2}
\sum_{i\in\mathcal{S}}d_{i,m}^+.
$$

原因是 $\sum_{i\in\mathcal{S}}d_{i,m}^+$ 至少把 selected set 内部每一条 positive pairwise interaction 计数两次。因此可构造 separable correction：

$$
\tilde{s}_{i,m}
=
s_{i,m}
-
\frac{\eta}{K^2}d_{i,m}^+,
$$

其中 $\eta\in[0,1]$ 是 correction strength。$\eta=0$ 对应 plain P2-L；$\eta=1$ 对应使用 positive interaction upper-bound correction。

最终的 gap-aware P2-L 为：

$$
(P_{2\text{-L}})\quad
\max_{\{z_{i,m}\}}
\sum_{i=1}^{K}\sum_{m=1}^{M}z_{i,m}\tilde{s}_{i,m}
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

---

## 8. Leave-one-out shared direction

P2 score 中需要 $\langle\mu_m,\Phi_{i,m}\rangle_F$。实际估计时，如果直接用所有 clients 的平均值：

$$
\hat\mu_m=\frac{1}{K}\sum_{j=1}^{K}\Phi_{j,m},
$$

则 $\langle\hat\mu_m,\Phi_{i,m}\rangle_F$ 包含 self-inner-product $\|\Phi_{i,m}\|_F^2$，会把 P2 alignment 部分拉回 magnitude-based selection。

因此，本文使用 leave-one-out shared direction：

$$
\hat\mu_{-i,m}
=
\frac{1}{K-1}\sum_{j\ne i}\Phi_{j,m}.
$$

对应 alignment 为：

$$
\langle\mu_m,\Phi_{i,m}\rangle_F
\approx
\langle\hat\mu_{-i,m},\Phi_{i,m}\rangle_F.
$$

这样可以更准确地度量 client $i$ 的 update 是否与其他 clients 的共享方向一致，而不是奖励它自己的 norm。

---

## 9. Shared normalized Rademacher bilinear random projection sketch

P1 和 P2 需要估计：

$$
\|\Phi_{i,m}\|_F^2,
\qquad
\langle\Phi_{i,m},\Phi_{j,m}\rangle_F,
\qquad
\langle\mu_m,\Phi_{i,m}\rangle_F.
$$

如果直接上传完整 $\Phi_{i,m}\in\mathbb{R}^{d_{\mathrm{out},m}\times d_{\mathrm{in},m}}$，通信开销过高。因此，本文采用 shared normalized Rademacher bilinear random projection sketch。

### 9.1 Sketch 定义

对每个 module $m$，server 和 clients 共享随机种子，生成 $s$ 组随机向量：

$$
u_{\ell,m}\in\mathbb{R}^{d_{\mathrm{out},m}},
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

这里用 $\nu$ 表示 left projection vector，以避免和 LoRA factor $U$ 混淆。

client $i$ 对 module $m$ 计算：

$$
r_{i,m}[\ell]
=
\sqrt{d_{\mathrm{out},m}d_{\mathrm{in},m}}
\cdot
\nu_{\ell,m}^{\top}\Phi_{i,m}v_{\ell,m}.
$$

于是：

$$
r_{i,m}\in\mathbb{R}^{s}.
$$

### 9.2 无偏内积估计

对任意两个 matrices $X,Y$，定义：

$$
r_X[\ell]
=
\sqrt{d_{\mathrm{out}}d_{\mathrm{in}}}\cdot
\nu_{\ell}^{\top}Xv_{\ell},
$$

$$
r_Y[\ell]
=
\sqrt{d_{\mathrm{out}}d_{\mathrm{in}}}\cdot
\nu_{\ell}^{\top}Yv_{\ell}.
$$

由于 normalized Rademacher vectors 满足：

$$
\mathbb{E}[\nu\nu^\top]=\frac{1}{d_{\mathrm{out}}}I,
\qquad
\mathbb{E}[vv^\top]=\frac{1}{d_{\mathrm{in}}}I,
$$

因此：

$$
\mathbb{E}[r_X[\ell]r_Y[\ell]]
=
\langle X,Y\rangle_F.
$$

于是可以用：

$$
\widehat{\langle X,Y\rangle_F}
=
\frac{1}{s}\langle r_X,r_Y\rangle.
$$

该 sketch 需要满足以下性质：zero-mean、isotropic、$\nu$ 和 $v$ 相互独立、不同 sketch dimensions 相互独立或近似独立、同一轮同一 module 的所有 clients 使用相同随机投影，并具有 concentration，使误差随 $s$ 增加大致按 $O(1/\sqrt{s})$ 下降。

### 9.3 低秩计算

client 不需要显式构造 $\Phi_{i,m}$。因为：

$$
\Phi_{i,m}
=
B_mU_{A,i,m}+U_{B,i,m}A_m+U_{B,i,m}U_{A,i,m}.
$$

对任意 sketch 向量 $\nu,v$：

$$
\nu^\top\Phi_{i,m}v
=
(\nu^\top B_m)(U_{A,i,m}v)
+
(\nu^\top U_{B,i,m})(A_mv)
+
(\nu^\top U_{B,i,m})(U_{A,i,m}v).
$$

这只需要低秩乘法，单个 module 的 sketch 计算复杂度约为：

$$
O\left(sr(d_{\mathrm{in},m}+d_{\mathrm{out},m})\right).
$$

---

## 10. 统计量估计

每个 client 对每个 module 上传：

1. 一个 norm scalar：

$$
q_{i,m}=\|\Phi_{i,m}\|_F^2;
$$

2. 一个 sketch vector：

$$
r_{i,m}\in\mathbb{R}^{s}.
$$

也就是说，调度阶段上传的是：

$$
\{q_{i,m},r_{i,m}\}_{m=1}^{M}.
$$

它不是矩阵，也不是完整 LoRA A/B updates。

### 10.1 Shared signal strength

server 用 pairwise inner product 的无偏估计来估计 $a_m=\|\mu_m\|_F^2$：

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

实际中使用截断：

$$
\hat a_m\leftarrow\max(\hat a_m,0).
$$

### 10.2 Total magnitude and noise

估计 total magnitude：

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

### 10.3 Leave-one-out shared direction in sketch space

定义 sketch-space leave-one-out shared direction：

$$
\hat r_{\mu,-i,m}
=
\frac{1}{K-1}\sum_{j\ne i}r_{j,m}.
$$

则：

$$
\widehat{\langle\mu_m,\Phi_{i,m}\rangle_F}
=
\frac{1}{s}\langle\hat r_{\mu,-i,m},r_{i,m}\rangle.
$$

### 10.4 Gap-aware redundancy

估计 positive interaction exposure：

$$
\hat d_{i,m}^+
=
\sum_{j\ne i}
\max\left(
\frac{1}{s}\langle r_{i,m},r_{j,m}\rangle,
0
\right).
$$

### 10.5 Final sketch-space P2 score

最终 gap-aware P2-L score 为：

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

这个 score 同时包含：

1. 与其他 clients shared direction 的 alignment；
2. 自身 update norm 的 residual penalty；
3. 与其他 clients 的 positive redundancy correction。

---

## 11. Min-cost flow formulation for P2-L

在 equal-cost setting 下，P2-L 可构造成 min-cost flow：

- source 到每个 client 节点，capacity 为 $B_i$；
- client $i$ 到 module $m$，capacity 为 1，cost 为 $-\tilde{s}_{i,m}$；
- module $m$ 到 sink，demand/capacity 为 $k_m^*$。

求最小费用最大流等价于最大化：

$$
\sum_{i,m}z_{i,m}\tilde{s}_{i,m}.
$$

若由于 pruning 或异构 budget 导致 quota 不可行，可以使用 dummy client 或 slack edges 允许：

$$
\sum_i z_{i,m}\le k_m^*.
$$

并对未满足 quota 加 penalty。该部分属于实现细节，不作为核心贡献展开。

---

## 12. 完整算法流程

**Algorithm: Signal-Noise-Aware Sparse FedLoRA Upload**

输入：当前全局 LoRA factors $\{A_m,B_m\}_{m=1}^M$，参与 clients $\mathcal{K}$，client budgets $\{B_i\}$，sketch dimension $s$，correction strength $\eta$。

1. **Local training**：每个 client $i$ 在本地数据上训练，得到 $\{U_{A,i,m},U_{B,i,m}\}_{m=1}^M$。
2. **Effective-update statistics**：每个 client 对每个 module 计算 $q_{i,m}=\|\Phi_{i,m}\|_F^2$ 和 sketch $r_{i,m}$。
3. **Lightweight statistics upload**：client 上传 $\{q_{i,m},r_{i,m}\}_{m=1}^M$ 到 server。
4. **Signal-noise estimation**：server 估计 $\hat a_m,\hat b_m$。
5. **P1 allocation**：server 根据 marginal gain $\Delta_m(k)$ 求解 module quota $k_m^*$。
6. **P2 score computation**：server 计算 leave-one-out alignment、positive redundancy exposure 和 $\tilde{s}_{i,m}$。
7. **Gap-aware assignment**：server 用 min-cost flow 求解 $z_{i,m}^*$。
8. **Assignment broadcast**：server 向每个 client 下发应上传的 module index set $\mathcal{S}_i=\{m:z_{i,m}^*=1\}$。
9. **Sparse A/B upload**：client 只上传被选中的 A/B pairs。
10. **Aggregation**：server 按 $1/K$ scaling 聚合 selected A/B updates，未上传 entries 视为 zero contribution。

---

## 13. Online scheduling 与 pipelined scheduling

严格 online 版本流程为：

1. client 本地训练；
2. client 上传轻量 statistics；
3. server 求 P1/P2；
4. server 下发 assignment；
5. client 上传 selected A/B pairs；
6. server 聚合。

这会增加一个轻量 scheduling phase。

为了减少额外 RTT，可以采用 pipelined scheduling：server 使用历史 EMA statistics 提前估计当前轮的 quota 和 assignment，当前轮上传后再更新 statistics。主文可以先描述 online version，因为它最清楚；系统讨论中说明 pipelined version 可以减少控制阶段延迟。

---

## 14. 通信开销

每个 client 每个 module 上传：

$$
q_{i,m}\in\mathbb{R},
\qquad
r_{i,m}\in\mathbb{R}^s.
$$

因此，每个 client 的调度统计上传量为：

$$
O(M(s+1)).
$$

所有 clients 合计为：

$$
O(KM(s+1)).
$$

若使用 fp16，则字节数约为：

$$
2KM(s+1)\ \mathrm{bytes}.
$$

由于 $s$ 是小常数，例如 $s=16$，这部分远小于 LoRA A/B pair upload payload：

$$
O\left(
\sum_{i,m}z_{i,m}r(d_{\mathrm{in},m}+d_{\mathrm{out},m})
\right).
$$

server 下发的 assignment 只是 module indices：

$$
\mathcal{S}_i=\{m:z_{i,m}^*=1\},
$$

下行开销约为 $O(B\log M)$ bits，可忽略。

---

## 15. 计算复杂度

### 15.1 Client-side overhead

每个 client 对每个 module 计算 sketch 和 norm。利用低秩结构，sketch 计算复杂度为：

$$
O\left(sr(d_{\mathrm{in},m}+d_{\mathrm{out},m})\right).
$$

所有 modules 合计：

$$
O\left(s\sum_{m=1}^{M}r(d_{\mathrm{in},m}+d_{\mathrm{out},m})\right)
=
O(sd_{\mathrm{LoRA}}).
$$

精确 norm $q_{i,m}=\|\Phi_{i,m}\|_F^2$ 的计算还会引入低秩 Gram operations，复杂度可写为：

$$
O(rd_{\mathrm{LoRA}})
$$

量级。由于 LoRA rank $r$ 和 sketch dimension $s$ 都是小常数，单个 client 每轮额外计算复杂度为：

$$
O((s+r)d_{\mathrm{LoRA}})=O(d_{\mathrm{LoRA}}).
$$

这就是与之前 ICML 论文中 $O(d)$ 风格相对应的结论，只是这里的 $d$ 应明确为 LoRA trainable dimension。

### 15.2 Server-side statistics

估计 $\hat a_m,\hat b_m$ 需要：

$$
O(MKs).
$$

计算 gap-aware redundancy $\hat d_{i,m}^+$ 需要 pairwise sketch inner products：

$$
O(MK^2s).
$$

### 15.3 P1 allocation

直接排序所有 marginal gains：

$$
O(MK\log(MK)).
$$

或用 priority queue：

$$
O(B\log M).
$$

### 15.4 P2 min-cost flow

P2 graph 的节点数为：

$$
V=K+M+2.
$$

边数为：

$$
E=KM+K+M.
$$

总流量为：

$$
B=\sum_m k_m^*.
$$

使用 successive shortest augmenting path，复杂度可写为：

$$
O(BE\log V)
=
O\left(B(KM+K+M)\log(K+M)\right).
$$

### 15.5 Overall complexity

完整每轮额外计算为：

$$
O\left(
Ks d_{\mathrm{LoRA}}
+
Kr d_{\mathrm{LoRA}}
+
MK^2s
+
MK\log(MK)
+
B(KM+K+M)\log(K+M)
\right).
$$

从单个 client 每轮角度看：

$$
O((s+r)d_{\mathrm{LoRA}})=O(d_{\mathrm{LoRA}}).
$$

从系统每轮角度看，如果 $K,s,r$ 视为固定小常数，server 侧调度又只发生在 $K\times M$ 的小规模图上，则主导项仍然是线性的 LoRA 统计计算：

$$
O(d_{\mathrm{LoRA}}).
$$

因此，论文中可以写：

> The per-client per-round overhead is $O(d_{\mathrm{LoRA}})$, i.e., linear in the trainable LoRA parameter dimension. The server-side scheduling operates on a small client-module graph and is negligible compared with local LLM fine-tuning.

---

## 16. 方法总结

本文方法的完整逻辑是：

1. **从 $P$ 出发**：sparse upload 应保留 dense effective-update FedAvg；
2. **解释 $P$ 的困难**：它是高维 effective-update space 中的 binary quadratic constrained selection，且 dense aggregate 不能直接获得；
3. **引入 signal-noise model**：dense update 中包含 shared signal 和 heterogeneity deviation；
4. **得到 $P_0$**：目标转成 shared-signal preservation；
5. **推导 $P_1$**：先决定每个 module 的上传 quota $k_m^*$，边际收益为 $((2(K-k)-1)a_m-b_m)/K^2$；
6. **给出 $P_1$ 解法**：equal-cost 下选择 top positive marginal gains，unequal-cost relaxation 下得到 water-filling-like solution；
7. **推导 $P_2$**：从单 module residual 展开得到 individual score 和 pairwise interaction；
8. **使用 gap-aware P2-L**：从完整 quadratic P2 的 interaction gap 推导 positive redundancy correction；
9. **使用 Rademacher bilinear sketch**：低成本估计 P1/P2 所需 effective-update statistics；
10. **复杂度线性**：per-client per-round overhead 为 $O(d_{\mathrm{LoRA}})$。

最终方法不是 naive Top-K，而是一个从 dense preservation、signal-noise decomposition、diminishing returns 和 assignment gap 推导出来的 server-assisted sparse upload algorithm。
