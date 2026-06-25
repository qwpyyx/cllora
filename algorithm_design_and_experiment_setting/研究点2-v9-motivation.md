# 研究点2-v9：Motivation 专项文档

> 本文档专门整理低预算 FedLoRA sparse upload 的 motivation。  
> 主要参考当前最终版主线，也就是 v9_revised 中确认的逻辑：  
> **从 tensor-level failure 出发，经过 A/B pair structural consistency、factor-norm representation inconsistency、effective-norm magnitude-only failure，最终收敛到 signal-noise-aware sparse upload。**  
> 本文档只讨论“为什么需要这个方法”，不展开完整算法求解细节。完整方法见 `研究点2-v9-method.md`。

---

## 0. Motivation 的一句话主线

低预算 FedLoRA sparse upload 不是普通的 tensor compression。LoRA 的有效更新由 A/B factors 共同诱导，LoRA factor 表示本身又具有非唯一性；进一步，在联邦异质数据下，client local update 同时包含跨客户端共享的下降信号和客户端特异的异质性噪声。因此，合理的 sparse upload 不能简单选择最大的 tensor、最大的 A/B factor norm，甚至不能只选择最大的 effective update norm，而应该在 effective-update space 中保留 shared descent signal，并抑制 heterogeneity noise 和重复上传带来的边际收益衰减。

英文可以概括为：

> Low-budget sparse FedLoRA should preserve shared effective descent signals under client heterogeneity, rather than selecting large factor-space or magnitude-only effective updates.

这句话对应四层含义：

1. **structural consistency**：上传单元必须对应完整的 LoRA effective update；
2. **representation consistency**：重要性度量不能依赖非唯一的 LoRA factor 表示；
3. **signal-noise awareness**：重要性不能只看 raw magnitude，而要区分 shared signal 和 heterogeneity noise；
4. **allocation awareness**：上传选择不能是独立 Top-K，而要考虑同一 module 被多个 clients 上传时的 diminishing returns。

---

## 1. 背景：为什么 FedLoRA sparse upload 不是普通压缩？

Federated LoRA fine-tuning 的基本目标是让多个 clients 在本地数据上训练 LoRA adapters，并只上传低秩 LoRA updates 到服务器聚合。相比 full-model fine-tuning，LoRA 已经显著降低了可训练参数量；但在 cross-device 或带宽受限场景中，完整上传所有 LoRA modules 仍然可能带来较高 uplink cost。于是，一个自然问题是：

> 在每轮通信预算有限时，每个 client 应该上传哪些 LoRA updates？

表面上，这像是一个普通的 sparse compression 或 Top-K selection 问题。直觉上可以按 tensor norm、A/B pair norm 或 effective update norm 排序，然后上传最大的部分。但 FedLoRA 与普通模型压缩有本质差异，因为 LoRA update 并不是一个普通独立 tensor，而是一个由两个低秩 factors 共同定义的 effective weight update。

第 $t$ 轮，第 $m$ 个 LoRA module 的全局 factors 为：

$$
A_m^t \in \mathbb{R}^{r\times d_{\mathrm{in},m}},
\qquad
B_m^t \in \mathbb{R}^{d_{\mathrm{out},m}\times r}.
$$

client $i$ 本地训练后得到 factor-space update：

$$
U_{i,m}^t=(U_{A,i,m}^t,U_{B,i,m}^t).
$$

如果完整应用该 A/B pair，它诱导的 effective update 是：

$$
\Phi_{i,m}^t
=
(B_m^t+U_{B,i,m}^t)(A_m^t+U_{A,i,m}^t)-B_m^tA_m^t.
$$

展开后：

$$
\Phi_{i,m}^t
=
B_m^tU_{A,i,m}^t
+
U_{B,i,m}^tA_m^t
+
U_{B,i,m}^tU_{A,i,m}^t.
$$

这说明单独上传 $U_A$ 或 $U_B$ 并不能完整表示 client 在该 module 上的 effective update。LoRA sparse upload 的基本对象不是单个 tensor，而应该是完整 A/B pair induced effective update $\Phi_{i,m}^t$。因此，直接套用普通 tensor-level Top-K 会破坏 LoRA 的结构。

---

## 2. 现象一：Tensor-level Top-K 灾难性失败

### 2.1 实验现象

在 GSM8K 上，tensor-level Top-K 的性能非常差，而 A/B pair atomic upload 能显著恢复性能。已有实验结果如下：

| 上传粒度 | seed 42 | seed 45 | seed 28 | 平均 |
|---|---:|---:|---:|---:|
| tensor-level Top-K | 1.8954 | 1.5921 | 1.8196 | 1.7690 |
| A/B pair atomic | 28.3548 | 28.1274 | 27.5208 | 28.0010 |
| q/v block atomic | 28.2032 | 27.6725 | 28.3548 | 28.0768 |
| Dense full upload | 34.4958 | 34.7991 | 34.7991 | 34.6980 |

这个现象非常关键。tensor-level Top-K 并不是“略差”，而是几乎失效；而只要把上传单元改成 A/B pair，性能就大幅恢复。这说明低预算 FedLoRA 的第一层问题不是简单的压缩率，而是上传单元是否保持 LoRA effective update 的结构闭合性。

在 Dolly 上也观察到类似趋势。tensor-level Top-K 的生成长度接近生成上限，说明模型输出行为异常；A/B pair Top-K 后，ROUGE 和生成长度恢复到较正常范围。

| 方法 | Exact Match | ROUGE-1 | ROUGE-L | Gen Len |
|---|---:|---:|---:|---:|
| tensor-level Top-K | 0.0000 | 28.0820 | 20.3345 | 127.79 |
| A/B pair Top-K | 3.5333 | 44.3689 | 35.2605 | 64.52 |
| A/B pair + group mask G=4 | 2.9333 | 44.3953 | 35.3654 | 63.63 |
| A/B pair + coverage penalty $\beta=0.05$ | 3.2667 | 44.2471 | 35.0878 | 65.79 |
| A/B pair + coverage penalty $\beta=0.1$ | 3.0667 | 44.3184 | 35.1365 | 67.57 |

Dolly 是开放式 instruction generation，Exact Match 本身并不是最核心指标。这里更重要的是生成行为：tensor-level Top-K 使生成长度接近上限，而 A/B pair 后生成长度恢复正常。这进一步支持 tensor-level upload 破坏了 LoRA update 的有效结构。

### 2.2 机制解释：A/B split error

若只上传 $U_{A,i,m}$ 而不上传 $U_{B,i,m}$，服务器端有效更新变成：

$$
\widetilde{\Phi}^{A}_{i,m}=B_m^tU_{A,i,m}^t.
$$

与完整 effective update 的误差为：

$$
\Phi_{i,m}^t-\widetilde{\Phi}^{A}_{i,m}
=
U_{B,i,m}^tA_m^t+U_{B,i,m}^tU_{A,i,m}^t.
$$

若只上传 $U_{B,i,m}$ 而不上传 $U_{A,i,m}$，服务器端有效更新变成：

$$
\widetilde{\Phi}^{B}_{i,m}=U_{B,i,m}^tA_m^t.
$$

对应误差为：

$$
\Phi_{i,m}^t-\widetilde{\Phi}^{B}_{i,m}
=
B_m^tU_{A,i,m}^t+U_{B,i,m}^tU_{A,i,m}^t.
$$

这些误差并不是普通 sparsification residual，而是由破坏 LoRA multiplicative structure 造成的 structural error。因此，A/B pair atomic upload 是低预算 FedLoRA sparse upload 的必要条件。

### 2.3 第一层 motivation

由此得到第一层结论：

> Sparse FedLoRA cannot treat LoRA tensors as independent compression units. The atomic upload unit should be an A/B pair that induces a complete effective update.

也就是：

> **低预算 FedLoRA 首先需要 structural consistency。**

---

## 3. 现象二：A/B pair 解决了上传单元，但没有解决 saliency 定义

### 3.1 A/B pair 不是终点

A/B pair atomic upload 解决了“哪些 tensors 应该一起上传”的问题，但并没有解决“哪个 A/B pair 更重要”的问题。最直接的做法是对每个 A/B pair 定义 factor-space norm：

$$
\|U_{A,i,m}\|_F^2+\|U_{B,i,m}\|_F^2,
$$

然后选择最大的 A/B pairs。这就是 factor-norm A/B pair Top-K。

这个方法虽然比 tensor-level Top-K 好很多，但它仍然有一个根本问题：LoRA factorization 本身不是唯一的。同一个 low-rank effective matrix 可以由不同的 A/B factors 表示，而 factor norm 会随表示变化。

### 3.2 LoRA factor 表示非唯一

对任意可逆矩阵 $S\in\mathbb{R}^{r\times r}$，有：

$$
BA=(BS)(S^{-1}A).
$$

这说明 $B,A$ 的具体 factor 表示不是唯一的。进一步，对 local update induced effective update 而言，不同 factor parameterization 可能诱导相同或非常接近的 effective update，但 factor norm 可以发生显著变化。

因此，factor-space saliency：

$$
\|U_A\|_F^2+\|U_B\|_F^2
$$

不是 effective update 的内在属性。它依赖具体 LoRA factor 表示。若一个上传策略的 saliency 会因为等价 factor reparameterization 而改变，那么该策略缺少 representation consistency。

### 3.3 实验诊断：factor selection 与 effective selection 差异明显

在 pair saliency diagnostics 中，factor-norm selection 和 effective-norm selection 的 Top set overlap 较低。典型 summary 结果包括：

| budget | mean Spearman factor vs effective | mean Top-set Jaccard | effective mass ratio factor/effective |
|---:|---:|---:|---:|
| 440 | 0.8067 | 0.1333 | 0.7214 |
| 880 | 0.8036 | 0.2080 | 0.7577 |
| 1320 | 0.7492 | 0.2279 | 0.7366 |
| 1760 | 0.6761 | 0.2605 | 0.7660 |
| 2200 | 0.6228 | 0.2879 | 0.7830 |
| 3080 | 0.5646 | 0.3394 | 0.7912 |

这些结果说明两点：

1. factor-norm ranking 与 effective-norm ranking 并不一致；
2. factor-norm selected pairs 只能覆盖 effective-norm selected mass 的一部分。

更重要的是，reparameterization diagnostics 显示，改变 factor scaling 会明显改变 factor-norm selection，但 effective update 本身保持一致。这进一步证明 factor-norm saliency 不是 representation-consistent 的。

### 3.4 第二层 motivation

由此得到第二层结论：

> A/B pair atomicity is necessary but not sufficient. Sparse FedLoRA also needs a saliency measure defined in the effective-update space rather than in the non-unique factor space.

也就是：

> **低预算 FedLoRA 需要 representation consistency。**

---

## 4. 现象三：Effective-norm Top-K 仍然不充分

### 4.1 Effective norm 是自然 baseline，但不是最终答案

既然 factor norm 不具备 representation consistency，那么一个自然改进是直接在 effective-update space 中定义 saliency：

$$
\|\Phi_{i,m}\|_F^2.
$$

这比 factor-norm 更合理，因为 $\Phi_{i,m}$ 是 A/B pair 对模型权重实际造成的 effective update，具有明确的表示意义。

但是，实验和理论都表明：仅按 effective norm 做 Top-K 仍然不充分。原因是 effective norm 只衡量 magnitude，不衡量这个 update 是否代表跨 clients 共享的下降方向，也不考虑多个 clients 重复上传同一 module 时的边际收益衰减。

### 4.2 Effective-norm selection 的 concentration 问题

在多 client 联邦场景中，如果每个 client 独立选择自己的 top effective-norm modules，选择结果容易高度集中在少数 modules 上。已有 selection overlap diagnostics 显示，effective-norm selection 的跨 client Jaccard overlap 很高，尤其在 GSM8K 后续轮次中甚至出现所有 clients 选择完全相同 modules 的情况：

| round | pairwise Jaccard | union selected modules | fully shared modules |
|---:|---:|---:|---:|
| 1 | 0.8561 | 26 | 14 |
| 2 | 0.9656 | 22 | 18 |
| 3 | 1.0000 | 20 | 20 |
| 4 | 1.0000 | 20 | 20 |
| 5 | 1.0000 | 20 | 20 |

这种现象说明 effective-norm Top-K 容易把所有 clients 的预算集中到相同 modules 上。适度集中不是问题，因为 shared signal 本来可能集中在少数 modules；但过度集中说明算法没有显式建模边际收益。当一个 module 已经被多个 clients 上传后，继续上传同一个 module 的收益不应与第一次上传相同。

### 4.3 Magnitude-only utility 的理论问题

在联邦异质环境下，client update 不是纯粹的全局下降方向。对每个 module $m$，可以写成：

$$
\Phi_{i,m}=\mu_m+\xi_{i,m},
$$

其中 $\mu_m$ 是跨 clients 共享的 descent signal，$\xi_{i,m}$ 是 client-specific deviation。设：

$$
a_m=\|\mu_m\|_F^2,
\qquad
b_m=\mathbb{E}\|\xi_{i,m}\|_F^2.
$$

那么 effective norm 的期望是：

$$
\mathbb{E}\|\Phi_{i,m}\|_F^2=a_m+b_m.
$$

这意味着 effective-norm Top-K 会把 shared signal 和 heterogeneity noise 都当作正贡献。可是对全局 FedLoRA 聚合来说，真正有价值的是 shared signal；client-specific noise 不仅不应被奖励，还可能造成偏移或不稳定。

当 module $m$ 被 $k_m$ 个 clients 上传，并保持 dense FedAvg 的 $1/K$ scaling 时，该 module 的 expected preservation error 可以写成：

$$
J_m(k_m)
=
\left(1-\frac{k_m}{K}\right)^2a_m
+
\frac{k_m}{K^2}b_m.
$$

它的边际收益为：

$$
\Delta_m(k)
=J_m(k)-J_m(k+1)
=
\frac{(2(K-k)-1)a_m-b_m}{K^2}.
$$

这个公式揭示了 effective-norm Top-K 的核心问题：

1. effective norm 使用 $a_m+b_m$，把 noise 当作正收益；
2. 真正的边际收益中 $b_m$ 是负项；
3. 边际收益随 $k$ 增大而下降，即存在 diminishing returns。

因此，representation-consistent magnitude 仍然不是 optimization-aware utility。

### 4.4 第三层 motivation

由此得到第三层结论：

> Effective-norm Top-K is representation-consistent but not optimization-aware. It ignores the signal-noise decomposition of federated updates and the diminishing returns of repeatedly uploading the same module.

也就是：

> **低预算 FedLoRA 需要 signal-noise-aware 和 allocation-aware utility。**

---

## 5. 为什么不是简单 diversity 或 coverage？

前期也探索过 coverage/diversity 方向，例如 group mask 或 coverage penalty。这些方法确实能改变跨 clients 的选择分布，但没有稳定带来性能提升。原因在于，FedLoRA sparse upload 的核心不是“让 clients 选择得越不一样越好”，而是：

> 在共享信号强的 module 上允许适度重复，在异质噪声高或边际收益衰减明显的 module 上避免无效重复。

换言之，重复上传本身不是坏事。如果某个 module 的 shared signal $a_m$ 很强、noise $b_m$ 很低，那么让多个 clients 上传它是合理的。真正的问题是 magnitude-only Top-K 无法判断这种重复是否仍有正边际收益。

因此，本文不把 diversity 或 coverage 作为核心 objective，而是从 signal-noise marginal gain 推导上传 allocation。这比简单惩罚 overlap 更有理论依据。

---

## 6. 统一理论解释：Sparse upload 是 shared-signal allocation

前面三个现象可以统一解释为：

> Low-budget FedLoRA sparse upload should be formulated as preserving shared descent signal under client heterogeneity.

具体来说：

1. tensor-level Top-K 失败，是因为它没有 preservation 的正确 atom；
2. factor-norm A/B pair 不稳定，是因为它没有在 effective-update space 中定义 preservation；
3. effective-norm Top-K 不充分，是因为它保留 raw magnitude，而不是 shared signal；
4. selection concentration 说明独立 Top-K 缺少全局 allocation 和 diminishing-return 控制。

因此，本文最终要解决的问题不是“如何找最大的 LoRA update”，而是：

> 在每轮通信预算下，如何把有限上传机会分配给那些具有强 shared signal、低 heterogeneity noise，并且仍有正边际收益的 LoRA modules，再把这些 module-level quota 分配给最能代表共享方向且冗余较低的 clients。

这自然导向两层决策：

1. **module-level allocation**：每个 module 应该被多少 clients 上传？
2. **client-level assignment**：给定每个 module 的 quota，具体由哪些 clients 上传？

---

## 7. 从 motivation 到方法的自然过渡

基于上述分析，最终方法应满足四个要求。

### 7.1 Requirement 1: Structural consistency

上传单元必须是完整 A/B pair，而不是单个 LoRA tensor。每个上传决策 $z_{i,m}$ 对应 client $i$ 是否上传 module $m$ 的完整 A/B update，从而对应完整 effective update $\Phi_{i,m}$。

### 7.2 Requirement 2: Representation consistency

utility 应定义在 effective-update space 中，而不是 factor space 中。也就是说，比较两个 candidate updates 时，应比较它们诱导的 $\Phi_{i,m}$，而不是单独比较 $U_A$ 或 $U_B$ 的 factor norm。

### 7.3 Requirement 3: Signal-noise awareness

utility 应区分 shared descent signal 和 client-specific heterogeneity noise。模块级重要性不应是 $a_m+b_m$，而应根据边际收益：

$$
\Delta_m(k)=\frac{(2(K-k)-1)a_m-b_m}{K^2}.
$$

这意味着 high-signal、low-noise 的 modules 更值得上传。

### 7.4 Requirement 4: Assignment-gap awareness

给定 module quota 后，选择具体 clients 时不能只看 individual alignment，还应考虑不同 clients 在同一 module 上的 redundancy interaction。因此，P2 不应停留在 naive linear score，而应从完整 quadratic assignment 推导 gap-aware P2-L。

---

## 8. 最终 motivation 结论

本文的 motivation 可以收敛为如下正式表述：

> Existing sparse FedLoRA methods implicitly treat LoRA updates as independent magnitude-ranked tensors. This simplification fails in three ways. First, tensor-level sparsification breaks the structural coupling between LoRA A/B factors and causes severe effective-update distortion. Second, although A/B-pair upload restores structural consistency, factor-norm saliency remains representation-dependent because LoRA factorization is non-unique. Third, effective-update norm is representation-consistent but still magnitude-only: under client heterogeneity, it rewards both shared descent signal and client-specific noise, and it ignores diminishing returns when many clients upload the same module. These observations motivate a signal-noise-aware sparse upload framework that allocates module-level upload quotas according to shared signal, heterogeneity noise, and marginal gain, and then assigns clients through a gap-aware effective-update matching problem.

中文版本：

> 现有 FedLoRA 稀疏上传方法通常隐含地把 LoRA updates 当作独立 tensor，并按照 magnitude 排序上传。这一简化在低预算下会出现三层问题：首先，tensor-level 裁剪破坏了 LoRA A/B factors 的结构耦合，导致 effective update 严重失真；其次，A/B pair 虽然恢复了结构一致性，但基于 factor norm 的 pair saliency 仍然依赖非唯一的 LoRA factor 表示；最后，effective-update norm 虽然具备表示一致性，却仍然只是 magnitude-only 指标，在联邦异质性下会同时奖励 shared descent signal 和 client-specific noise，并且忽略同一 module 多 client 重复上传时的边际收益递减。因此，低预算 FedLoRA sparse upload 应被建模为一个 signal-noise-aware allocation problem：先根据 shared signal、heterogeneity noise 和 marginal gain 分配 module-level upload quota，再通过 gap-aware effective-update matching 将 quota 分配给具体 clients。
