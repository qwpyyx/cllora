# INFOCOM 2027 — Motivation：为什么低预算 FedLoRA 需要 Signal-Noise-Aware Sparse Upload

> 本文档整理论文 motivation section 的完整论证链。  
> 核心逻辑为**两跳结构**：  
> **Jump 1**：Tensor-level Top-K 灾难性失败 → 上传原子必须是完整 A/B pair effective update  
> **Jump 2**：即使原子正确，magnitude-only selection 导致跨客户端选择高度集中 → 需要 signal-noise-aware allocation  
> 两跳之后自然过渡到方法：$P_1$ 做 module-level quota allocation，$P_2$ 做 client-level complementary assignment。

---

## 0. 核心叙事一句话

> Low-budget sparse FedLoRA upload cannot treat LoRA tensors as independent compression units, nor can it rank effective updates purely by magnitude. The correct approach is to preserve shared descent signals under client heterogeneity through signal-noise-aware allocation.

---

## 1. 背景：FedLoRA Sparse Upload 为什么不是普通压缩

Federated LoRA fine-tuning 中，每个 client 在第 $t$ 轮训练后产生一组 module-wise LoRA updates：

$$
U_{i,m} = \big(U_{A,i,m}, U_{B,i,m}\big), \quad m=1,\dots,M.
$$

在 cross-device 或带宽受限场景中，完整上传所有 $M$ 个 modules 的 A/B pairs 仍然可能超出通信预算。于是自然产生一个问题：

> **在每轮通信预算 $C_i$ 下，client $i$ 应该上传哪些 LoRA modules？**

表面上，这是一个 sparse selection 问题——直觉上可以按某种重要性度量排序，选择 top-K 个上传。但 FedLoRA 与普通 tensor compression 有本质区别：LoRA update 不是一个独立 tensor，而是由两个低秩 factors 共同定义的 effective weight update：

$$
\Phi_{i,m} = (B_m + U_{B,i,m})(A_m + U_{A,i,m}) - B_m A_m.
$$

这意味着：**上传单元的定义**和**重要性的度量**都必须在 effective-update space 中考虑，而不是在原始 factor space 中简单做 tensor-level 操作。

本文通过两个实验现象逐层揭示低预算 FedLoRA sparse upload 的核心挑战。

---

## 2. Jump 1：Tensor-level Top-K 的灾难性失败

### 2.1 现象

把 LoRA A/B tensors 当作独立压缩单元，按 tensor norm 做 Top-K 选择上传。在 GSM8K 上的结果是灾难性的：

| 上传粒度 | Seed 28 | Seed 42 | Seed 45 | **Mean** |
|---:|---:|---:|---:|---:|
| Tensor-level Top-K | 1.82 | 1.59 | 1.90 | **1.77** |
| A/B pair atomic | 27.52 | 28.13 | 28.35 | **28.00** |
| Q/V block atomic | 28.35 | 27.67 | 28.20 | **28.08** |
| Dense full upload | 34.80 | 34.80 | 34.50 | **34.70** |

> 实验设置：Qwen2.5-14B-Instruct + GSM8K，12.5% budget（2112 packets），50 clients / 10 per round，5 rounds，10 local epochs，3 seeds。

Tensor-level Top-K 不是"略差"——它几乎完全失效（EM = 1.77）。而**只要把上传单元从独立 tensor 改为完整 A/B pair，性能立即恢复到 28.00**——提升 26.2 个百分点，恢复了 dense 性能的 80.7%。

在 Dolly-15K 开放式 instruction generation 上，tensor-level Top-K 甚至导致生成行为异常：

| 方法 | EM | ROUGE-1 | ROUGE-L | Gen Len |
|---:|---:|---:|---:|---:|
| Tensor-level Top-K | 0.00 | 28.08 | 20.33 | **127.79** |
| A/B pair Top-K | 3.53 | 44.37 | 35.26 | 64.52 |

> Gen Len 接近生成上限 128，说明 tensor-level upload 破坏了模型的生成行为。

### 2.2 机制解释

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

### 2.3 第一跳结论

> Sparse FedLoRA cannot treat LoRA tensors as independent compression units. The atomic upload unit must be a complete A/B pair that induces an intact effective update $\Phi_{i,m}$.

这一步同时隐式地满足了 **representation consistency**：一旦以 $\Phi_{i,m}$ 作为上传原子和 saliency 定义空间，重要性度量就不再依赖非唯一的 LoRA factor 表示——这是 $\Phi$ 的数学属性，不需要额外的实验论证。

---

## 3. Jump 2：Magnitude-only Selection 导致 Cross-Client Concentration

### 3.1 A/B pair 正确但还不够

Jump 1 确定了上传单元应该是 A/B pair induced effective update $\Phi_{i,m}$。下一个问题是：如何度量每个 $\Phi_{i,m}$ 的"重要性"？

最直接的做法是在 effective-update space 中按 Frobenius norm 排序：

$$
\|\Phi_{i,m}\|_F^2,
$$

选择 top-K 上传。这个方法以正确的原子为基础，saliency 定义在 effective-update space 中——但它仍然有一个致命缺陷：**magnitude-only selection 在联邦多客户端环境下会导致所有 clients 把预算集中到完全相同的 modules 上**。

### 3.2 实验现象：Selection Concentration

在 Qwen2.5-14B + GSM8K，12.5% budget，seed 42 的设置下，逐轮跟踪每轮 10 个 clients 的选择行为：

| Round | AB-Effective Jaccard | AB-Factor Jaccard | **Ours Jaccard** | AB-Eff Union | **Ours Union** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.82 | 0.74 | **0.19** | 24 | **80** |
| 2 | **1.00** | 0.84 | **0.17** | 20 | **80** |
| 3 | **1.00** | 0.60 | **0.18** | 20 | **80** |
| 4 | **1.00** | 0.55 | **0.17** | 20 | **84** |
| 5 | **1.00** | 0.43 | **0.16** | 20 | **84** |
| **Mean** | **0.96** | 0.63 | **0.18** | 20.8 | **81.6** |

| 指标 | AB-Effective | AB-Factor | **Ours** |
|---|---:|---:|---:|
| Round 2-5 Fully Shared Modules | 20/20 | 12→6 | **0** |
| Union Selected (平均) | 20.8 | 38.4 | **81.6** |

**核心观察**：

- AB-Effective（按 $\|\Phi_{i,m}\|_F^2$ 排序）从第 2 轮开始，**所有 10 个 clients 选择完全相同的 20 个 modules**——Jaccard = 1.00。
- AB-Factor 也高度集中（Jaccard 0.43-0.84）。
- 而 Ours（signal-noise-aware P1/P2 allocation）保持 Jaccard ≈ 0.17，union coverage 是 AB-Effective 的 **4 倍**，fully shared modules **始终为 0**。

同现象在 Llama-3.1-8B 上也成立（多 seed 均值）：

| Method | Avg Jaccard | Union Selected | Fully Shared | Mean Selected |
|---:|---:|---:|---:|---:|
| AB-Effective | 0.605 | 22.1 | 4.7 | 12.0 |
| AB-Factor | 0.262 | 48.1 | 0.5 | 14.5 |
| **Ours** | **0.177** | **55.7** | **0.0** | **16.0** |

### 3.3 为什么 Magnitude-only Selection 导致 Concentration？

这个现象不是偶然的。在联邦异质环境下，每个 client 的 effective update 可以分解为：

$$
\Phi_{i,m} = \mu_m + \xi_{i,m},
$$

其中 $\mu_m$ 是跨 clients 共享的 descent signal，$\xi_{i,m}$ 是 client-specific deviation（heterogeneity noise）。

Effective norm 的期望是：

$$
\mathbb{E}\|\Phi_{i,m}\|_F^2 = \underbrace{\|\mu_m\|_F^2}_{a_m} + \underbrace{\mathbb{E}\|\xi_{i,m}\|_F^2}_{b_m}.
$$

如果所有 clients 面对相同的 shared signal $\mu_m$，那么 $\|\Phi_{i,m}\|_F^2$ 的 cross-client ranking 主要受 $a_m$ 驱动。在低预算下，每个 client 独立选出 top-K 时，自然都指向同一批 $a_m$ 最大的 modules——因为 shared signal 越强的 module，所有 clients 的 $\|\Phi_{i,m}\|_F^2$ 都倾向于更大。

**这揭示了一个核心矛盾**：magnitude-based saliency 把 heterogeneity noise $b_m$ 和 shared signal $a_m$ 一起当作正贡献，但：

- 对全局 FedLoRA 聚合来说，真正有价值的是 $a_m$（shared descent signal）；
- $b_m$（client-specific noise）不仅不应被奖励，还应被抑制；
- 当同一个 module 被多个 clients 重复上传时，边际收益递减。

更精确地，如果 module $m$ 被 $k_m$ 个 clients 上传（保持 $1/K$ FedAvg scaling），expected preservation error 为：

$$
J_m(k_m) = \left(1 - \frac{k_m}{K}\right)^2 a_m + \frac{k_m}{K^2} b_m.
$$

第 $k+1$ 次上传的边际收益为：

$$
\Delta_m(k) = J_m(k) - J_m(k+1) = \frac{(2(K-k)-1)a_m - b_m}{K^2}.
$$

关键含义：
1. **$a_m$ 是正贡献，$b_m$ 是负贡献**——magnitude-only 把两者都当正面信号；
2. **边际收益随 $k$ 增大严格递减**——存在 diminishing returns；
3. **当 $(2(K-k)-1)a_m \leq b_m$ 时，继续上传该 module 的边际收益变为负**。

因此，magnitude-only Top-K 无法回答"某个 module 被上传 5 次后，第 6 次还有没有正收益"——它看不到 diminishing returns，也分不清 signal 和 noise。

### 3.4 为什么不是简单的 Diversity 或 Coverage？

前期探索过 coverage penalty 和 group mask 等 diversity 方向。这些方法确实改变了选择分布，但没有稳定提升性能。原因在于：

> FedLoRA sparse upload 的核心不是"让 clients 选得越不一样越好"。如果某个 module 的 shared signal $a_m$ 很强、noise $b_m$ 很低，让多个 clients 上传它**是合理的**。真正的问题是 magnitude-only Top-K **无法判断这种重复是否仍有正边际收益**。

因此，本文不把 diversity 或 coverage 作为独立优化目标，而是从 signal-noise marginal gain 推导 allocation——这比简单惩罚 overlap 更有理论依据。

### 3.5 第二跳结论

> Even with the correct effective-update atom, magnitude-only Top-K selection causes extreme cross-client concentration (Jaccard → 1.00). The fundamental issue is that raw magnitude conflates shared descent signal with heterogeneity noise and ignores diminishing returns. Low-budget FedLoRA sparse upload requires signal-noise-aware allocation: each module's upload quota should be determined by its shared signal strength, noise level, and marginal gain.

---

## 4. 最终 Performance 数据：动机转化为方法后的结果

在 Qwen2.5-14B 和 Llama-3.1-8B 两个模型上，基于上述 motivation 构建的 Ours 方法（qv-block + depth-balanced P1 + P2-L）显著优于 magnitude-only baselines：

| Model | Dense | AB-Factor | AB-Effective | **Ours** | Ours vs Best AB | Sparse-to-Dense Gap Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-14B | 34.45 | 28.81 | 27.75 | **30.45** | +1.64 | 29.1% |
| Llama-3.1-8B | 17.46 | 6.44 | 5.53 | **15.72** | +9.27 | 84.5% |

> 12.5% budget，3 seeds (28, 42, 45)，GSM8K。

---

## 5. 从 Motivation 到 Method 的自然过渡

基于上述分析，低预算 FedLoRA sparse upload 应满足两个核心要求：

### Requirement 1：有效的上传原子
上传单元必须是完整 A/B pair induced effective update $\Phi_{i,m}$。这同时保证了 structural consistency 和 representation consistency——前者来自 A/B pair 的乘法结构闭合性，后者来自 $\Phi$ 的表示不变性。

### Requirement 2：Signal-noise-aware allocation
重要性度量不能只是 magnitude。需要区分 shared signal $a_m$ 和 heterogeneity noise $b_m$，并根据边际收益 $\Delta_m(k)$ 决定每个 module 的上传配额 $k_m^*$，再通过 gap-aware assignment 将配额分配给具体 clients。

这自然导向本文方法的两层决策结构：

> **P1 (module-level allocation)**：按 $\Delta_m(k) = \frac{(2(K-k)-1)a_m - b_m}{K^2}$ 分配每个 module 的上传次数 $k_m^*$；
>
> **P2 (client-level assignment)**：给定 $k_m^*$，通过 gap-aware quadratic-to-linear surrogate 决定具体哪些 clients 上传哪些 modules。

完整方法推导见 `研究点2-v9-method.md`。

---

## 6. 推荐图表

| Fig | 内容 | 数据类型 | 放置 |
|---|---|---|---|
| **Fig 1** | Tensor-TopK catastrophic failure：tensor vs A/B pair vs qv-block vs Dense 柱状图 | 柱状图 | Jump 1 末尾 |
| **Fig 2** | Cross-client selection concentration：逐轮 Jaccard + Union modules（3 条线覆盖 5 轮） | 双面板折线图 | Jump 2 末尾 |

仅需 2 张图即可完成 motivation 的全部实验论证。

---

## 7. 实验设置速查

| Setting | Value |
|---|---:|
| Model | Qwen2.5-14B-Instruct（主 motivation） |
| Dataset | GSM8K |
| Total clients | 50 |
| Clients per round | 10 |
| Global rounds | 5 |
| Local epochs | 10 |
| LoRA rank | 8 |
| Upload target | q_proj, v_proj |
| Main budget ratio | 12.5% |
| Seeds | 28, 42, 45 |

> 注：motivation 实验使用 Qwen2.5-14B 已完成数据。Tensor-TopK 和 concentration 现象是 LoRA 数学结构 + magnitude-based selection 的直接后果，与具体模型无关。论文中主实验将切换为 Qwen3-14B-Instruct（详见实验计划文档）。
