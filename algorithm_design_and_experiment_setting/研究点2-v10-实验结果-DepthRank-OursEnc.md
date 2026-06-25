# 研究点2-v10：实验结果记录（Depth-Rank Ours-Enc）

> 本文档基于截至 2026-06-21 已完成的 GSM8K 实验结果更新。  
> 相比上一版，核心变化是：最终方法不再只是 `enc_compeft_m3`，而应进一步调整为 **Depth-Rank Ours-Enc**，即 `depth_rank_compeft_m3`。  
> 当前两个模型（Qwen2.5-14B 与 Llama-3.1-8B）的多 seed 结果都表明：depth-rank candidate allocation 能稳定增强 scheduled fine-grained encoding。

---

## 0. 当前结论总览

当前方法主线建议改为：

```text
Depth-Rank Ours-Enc
= signal-noise-aware qv-block candidate scheduling
+ depth-rank allocation
+ ComPEFT-style fine-grained encoding
```

推荐默认配置：

```text
depth_rank_compeft_m3:
  upload_atomic_mode = qv_block
  upload_score_mode = sn_p1p2
  sn_p1_norm_mode = depth_rank
  sn_depth_group_ratios = ""
  sn_gap_eta = 0.0
  sn_force_full_budget = True
  sn_encoder_mode = compeft
  sn_candidate_budget_multiplier = 3
  sn_encoder_packet_num = comm_budget
```

当前关键结论：

1. **Raw Ours** 证明了 qv-block + signal-noise scheduling 有效，但 raw block upload 编码效率不足。
2. **Ours-Enc** 证明了 scheduling 与 fine-grained encoding 互补，能显著超过单独 ComPEFT/FedComp。
3. **Depth-Rank Ours-Enc** 进一步说明，候选区域不能只依赖固定 `lower:middle:upper = 1:1:2`，更均衡的 depth-rank allocation 在两个模型上更稳。
4. Qwen2.5 与 Llama31 都支持同一个最终叙事：**where-to-encode 比 how-to-encode alone 更关键，而 depth-aware scheduling 决定 where-to-encode 的质量。**

---

## 1. 实验设置

### 1.1 统一设置

| Setting | Value |
|---|---:|
| Dataset | GSM8K |
| Total clients | 50 |
| Clients per round | 10 |
| Global rounds | 5 |
| Local epochs | 10 |
| LoRA rank | 8 |
| Upload target | q_proj, v_proj |
| Partition | quantity / Dirichlet alpha 10.0 |
| Main seeds | 28, 42, 45 |
| Main metric | GSM8K exact match / EM |

### 1.2 模型与预算

| Model | Full qv-block cost | Main budget | Budget ratio |
|---|---:|---:|---:|
| Qwen2.5-14B-Instruct | 16896 packets/client/round | 2112 | 12.5% |
| Llama-3.1-8B-Instruct | 9152 packets/client/round | 1144 | 12.5% |

说明：

- `comm_budget` 是实际上传预算；
- `sn_candidate_budget_multiplier` 只控制候选区域大小；
- 实际 encoder 上传预算始终为 `sn_encoder_packet_num = comm_budget`；
- 因此 Ours-Enc 与 ComPEFT/FedComp/FLASC 在实际 packet budget 上是对齐的。

---

## 2. 方法版本定位

### 2.1 Raw Ours：scheduling-only ablation

```text
Raw Ours
= qv-block atomic upload
+ SN-P1/P2 scheduling
+ raw selected qv-block upload
```

Raw Ours 的价值：

- 证明 tensor-level / A-B-level magnitude selection 不是最优；
- 证明 qv-block 更符合 LoRA effective update 的结构；
- 证明 P1/P2 可以让不同 clients 上传互补区域。

Raw Ours 的局限：

- 每个 qv-block 成本较高；
- 低预算下候选区域太少；
- block 内部没有细粒度编码，通信效率不如成熟压缩方法。

因此 Raw Ours 在论文中建议作为：

```text
Ours w/o Encoding
```

### 2.2 Ours-Enc：scheduled fine-grained encoding

```text
Ours-Enc
= SN-P1/P2 candidate scheduling
+ in-candidate ComPEFT-style TopK + PQ
```

它回答的问题是：

```text
Given a fixed encoder budget, where should the encoder spend it?
```

这也是区别于 ComPEFT/FedComp 的核心：

- ComPEFT/FedComp 主要决定 **how to encode**；
- Ours-Enc 主要决定 **where to encode**；
- Depth-Rank Ours-Enc 进一步决定 **how to allocate candidate support across model depth**。

### 2.3 Depth-Rank Ours-Enc：当前默认主方法

早期 `enc_compeft_m3` 使用：

```text
sn_p1_norm_mode = depth_balanced
sn_depth_group_ratios = 1,1,2
```

也就是更偏向 upper layers。全天实验表明，这个固定比例不是最稳。Depth-rank 配置改为：

```text
sn_p1_norm_mode = depth_rank
sn_depth_group_ratios = ""
```

其效果是让候选 quota 在深度上更均衡。例如：

| Model | Variant | Budget | Candidate quota by depth |
|---|---|---:|---|
| Llama31 | enc_compeft_m3 | 1144 | lower 30 / middle 30 / upper 60 |
| Llama31 | depth_rank_compeft_m3 | 1144 | lower 40 / middle 40 / upper 40 |
| Qwen2.5 | enc_compeft_m3 | 2112 | lower 45 / middle 45 / upper 90 |
| Qwen2.5 | depth_rank_compeft_m3 | 2112 | lower 60 / middle 60 / upper 60 |

这个变化使方法叙事从“固定 depth heuristic”升级为“depth-aware candidate allocation”。

---

## 3. Qwen2.5-14B + GSM8K

### 3.1 12.5% budget 主结果

结合已完成结果，Qwen2.5 在 2112 packets/client/round 下的主表如下：

| Method | Seeds | Mean EM | Std | Notes |
|---|---:|---:|---:|---|
| Dense full upload | 42 | 35.1782 | - | Full upload reference |
| ComPEFT | 42 | 31.8423 | - | Global TopK + PQ |
| FedComp | 42 | 32.6763 | - | Row-vector compression + residual |
| FLASC | 42 | 22.2896 | - | Global TopK |
| Raw Ours | 42 | 29.9469 | - | SN scheduling + raw qv-block upload |
| enc_compeft_m3 | 28/42/45 | 35.5825 | 0.4377 | depth-balanced Ours-Enc |
| rawp1_compeft_m3 | 28/42/45 | 35.6078 | 0.2663 | raw P1 score + encoder |
| enc_compeft_m4 | 28/42/45 | 35.5573 | **0.1313** | larger candidate region |
| **depth_rank_compeft_m3** | **28/42/45** | **35.7341** | **0.3064** | current default |
| depth_rank_compeft_m4 | 28/42/45 | 35.3551 | 0.3740 | depth-rank + larger m |
| enc_flasc_m3 | 28/42/45 | 34.9002 | 0.9171 | scheduler with FLASC encoder |

核心比较：

| Comparison | EM Gain |
|---|---:|
| depth_rank_m3 - ComPEFT | +3.8918 |
| depth_rank_m3 - FedComp | +3.0578 |
| depth_rank_m3 - Raw Ours | +5.7872 |
| depth_rank_m3 - Dense seed42 reference | +0.5559 |
| depth_rank_m3 - enc_compeft_m3 | +0.1516 |
| depth_rank_m3 - enc_compeft_m4 | +0.1768 |
| depth_rank_m3 - rawp1_compeft_m3 | +0.1263 |

结论：

> Qwen2.5 上，`depth_rank_compeft_m3` 是当前三 seed 平均最高的稳定配置。`enc_compeft_m4` 方差更低，但平均略低；`m5/m6/m7/m8` 单 seed 很强，需要下一轮补 seed。

### 3.2 Candidate multiplier sweep

Qwen2.5 在 2112 budget、seed42 下的 m sweep：

| Variant | m | Seed42 EM | Candidate qv-blocks/client | Candidate tensors/client | Density |
|---|---:|---:|---:|---:|---:|
| enc_compeft_m2 | 2 | 35.1782 | 12 | 48 | 41.32% |
| enc_compeft_m3 | 3 | 35.3298 | 18 | 72 | 26.86% |
| enc_compeft_m4 | 4 | 35.6331 | 24 | 96 | 20.14% |
| enc_compeft_m5 | 5 | 35.9363 | 30 | 120 | 16.11% |
| **enc_compeft_m6** | **6** | **36.0879** | **36** | **144** | **13.10%** |
| enc_compeft_m7 | 7 | 35.9363 | 42 | 168 | 11.23% |
| enc_compeft_m8 | 8 | 35.7089 | 48 | 192 | 9.83% |

当前判断：

- Qwen2.5 上，candidate region 扩大到 m6 时单 seed 最强；
- m7/m8 开始回落，说明候选区域过大后 SN gate 被稀释；
- 由于 m5/m6/m7/m8 目前只有 seed42，不能直接作为最终默认；
- 下一轮最值得补的是 `enc_compeft_m6 seed28/45` 和 `depth_rank_compeft_m5/m6 multi-seed`。

### 3.3 Qwen budget curve

#### enc_compeft_m3

| Budget | Ratio | Seed42 EM |
|---:|---:|---:|
| 1056 | 6.25% | 33.5102 |
| 1408 | 8.33% | 34.5716 |
| 1760 | 10.42% | 35.3298 |
| 2112 | 12.50% | 35.3298 |
| 2464 | 14.58% | 35.2540 |
| 2816 | 16.67% | 35.7847 |
| 3520 | 20.83% | 35.4814 |

#### depth_rank_compeft_m3

| Budget | Ratio | Seed42 EM |
|---:|---:|---:|
| 1056 | 6.25% | 34.5716 |
| 1408 | 8.33% | 35.3298 |
| 1760 | 10.42% | 34.8749 |
| 2112 | 12.50% | 36.0121 |
| 2816 | 16.67% | 34.9507 |

解释：

- Qwen 在低预算下，depth-rank 明显强于 depth-balanced；
- 1056 budget 时，depth-rank 比 enc_m3 高 +1.0614 EM；
- 1408 budget 时，depth-rank 比 enc_m3 高 +0.7582 EM；
- 2112 budget 是目前最稳的主预算；
- 2816 budget 下 depth_rank_m3 反而下降，说明更高预算不一定单调提升，需要结合 candidate multiplier 重新调参。

### 3.4 Encoder ablation

| Method | Seeds | Mean EM | Std |
|---|---:|---:|---:|
| FLASC baseline | 42 | 22.2896 | - |
| enc_flasc_m3 | 28/42/45 | 34.9002 | 0.9171 |
| enc_flasc_m4 | 42 | 35.0265 | - |
| enc_compeft_m3 | 28/42/45 | 35.5825 | 0.4377 |
| depth_rank_compeft_m3 | 28/42/45 | 35.7341 | 0.3064 |

结论：

> SN scheduling 不依赖 ComPEFT encoder。即使换成 FLASC encoder，也能显著超过原始 FLASC baseline。但 ComPEFT-style encoder 仍然提供更高上限和更稳定结果。

---

## 4. Llama-3.1-8B + GSM8K

### 4.1 12.5% budget 主结果

Llama31 在 1144 packets/client/round 下：

| Method | Seeds | Mean EM | Std | Notes |
|---|---:|---:|---:|---|
| Dense full upload | multi-seed | 17.4627 | - | Full upload reference |
| ComPEFT | multi-seed | 16.1486 | - | Baseline |
| FedComp | multi-seed | 13.9753 | - | Baseline |
| FLASC | multi-seed | 12.5095 | - | Baseline |
| FLM-TopK | multi-seed | 17.9429 | - | Strong but expensive baseline |
| Raw Ours | multi-seed | 15.7190 | - | Scheduling-only ablation |
| enc_compeft_m3 | 28/42/45 | 17.8165 | 0.6206 | depth-balanced Ours-Enc |
| rawp1_compeft_m3 | 28/42/45 | 17.9681 | 0.3305 | raw P1 + encoder |
| enc_compeft_m4 | 28/42/45 | 17.6143 | 0.4175 | larger candidate region |
| **depth_rank_compeft_m3** | **28/42/45** | **18.4230** | **0.4612** | current default |
| enc_flasc_m3 | 28/42/45 | 16.5782 | 0.4442 | scheduler with FLASC encoder |

核心比较：

| Comparison | EM Gain |
|---|---:|
| depth_rank_m3 - ComPEFT | +2.2744 |
| depth_rank_m3 - FedComp | +4.4477 |
| depth_rank_m3 - FLASC | +5.9135 |
| depth_rank_m3 - Raw Ours | +2.7040 |
| depth_rank_m3 - Dense reference | +0.9603 |
| depth_rank_m3 - FLM-TopK | +0.4801 |
| depth_rank_m3 - enc_compeft_m3 | +0.6065 |

结论：

> Llama31 上 depth-rank 的收益比 Qwen 更明显。它不仅超过 ComPEFT/FedComp/Raw Ours，也超过之前最强的 FLM-TopK mean。

### 4.2 Llama budget curve

#### enc_compeft_m3

| Budget | Ratio | Seeds | Mean EM | Std |
|---:|---:|---:|---:|---:|
| 858 | 9.375% | 42 | 16.6793 | - |
| 1144 | 12.5% | 28/42/45 | 17.8165 | 0.6206 |
| 1716 | 18.75% | 28/42/45 | 18.5494 | 0.7744 |
| 2288 | 25.0% | 28/42/45 | **19.4339** | 0.6175 |

#### enc_compeft_m4

| Budget | Ratio | Seeds | Mean EM | Std |
|---:|---:|---:|---:|---:|
| 1144 | 12.5% | 28/42/45 | 17.6143 | 0.4175 |
| 1716 | 18.75% | 42 | 18.1956 | - |
| 2288 | 25.0% | 28/42/45 | **19.4339** | 0.7781 |

#### depth_rank_compeft_m3

| Budget | Ratio | Seeds | EM |
|---:|---:|---:|---:|
| 1144 | 12.5% | 28/42/45 | 18.4230 mean |
| 1716 | 18.75% | 42 | 18.2714 |
| 2288 | 25.0% | 42 | 19.0296 |

解释：

- Llama 对预算更敏感；
- 12.5% budget 已经超过 dense/ComPEFT/FedComp；
- 25% budget 时，`enc_m3` 与 `enc_m4` 都达到 19.4339 mean；
- 下一轮需要补 `depth_rank_m3 budget=1716/2288 seed28/45`，判断高预算下 depth-rank 是否继续领先。

### 4.3 Candidate multiplier and encoder ablation

| Variant | Budget | Seeds | Mean EM | Notes |
|---|---:|---:|---:|---|
| enc_compeft_m2 | 1144 | 42 | 16.9826 | candidate region too small |
| enc_compeft_m3 | 1144 | 28/42/45 | 17.8165 | default depth-balanced |
| enc_compeft_m4 | 1144 | 28/42/45 | 17.6143 | slightly lower than m3 |
| enc_compeft_m5 | 1144 | 42 | 17.2100 | not promising |
| enc_compeft_m6 | 1144 | 42 | 17.8923 | worth checking seeds |
| enc_flasc_m3 | 1144 | 28/42/45 | 16.5782 | weaker but still above FLASC baseline |
| enc_flasc_m4 | 1144 | 42 | 16.6793 | no clear gain |

Llama 与 Qwen 的差异：

- Qwen 在 2112 budget 下 m6 单 seed 最强；
- Llama 在 1144 budget 下 m3/depth_rank 更稳；
- Llama 高预算时 m3/m4 都明显提升；
- 因此 candidate multiplier 不应写成固定最优，而应写成可配置的 candidate support size；论文默认取 m3，因为跨模型最稳。

---

## 5. 综合结论

### 5.1 跨模型主结果

| Model | Final default | Budget ratio | Seeds | Mean EM | Main conclusion |
|---|---|---:|---:|---:|---|
| Qwen2.5-14B | depth_rank_compeft_m3 | 12.5% | 28/42/45 | **35.7341** | 超过 Dense/ComPEFT/FedComp/Raw Ours |
| Llama-3.1-8B | depth_rank_compeft_m3 | 12.5% | 28/42/45 | **18.4230** | 超过 Dense/ComPEFT/FedComp/Raw Ours/FLM-TopK |

这说明：

```text
Depth-rank scheduled encoding is not a Qwen-only phenomenon.
It generalizes to a different model family and size.
```

### 5.2 当前论文表述建议

英文可以这样写：

```text
Ours-Enc achieves strong performance across both Qwen2.5-14B and Llama-3.1-8B under the same 12.5% uplink budget. More importantly, replacing the fixed depth-balanced allocation with depth-rank scheduling further improves the three-seed average on both models, reaching 35.73 EM on Qwen2.5 and 18.42 EM on Llama3.1. These results show that the key benefit does not come from transmitting more nonzero values, but from assigning the limited encoding budget to signal-noise-favorable and depth-balanced candidate regions.
```

中文理解：

```text
Ours-Enc 在 Qwen2.5-14B 和 Llama-3.1-8B 上均能在 12.5% 上传预算下取得强性能。更重要的是，将固定 depth-balanced allocation 替换为 depth-rank scheduling 后，两个模型的三 seed 平均性能进一步提升。这说明性能收益并不是来自上传更多非零值，而是来自把有限编码预算分配到 signal-noise 更有利、深度覆盖更合理的候选区域。
```

### 5.3 方法章节应如何调整

原 v9 写法：

```text
qv-block scheduling is the final sparse upload method.
```

建议 v10 写法：

```text
qv-block scheduling is a candidate-region scheduler for fine-grained encoding.
```

方法结构建议：

1. **Effective-update atomicity**：解释为什么 qv-block 是合理通信原子；
2. **Signal-noise scheduling**：解释 P1/P2 如何决定候选区域；
3. **Depth-rank allocation**：解释为什么不固定偏向 upper layers，而采用 rank-based depth allocation；
4. **Fine-grained in-candidate encoding**：解释在候选区域内做 TopK+PQ；
5. **Aggregation**：按 FedAvg 聚合 encoded sparse updates。

---

## 6. 当前仍需补充的实验

### 6.1 Qwen2.5 最优配置确认

必须补：

```text
enc_compeft_m6 seed28/45
depth_rank_compeft_m5 seed28/42/45
depth_rank_compeft_m6 seed28/42/45
```

原因：

- `enc_compeft_m6 seed42 = 36.0879`，当前 Qwen 单 seed 最强；
- 但只有单 seed，不能作为最终默认；
- 如果 depth-rank + m5/m6 多 seed 超过 depth_rank_m3，则最终默认可能从 m3 改为 m5/m6。

### 6.2 Llama 高预算 depth-rank curve

必须补：

```text
depth_rank_compeft_m3 budget=1716 seed28/45
depth_rank_compeft_m3 budget=2288 seed28/45
depth_rank_compeft_m4 budget=1144/2288 seeds28/42/45
```

原因：

- Llama 高预算下 `enc_m3/m4` 已经达到 19.4339 mean；
- depth_rank_m3 在 1144 budget 下明显更强；
- 需要确认高预算下 depth-rank 是否继续领先。

### 6.3 推荐下一轮脚本

下一轮脚本命名：

```text
run_gsm8k_ours_encoding_followup_depthrank_m6.sh
```

跑完后建议打包：

```text
gsm8k_ours_encoding_followup_depthrank_m6_pack.zip
```

---

## 7. 当前投稿风险与应对

### 风险 1：Dense reference 被超过，审稿人可能疑惑

应对：

```text
Dense full upload is not an optimization upper bound under non-IID FL and finite-round training.
Sparse/encoded updates may act as regularization and suppress noisy client-specific directions.
```

写法上避免说：

```text
exceeds the upper bound
```

建议说：

```text
matches or slightly exceeds the dense reference
```

### 风险 2：candidate multiplier 是否是调参堆出来的

应对：

- 主默认用 m3，因为跨模型稳定；
- m-sweep 作为 sensitivity；
- Qwen m6 单 seed 强，但不直接作为最终默认，除非补 seed 后仍显著领先；
- 强调 m 控制的是 candidate support size，不改变实际 uplink budget。

### 风险 3：depth-rank 是否只是 heuristic

应对：

- 不把 depth-rank 写成孤立 heuristic；
- 把它放在 P1 allocation 的 normalization layer 中；
- 解释其目标是避免固定 depth ratio 对不同模型不稳；
- 用两个模型三 seed 支持其泛化性。

---

## 8. 当前最终判断

当前最稳的论文结论是：

> **The final contribution is not raw qv-block sparse upload, but depth-rank signal-noise-aware scheduled encoding. Under the same 12.5% uplink budget, it consistently improves over standalone fine-grained compression and scheduling-only upload on both Qwen2.5-14B and Llama-3.1-8B.**

对应中文：

> **我们的最终贡献不是 raw qv-block sparse upload，而是 depth-rank signal-noise-aware scheduled encoding。它在相同 12.5% 上传预算下，在 Qwen2.5-14B 和 Llama-3.1-8B 上都稳定超过单独细粒度压缩和单独调度。**

因此，后续论文版本建议统一采用：

```text
Depth-Rank Ours-Enc
```

作为最终主方法名或默认配置名。
