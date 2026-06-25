# 研究点2-v11：实验结果记录（Follow-up: DepthRank / m5-m6）

> 本文档基于 2026-06-22 已完成的 `gsm8k_ours_encoding_followup_depthrank_m6_pack.zip` 更新。  
> 这轮实验的目标是补齐上一版留下的两个关键问题：  
> 1. Qwen2.5 上 `enc_compeft_m6` 的单 seed 高结果是否能多 seed 复现；  
> 2. Llama31 上 depth-rank 在更高预算和更大 candidate multiplier 下是否继续领先。  

---

## 0. 本轮实验状态

本轮 follow-up 实验全部成功：

| Model | Result dir | Status |
|---|---|---|
| Llama-3.1-8B | `results/Llama31_gsm8k/ours_encoding_followup_depthrank_m6` | `overall_failed=0` |
| Qwen2.5-14B | `results/Qwen2_gsm8k/ours_encoding_followup_depthrank_m6` | `overall_failed=0` |

没有发现 Traceback / RuntimeError / OOM。所有 `.status` 均为 `SUCCESS`。

---

## 1. 当前总判断

上一版结论是：

```text
Depth-Rank Ours-Enc / depth_rank_compeft_m3
是当前跨模型默认配置。
```

本轮实验后，需要更精细地表述：

```text
跨模型默认配置：depth_rank_compeft_m3
Qwen2.5 单模型最佳：enc_compeft_m5
Llama31 12.5% 预算最佳：depth_rank_compeft_m3
Llama31 高预算最佳：enc/depth-rank m3/m4 接近，差异很小
```

换句话说，最终论文中不建议把 `m5` 直接写成通用默认。更稳妥的写法是：

> We use `depth_rank_compeft_m3` as the default configuration because it is the most robust across model families. We also report a candidate-size sensitivity study, where Qwen2.5 benefits from a larger candidate multiplier around `m=5`, while Llama31 favors the depth-rank `m=3` configuration under the main 12.5% budget.

---

## 2. Qwen2.5-14B + GSM8K

### 2.1 12.5% budget 主结果（2112 packets）

合并 all-day 与 follow-up 后，Qwen2.5 在 2112 budget 下的关键结果为：

| Variant | Seeds | Mean EM | Std | 结论 |
|---|---:|---:|---:|---|
| enc_compeft_m3 | 28/42/45 | 35.5825 | 0.4377 | 原 Ours-Enc 默认 |
| enc_compeft_m4 | 28/42/45 | 35.5573 | **0.1313** | 最稳定，但均值略低 |
| **enc_compeft_m5** | **28/42/45** | **35.8605** | 0.2734 | Qwen 当前最佳 |
| enc_compeft_m6 | 28/42/45 | 35.7847 | 0.5252 | seed42 很强，但方差较大 |
| enc_compeft_m7 | 28/42/45 | 35.6836 | 0.2316 | 开始回落 |
| enc_compeft_m8 | 28/42/45 | 35.7089 | 0.2275 | 未超过 m5 |
| depth_rank_compeft_m3 | 28/42/45 | 35.7341 | 0.3064 | 跨模型默认候选 |
| depth_rank_compeft_m4 | 28/42/45 | 35.3551 | 0.3740 | 不如 m3 |
| depth_rank_compeft_m5 | 28/42/45 | 35.6583 | 0.1578 | 稳定但不如 enc_m5 |
| depth_rank_compeft_m6 | 28/42/45 | 35.6836 | 0.1908 | 稳定但不如 enc_m5 |
| rawp1_compeft_m3 | 28/42/45 | 35.6078 | 0.2663 | P1 ablation |
| enc_flasc_m3 | 28/42/45 | 34.9002 | 0.9171 | encoder ablation |

排序上，Qwen2.5 的 12.5% budget 三 seed 最强配置是：

```text
enc_compeft_m5: 35.8605
enc_compeft_m6: 35.7847
depth_rank_compeft_m3: 35.7341
enc_compeft_m8: 35.7089
enc_compeft_m7 / depth_rank_m6: 35.6836
```

### 2.2 Qwen 的关键解释

本轮最重要的新发现是：

```text
Qwen2.5 上，candidate multiplier 从 m3 增大到 m5 后，多 seed 平均继续提升；
但继续增大到 m6/m7/m8 后不再稳定提升。
```

这说明 Qwen2.5 的候选区域需要比最初的 m3 更大，但也不是越大越好。

| Variant | m | Mean EM | Std |
|---|---:|---:|---:|
| enc_compeft_m3 | 3 | 35.5825 | 0.4377 |
| enc_compeft_m4 | 4 | 35.5573 | 0.1313 |
| **enc_compeft_m5** | **5** | **35.8605** | 0.2734 |
| enc_compeft_m6 | 6 | 35.7847 | 0.5252 |
| enc_compeft_m7 | 7 | 35.6836 | 0.2316 |
| enc_compeft_m8 | 8 | 35.7089 | 0.2275 |

因此，Qwen2.5 的 sensitivity 结论应写为：

> A moderate candidate expansion improves the encoder. The best three-seed result appears around `m=5`; excessively large candidate regions no longer improve the average, suggesting that the scheduled support should remain selective.

### 2.3 Qwen：depth-rank 的位置要重新表述

上一版我们把 `depth_rank_compeft_m3` 作为当前最强方法。但 follow-up 表明：

```text
depth_rank_m3 仍然强，但不是 Qwen 单模型最佳；
Qwen 单模型最佳是 enc_compeft_m5。
```

对比：

| Variant | Mean EM | Difference vs depth_rank_m3 |
|---|---:|---:|
| depth_rank_compeft_m3 | 35.7341 | 0 |
| enc_compeft_m5 | **35.8605** | **+0.1264** |
| enc_compeft_m6 | 35.7847 | +0.0506 |
| depth_rank_compeft_m6 | 35.6836 | -0.0505 |
| depth_rank_compeft_m5 | 35.6583 | -0.0758 |

解释：

- depth-rank 在 m3 下很稳，适合作为跨模型默认；
- 但 Qwen2.5 更偏好更大的 candidate support；
- 当 m 增大到 5/6 后，depth-rank 反而不如 depth-balanced enc_m5；
- 这说明 Qwen 的收益更多来自 **candidate support size**，而不是继续加强 depth-rank。

因此论文里建议：

```text
Default: depth_rank_compeft_m3
Qwen best tuned: enc_compeft_m5
```

### 2.4 Qwen 低预算 depth-rank curve

本轮补齐了 depth_rank_m3 在低预算下的多 seed：

| Budget | Ratio | Seeds | Mean EM | Std |
|---:|---:|---:|---:|---:|
| 1056 | 6.25% | 28/42/45 | 34.6980 | 0.5790 |
| 1408 | 8.33% | 28/42/45 | 35.2793 | 0.1578 |
| 1760 | 10.42% | 28/42/45 | 35.3045 | 0.3891 |
| 2112 | 12.50% | 28/42/45 | **35.7341** | 0.3064 |

对比 dense seed42 reference：

```text
Dense full upload seed42 = 35.1782
```

结论：

- `depth_rank_m3` 在 8.33% budget 就达到 35.2793，已经超过 dense seed42 reference；
- 12.5% budget 达到 35.7341，是低预算 curve 中最强；
- 这说明 depth-rank 对低预算尤其有价值。

### 2.5 Qwen 高预算补充

| Variant | Budget | Seed42 EM | 说明 |
|---|---:|---:|---|
| enc_compeft_m4 | 2816 | 36.0121 | all-day seed42 |
| enc_compeft_m4 | 2816 | 35.7847 mean | 28/42/45 mean |
| enc_compeft_m6 | 2816 | 35.4814 | follow-up seed42 |
| depth_rank_compeft_m6 | 2816 | 35.4814 | follow-up seed42 |

高预算并没有简单带来更高平均值。当前可写为：

> Increasing the actual upload budget beyond the main 12.5% point does not monotonically improve Qwen2.5, indicating that the best candidate multiplier and the actual budget should be jointly selected.

---

## 3. Llama-3.1-8B + GSM8K

### 3.1 12.5% budget 主结果（1144 packets）

Llama31 在 1144 budget 下的最终对比：

| Variant | Seeds | Mean EM | Std | 结论 |
|---|---:|---:|---:|---|
| enc_compeft_m3 | 28/42/45 | 17.8165 | 0.6206 | 原 Ours-Enc |
| enc_compeft_m4 | 28/42/45 | 17.6143 | 0.4175 | 不如 m3 |
| enc_compeft_m6 | 28/42/45 | 18.1450 | 0.5048 | 有提升但不如 depth-rank |
| rawp1_compeft_m3 | 28/42/45 | 17.9681 | 0.3305 | P1 ablation |
| depth_rank_compeft_m4 | 28/42/45 | 18.2714 | **0.2626** | 稳定但略低 |
| **depth_rank_compeft_m3** | **28/42/45** | **18.4230** | 0.4612 | Llama 12.5% 最佳 |
| enc_flasc_m3 | 28/42/45 | 16.5782 | 0.4442 | encoder ablation |

结论：

```text
Llama31 在主预算 12.5% 下仍然最支持 depth_rank_compeft_m3。
```

这和 Qwen 不同：Qwen 的 tuned best 是 m5，Llama 的 best 是 depth-rank m3。

### 3.2 Llama 高预算 curve

Llama 对实际上传预算更敏感。本轮补齐后，关键 curve 如下：

#### enc_compeft_m3

| Budget | Ratio | Seeds | Mean EM | Std |
|---:|---:|---:|---:|---:|
| 1144 | 12.5% | 28/42/45 | 17.8165 | 0.6206 |
| 1716 | 18.75% | 28/42/45 | 18.5494 | 0.7744 |
| 2288 | 25.0% | 28/42/45 | **19.4339** | 0.6175 |
| 2860 | 31.25% | 42 | 19.3328 | - |
| 3432 | 37.5% | 42 | 19.8635 | - |

#### depth_rank_compeft_m3

| Budget | Ratio | Seeds | Mean EM | Std |
|---:|---:|---:|---:|---:|
| 1144 | 12.5% | 28/42/45 | 18.4230 | 0.4612 |
| 1716 | 18.75% | 28/42/45 | 19.0296 | 0.6610 |
| 2288 | 25.0% | 28/42/45 | 19.2823 | 0.5740 |
| 2860 | 31.25% | 42 | 19.1812 | - |
| 3432 | 37.5% | 42 | 19.8635 | - |

#### depth_rank_compeft_m4

| Budget | Ratio | Seeds | Mean EM | Std |
|---:|---:|---:|---:|---:|
| 1144 | 12.5% | 28/42/45 | 18.2714 | 0.2626 |
| 2288 | 25.0% | 28/42/45 | **19.4339** | 0.7781 |

解释：

- 12.5% budget 下，`depth_rank_m3` 最好；
- 18.75% budget 下，`depth_rank_m3` 继续明显强于 `enc_m3`；
- 25% budget 下，`enc_m3` / `enc_m4` / `depth_rank_m4` 都达到 19.4339 mean，`depth_rank_m3` 略低；
- 31.25%/37.5% 目前只有 seed42，不能作为主结论；
- Llama 高预算下方法差异变小，实际 budget 的影响更强。

### 3.3 Llama 与 Qwen 的差异

| Observation | Qwen2.5 | Llama31 |
|---|---|---|
| 主预算最佳 | enc_compeft_m5 | depth_rank_compeft_m3 |
| depth-rank m3 是否稳 | 是，35.7341 | 是，18.4230 |
| 增大 m 是否持续收益 | m5 最好，m6 后波动 | m6 不如 depth-rank m3 |
| 提高 actual budget | 不单调 | 明显提升 |

这说明：

```text
candidate multiplier 是模型相关的 sensitivity parameter；
depth-rank m3 是跨模型鲁棒默认；
Qwen 可以通过更大的 m 得到 tuned best；
Llama 更依赖 actual budget 增加。
```

---

## 4. 方法与论文叙事更新

### 4.1 不建议再把单一配置写成绝对最优

现在实验已经说明：

```text
Qwen tuned best: enc_compeft_m5
Llama main-budget best: depth_rank_compeft_m3
```

所以论文中不建议说：

```text
depth_rank_compeft_m3 is always the best.
```

更准确的说法是：

```text
depth_rank_compeft_m3 is the default robust configuration.
candidate multiplier m controls the size of the scheduled candidate support.
The sensitivity study shows that Qwen2.5 benefits from a larger support around m=5,
whereas Llama31 favors depth-rank m=3 under the main budget.
```

### 4.2 最终方法命名建议

方法名仍建议保持：

```text
Ours-Enc / Depth-Rank Ours-Enc
```

但实验表中可以区分：

| Name in paper | Implementation |
|---|---|
| Ours-Enc | `enc_compeft_m3` or scheduled encoder |
| Ours-Enc-DR | `depth_rank_compeft_m3` |
| Ours-Enc-Tuned | best per-model m, e.g., Qwen `enc_compeft_m5` |

如果担心审稿人认为 tuned variant 不公平，主表建议用：

```text
Ours-Enc-DR / depth_rank_compeft_m3
```

sensitivity 图中再展示：

```text
Qwen m5 can further improve the tuned result.
```

### 4.3 当前最稳主表建议

主表可采用：

| Model | Method | Budget ratio | Seeds | Mean EM |
|---|---|---:|---:|---:|
| Qwen2.5 | ComPEFT | 12.5% | 42 | 31.8423 |
| Qwen2.5 | FedComp | 12.5% | 42 | 32.6763 |
| Qwen2.5 | Raw Ours | 12.5% | 42 | 29.9469 |
| Qwen2.5 | Ours-Enc-DR | 12.5% | 28/42/45 | 35.7341 |
| Qwen2.5 | Ours-Enc-Tuned | 12.5% | 28/42/45 | 35.8605 |
| Llama31 | ComPEFT | 12.5% | multi-seed | 16.1486 |
| Llama31 | FedComp | 12.5% | multi-seed | 13.9753 |
| Llama31 | Raw Ours | 12.5% | multi-seed | 15.7190 |
| Llama31 | FLM-TopK | 12.5% | multi-seed | 17.9429 |
| Llama31 | Ours-Enc-DR | 12.5% | 28/42/45 | 18.4230 |

注意：

- Qwen 的 `Ours-Enc-Tuned` 是 `enc_compeft_m5`；
- Llama 的 `Ours-Enc-DR` 就是当前主预算最佳；
- 如果篇幅有限，主表可以只放 `Ours-Enc-DR`，把 tuned result 放 sensitivity。

---

## 5. 当前最终结论

本轮实验后的最准确结论是：

> **Ours-Enc 的核心价值已经稳定成立：在相同上传预算下，先进行 signal-noise-aware candidate scheduling，再做 fine-grained encoding，能显著优于单独 encoding 或单独 scheduling。Depth-rank m3 是跨模型最稳默认配置；Qwen2.5 可以通过更大的 candidate multiplier（m5）得到更高 tuned performance；Llama31 则在主预算下更支持 depth-rank m3，并在更高实际预算下持续提升。**

对应论文叙事：

```text
The method is not a single hard-coded m value.
It is a scheduled encoding framework.
The default configuration uses depth-rank m3 for robustness,
and the sensitivity study shows how candidate support size affects different models.
```

---

## 6. 后续实验建议

当前不建议继续大规模盲跑。最值得补的是：

### 6.1 Qwen2.5

如果还想把 tuned result 做稳：

```text
enc_compeft_m5 seed=41/43
depth_rank_compeft_m3 seed=41/43
```

目的：

- 进一步确认 `enc_m5` 是否稳定高于 `depth_rank_m3`；
- 若 m5 在 5 seeds 上仍最高，可在论文里作为 tuned variant。

### 6.2 Llama31

如果要强化高预算曲线：

```text
enc_compeft_m3 budget=3432 seed28/45
depth_rank_compeft_m3 budget=3432 seed28/45
```

目的：

- 验证 37.5% budget 的 seed42 高结果是否稳定；
- 但这不是主表必须项。

### 6.3 更重要的下一步

比继续 GSM8K 细扫更重要的是：

```text
换数据集 / 换任务
```

建议下一阶段优先：

1. Dolly / instruction generation；
2. financial / medical dataset 中选一个；
3. Qwen3 或 Gemma 的 smoke test。

因为当前 GSM8K 证据已经足够支撑方法主线，继续只扫 GSM8K 的边际收益会变低。
