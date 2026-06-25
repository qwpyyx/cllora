# INFOCOM-论文motivation 数据溯源文档

> 每个数据点均标注来源文件路径与提取方式。  
> 数据基于 Qwen2.5-14B-Instruct + GSM8K，12.5% budget，seed 42。

---

## 1. Jump 1：Tensor-TopK 灾难性失败（§2.1）

### 1.1 GSM8K EM 数据（已有）-以图1数据来源的excel为准，这是正确的

|           上传粒度 | Seed 28 | Seed 42 | Seed 45 |    **Mean** |
|---:|---:|---:|---:|---:|
| Tensor-level Top-K |  1.8196 |  1.8954 |  1.5921 |   **1.769** |
|    A/B pair atomic | 27.5208 | 28.3548 | 28.1274 |  **28.001** |
|   Q/V block atomic | 28.3548 | 28.2032 | 27.6725 | **28.0768** |
|  Dense full upload | 34.7991 | 34.4958 | 34.7991 |  **34.698** |

**来源**：`研究点2-v9-实验结果.md` §1.2（原始记录）。  
**对应 zip**：`D:\CodeLife\Third\results\gsm8k_effnorm_vs_factor\gsm8k_multiseed_key_qvnorm_dense_pack.zip`  
**提取路径**：各子目录 `*/all_results.json` → `predict_gsm8k_em`

| 数据点 | 文件 |
|---|---|
| Tensor seed28 | `tensor_topk_budget2112_K10_seed28/all_results.json` |
| AB pair seed28 | `ab_effective_budget2112_K10_seed28/all_results.json`（注：AB-Effective 和 AB-Factor 都用了 A/B pair 原子，seed28 的 AB-Factor=29.26，AB-Effective=27.60） |
| QV block seed28 | `ours_depth_balanced_budget2112_K10_seed28/all_results.json` |
| Dense seed28 | `dense_full_K10_seed28/all_results.json` |

**注意**：Tensor-level Top-K 的逐 seed 数据来自 `研究点2-v9-motivation.md` §2.1 原始记录，zip 中未直接包含 tensor_topk 目录。AB pair 和 QV block 的逐 seed 数据在 zip 中的实际目录名与 motivation 文档记录略有不同，平均值已在 `研究点2-v9-实验结果.md` 中核对一致。

### 1.2 Dolly 生成行为数据（已有）

| 法                                       | Exact Match | ROUGE-1 | ROUGE-L | Gen Len |
| ---------------------------------------- | ----------: | ------: | ------: | ------: |
| tensor-level Top-K                       |      0.0000 | 28.0820 | 20.3345 |  127.79 |
| A/B pair Top-K                           |      3.5333 | 44.3689 | 35.2605 |   64.52 |
| A/B pair + group mask G=4                |      2.9333 | 44.3953 | 35.3654 |   63.63 |
| A/B pair + coverage penalty $\beta=0.05$ |      3.2667 | 44.2471 | 35.0878 |   65.79 |
| A/B pair + coverage penalty $\beta=0.1$  |      3.0667 | 44.3184 | 35.1365 |   67.57 |
| Dense                                    |      5.4667 | 42.5395 | 35.1083 | 51.6567 |

**来源**：`研究点2-v9-motivation.md` §2.1 原始记录。  
**对应 zip**：Dolly 数据不在 `gsm8k_effnorm_vs_factor` 目录中，来源于早期实验，原始文件路径在**/home/qiuwenqi/LLM/Third_work/results/Qwen2_dolly/lora_exp_dolly_alpha05**。

---

## 2. Jump 2：Selection Concentration（§3.2）

### 2.1 逐轮 Jaccard + Union + Fully Shared（Qwen2.5-14B, GSM8K, 12.5% budget, seed 42）

**来源 zip**：`D:\CodeLife\Third\results\gsm8k_effnorm_vs_factor\gsm8k_multiseed_key_qvnorm_dense_pack.zip`

#### AB-Effective

| Round | Pairwise Jaccard | Union Selected | Fully Shared |
|---:|---:|---:|---:|
| 1 | 0.82 | 24 | 10 |
| 2 | 1.00 | 20 | 20 |
| 3 | 1.00 | 20 | 20 |
| 4 | 1.00 | 20 | 20 |
| 5 | 1.00 | 20 | 20 |

**来源文件**：`ab_effective_budget2112_K10_seed42/selection_overlap_history.json`  
**提取字段**：`pairwise_jaccard_mean`, `union_selected_layers`, `fully_shared_layers`

#### AB-Factor

| Round | Pairwise Jaccard | Union Selected | Fully Shared |
|---:|---:|---:|---:|
| 1 | 0.74 | 30 | 10 |
| 2 | 0.84 | 34 | 12 |
| 3 | 0.60 | 32 | 8 |
| 4 | 0.55 | 44 | 6 |
| 5 | 0.43 | 52 | 6 |

**来源文件**：`ab_factor_budget2112_K10_seed42/selection_overlap_history.json`  
**提取字段**：同上

#### Ours (SN-P1/P2, depth_balanced)

| Round | Pairwise Jaccard | Union Selected | Fully Shared |
|---:|---:|---:|---:|
| 1 | 0.19 | 80 | 0 |
| 2 | 0.17 | 80 | 0 |
| 3 | 0.18 | 80 | 0 |
| 4 | 0.17 | 84 | 0 |
| 5 | 0.16 | 84 | 0 |

**来源文件**：`ours_depth_balanced_budget2112_K10_seed42/selection_overlap_history.json`  
**提取字段**：同上

### 2.2 汇总对比表

| 指标 | AB-Effective | AB-Factor | Ours |
|---|---:|---:|---:|
| Jaccard 均值 (R1-R5) | 0.96 | 0.63 | 0.18 |
| Union 均值 (R1-R5) | 20.8 | 38.4 | 81.6 |
| Fully Shared 均值 (R1-R5) | 18.0 | 8.4 | 0.0 |
| Fully Shared (R1) | 10 | 10 | 0 |
| Fully Shared (R5) | 20 | 6 | 0 |
| Fully Shared 趋势 | 递增（10→20） | 递减（10→6） | 始终为 0 |

**解释**：
- AB-Effective：fully_shared 从 R1 的 10 个增加到 R2-R5 的 20 个。即从第 2 轮开始，所有 10 个 clients 选中的 20 个 modules **完全相同**（Jaccard=1.00）。
- AB-Factor：fully_shared 从 R1 的 10 个逐渐降到 R5 的 6 个。即虽然选择有一定集中度，但不像 AB-Effective 那样完全 collapse。Jaccard 从 0.74 降到 0.43，说明选择越来越分散。
- Ours：fully_shared 始终为 0，即**没有任何 module 被所有 10 个 clients 同时选中**。Jaccard 稳定在 0.16-0.19，union 覆盖 80-84 个 modules。

### 2.3 Llama-3.1-8B 验证数据（多 seed 均值）

| Method | Avg Jaccard | Union Selected | Fully Shared | Mean Selected |
|---:|---:|---:|---:|---:|
| AB-Effective | 0.605 | 22.1 | 4.7 | 12.0 |
| AB-Factor | 0.262 | 48.1 | 0.5 | 14.5 |
| Ours | 0.177 | 55.7 | 0.0 | 16.0 |

**来源**：`研究点2-v9-实验结果.md` §2.4（Llama-3.1-8B + GSM8K，12.5% budget，3 seeds 平均）。  
**对应 zip**：`D:\CodeLife\Third\results\gsm8k_effnorm_vs_factor\llama31_gsm8k_formal_overnight_pack.zip`  
**提取路径**：`*/selection_overlap_history.json`（跨 seed 平均后汇总）

---

## 3. 最终 Performance 数据（§4）

### 3.1 Qwen2.5-14B + GSM8K, 12.5% budget, 3 seeds

| 方法 | Mean EM | Std |
|---:|---:|---:|
| Dense | 34.45 | 0.64 |
| AB-Factor | 28.81 | 0.42 |
| AB-Effective | 27.75 | 0.13 |
| **Ours** | **30.45** | **0.49** |

**来源**：`研究点2-v9-实验结果.md` §1.2  
**对应 zip**：`gsm8k_multiseed_key_qvnorm_dense_pack.zip`  
**提取路径**：各 `*_seed*/all_results.json` → `predict_gsm8k_em`。Std 为 3 seeds 的 sample standard deviation。

| 数据点 | Seed 28 | Seed 42 | Seed 45 |
|---|---|---|---:|
| Dense | 34.12 | 35.18 | 34.04 |
| AB-Factor | 29.26 | 28.43 | 28.73 |
| AB-Effective | 27.60 | 27.82 | 27.82 |
| Ours | 30.93 | 29.95 | 30.48 |

### 3.2 Llama-3.1-8B + GSM8K, 12.5% budget, 3 seeds

| 方法 | Mean EM | Std |
|---:|---:|---:|
| Dense | 17.46 | 0.34 |
| AB-Factor | 6.44 | 0.42 |
| AB-Effective | 5.53 | 0.30 |
| **Ours** | **15.72** | **0.69** |

**来源**：`研究点2-v9-实验结果.md` §2.2  
**对应 zip**：`llama31_gsm8k_formal_overnight_pack.zip`  
**提取路径**：各 `*_seed*/all_results.json` → `predict_gsm8k_em`。Std 为 3 seeds 的 sample standard deviation。

---

## 4. 数据文件清单

| 数据 | 本地路径 | 服务器路径 |
|---|---|---|
| Qwen2.5 multi-seed | `D:\CodeLife\Third\results\gsm8k_effnorm_vs_factor\gsm8k_multiseed_key_qvnorm_dense_pack.zip` | 服务器 `results/Qwen2_gsm8k/multiseed_key_qvnorm_dense/` |
| Qwen2.5 AB vs Ours | `D:\CodeLife\Third\results\gsm8k_effnorm_vs_factor\gsm8k_matched_ab_vs_ours_qvnorm_seed42_pack.zip` | 服务器 `results/Qwen2_gsm8k/` |
| Llama-3.1 formal | `D:\CodeLife\Third\results\gsm8k_effnorm_vs_factor\llama31_gsm8k_formal_overnight_pack.zip` | 服务器 `results/Llama31_gsm8k/` |
| Llama-3.1 baselines | `D:\CodeLife\Third\results\gsm8k_effnorm_vs_factor\llama31_gsm8k_baselines_formal_pack.zip` | 服务器 `results/Llama31_gsm8k/` |

---

## 5. 画图脚本数据嵌入位置

`plot_motivation_figures.py` 中硬编码的数组及其对应 JSON 字段：

### Fig 1

| Python 变量 | 值 | 来源 |
|---|---|---|
| `em` | `[1.77, 28.00, 28.08, 34.70]` | §1.1 均值 |
| `genlen` | `[127.79, 64.52]` | §1.2 Dolly 数据 |
| `rougel` | `[20.33, 35.26]` | §1.2 Dolly 数据 |

### Fig 2

| Python 变量 | 值 | 来源 JSON 字段 |
|---|---|---|
| `jac_eff` | `[0.82, 1.00, 1.00, 1.00, 1.00]` | `ab_effective_.../selection_overlap_history.json` → `pairwise_jaccard_mean` |
| `jac_fac` | `[0.74, 0.84, 0.60, 0.55, 0.43]` | `ab_factor_.../selection_overlap_history.json` → `pairwise_jaccard_mean` |
| `jac_ours` | `[0.19, 0.17, 0.18, 0.17, 0.16]` | `ours_.../selection_overlap_history.json` → `pairwise_jaccard_mean` |
| `uni_eff` | `[24, 20, 20, 20, 20]` | `ab_effective_.../selection_overlap_history.json` → `union_selected_layers` |
| `uni_fac` | `[30, 34, 32, 44, 52]` | `ab_factor_.../selection_overlap_history.json` → `union_selected_layers` |
| `uni_ours` | `[80, 80, 80, 84, 84]` | `ours_.../selection_overlap_history.json` → `union_selected_layers` |
| `shared_eff` | `[10, 20, 20, 20, 20]` | `ab_effective_.../selection_overlap_history.json` → `fully_shared_layers` |
| `shared_fac` | `[10, 12, 8, 6, 6]` | `ab_factor_.../selection_overlap_history.json` → `fully_shared_layers` |
| `shared_ours` | `[0, 0, 0, 0, 0]` | `ours_.../selection_overlap_history.json` → `fully_shared_layers` |

所有 JSON 文件均位于 `gsm8k_multiseed_key_qvnorm_dense_pack.zip` 解压后的目录中。
