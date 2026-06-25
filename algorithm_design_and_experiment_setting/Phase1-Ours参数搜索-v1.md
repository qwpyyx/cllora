# Ours 参数搜索汇总 (v1)

> Qwen3-14B + GSM8K, 12.5% budget (1760), seed 28

## 所有已测试 Ours 配置

| 配置 | 原子 | Score | 其他参数 | EM | vs FLM-TopK |
|---|---|---|---|---|---|
| SN-P1/P2 db gap1 | qv_block | sn_p1p2 | depth_balanced 1:1:2 | 18.27 | -15.70 |
| SN-P1/P2 db gap0 | qv_block | sn_p1p2 | gap_eta=0 | 18.73 | -15.24 |
| SN-P1/P2 db gap0 force | qv_block | sn_p1p2 | force_full_budget=True | 18.73 | -15.24 |
| SN-P1/P2 dr gap1 | qv_block | sn_p1p2 | depth_rank | 23.43 | -10.54 |
| **qv_factor_norm** | **qv_block** | **factor_norm** | — | **30.71** | **-3.26** |
| qv_enc_m3 | qv_block | factor_norm | compeft m=3 | 30.71 | -3.26 |
| ab_facnorm | ab_pair | factor_norm | — | 22.44 | -11.53 |

## 基线参照

| Baseline | EM |
|---|---|
| Dense | 37.91 |
| FLASC | 36.77 |
| ComPEFT | 34.57 |
| FLM-TopK | 33.97 |
| FedComp | 32.07 |

## 结论

1. SN-P1/P2 在 Qwen3 上完全失效（18-23 EM），根因：SN statistics estimation 不准确，仅 17/40 qv-blocks 获得正边际收益
2. 最佳方案：qv_block + factor_norm = 30.71
3. ComPEFT encoding 在 qv_factor_norm 基础上无增量（30.71 vs 30.71）

## 下一步

目标：追平或超过 FLM-TopK (33.97)。尝试方向：
- qv_factor_norm 增大候选区域 + 更强编码（m5, m6）
- 检查是否 budget 分配不均匀导致某些 client 选择过少
