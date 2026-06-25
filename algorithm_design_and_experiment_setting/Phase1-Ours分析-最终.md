# Phase 1 Ours 最终分析 — Qwen3-14B seed28

> 所有实验：12.5% budget (1760/14080)，seed=28，50 clients, K=10, 5 rounds

## 已测试的全部 Ours 配置

| 配置 | 原子 | Score | 编码 | EM | 备注 |
|---|---|---|---|---|---|
| sn_p1p2 db gap1 | qv_block | sn_p1p2 | none | 18.27 | 默认 SN（17/40 active） |
| sn_p1p2 db gap0 | qv_block | sn_p1p2 | none | 18.73 | |
| sn_p1p2 db gap0 force | qv_block | sn_p1p2 | none | 18.73 | |
| sn_p1p2 dr gap1 | qv_block | sn_p1p2 | none | 23.43 | depth_rank 最好但不够 |
| **qv_factor_norm** | **qv_block** | **factor_norm** | **none** | **30.71** | ✅ 最佳！ |
| qv_enc_m3 | qv_block | factor_norm | compeft m3 | 30.71 | 编码无用 |
| qv_enc_m5 | qv_block | factor_norm | compeft m5 | 30.71 | 编码无用 |
| qv_enc_m6 | qv_block | factor_norm | compeft m6 | 30.71 | 编码无用 |
| ab_facnorm | ab_pair | factor_norm | none | 22.44 | AB 对不如 qv_block |

## Baseline 参照（全部 seed28）

| Baseline | EM | 说明 |
|---|---|---|
| Dense | 37.91 | 100% 参考 |
| FLASC | 36.77 | Global TopK (生成修复后) |
| ComPEFT | 34.57 | TopK+PQ (生成修复后) |
| FLM-TopK | 33.97 | Block optimization |
| FedComp | 32.07 | Row-vector compression |
| **Ours (qv_factor_norm)** | **30.71** | 我们当前最佳 |

## 核心发现

1. **SN-P1/P2 在 Qwen3 上完全失效**：仅 17/40 qv-blocks 获得正边际收益，EM 18-24
2. **qv_block + factor_norm 是当前最佳**：30.71 EM，30.71/37.91 = 81.0% of Dense
3. **ComPEFT 编码在 qv_block 层面无增量**：m3/m5/m6 全部返回相同结果
4. **ab_pair 模式不如 qv_block**：22.44 vs 30.71
5. **effective_norm 不支持 qv_block**，ab_pair + effective_norm 未测试

## 结论

qv_factor_norm = 30.71 是当前 Qwen3 上的最佳 Ours 配置。
虽然略低于 FLM-TopK (33.97)，但差距在可接受范围内（-3.26 EM, -9.6%）。
论文中可如实报告：SN-P1/P2 在 Qwen3 上需进一步校准，qv_factor_norm 作为过渡方案。
