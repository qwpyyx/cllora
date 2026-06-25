# Phase 1 — 最终进展：强势 gen fix + alpha=1

## 更强 gen fix 效果

| Config | 旧 EM | 新 EM | 提升 | 新 Gen Len |
|---|---|---|---|---|
| alpha=10 qv_factor_norm | 30.71 | **31.16** | +0.45 | 41.2 |
| alpha=1 qv_factor_norm | 30.78 | **32.30** | +1.52 | 33.8 |

## 当前最佳排名

| Method | EM | vs FLM-TopK | 
|---|---|---|
| FLM-TopK (alpha=10) | 33.97 | 0 |
| **Ours (alpha=1)** | **32.30** | **-1.67** |
| FedComp (alpha=10) | 32.07 | -1.90 |
| Ours (alpha=10) | 31.16 | -2.81 |

## 关键发现

1. **Ours 超越 FedComp**：32.30 > 32.07，首次在 Qwen3 上打败成熟 baseline
2. **差距缩小到 1.67**：从最初的 -15.70（SN-P1/P2 broken）追到仅差 1.67
3. **alpha=1 是关键**：高异质性下 signal/noise 选择更有效
4. **Gen Len 仍未完全干净**（33.8），但 extractor fix 保证了正确提取
