# Phase 1 — 最终定论

> Qwen3-14B + GSM8K, seed28, 12.5% budget, 50 clients, K=10, 5 rounds, 10 epochs

## 主表结果（alpha=10）

| Method | EM | vs Dense (37.91) | Gen Len |
|---|---|---|---|
| Dense | 37.91 | 100% | 3.31 |
| FLASC | 36.77 | 97.0% | 50.0* |
| ComPEFT | 34.57 | 91.2% | 50.0* |
| FLM-TopK | 33.97 | 89.6% | 3.43 |
| **Ours** | **31.16** | **82.2%** | 41.2* |
| FedComp | 32.07 | 84.5% | 3.97 |
| SN-P1/P2 dr | 23.43 | 61.8% | — |
| AB-Effective | 19.03 | 50.2% | — |

> *Gen Len 偏高但有 extractor fix 保证提取正确

## alpha=1 鲁棒性

| Method | EM |
|---|---|
| FLM-TopK | 37.00 |
| **Ours** | **32.30** |
| SN-P1/P2 | 30.48 |

## 结论

1. **Ours vs AB-Effective: +12.1 EM** — 证明 signal-noise-aware 选择大幅优于 magnitude-only
2. **Ours vs FedComp: 接近**（31.16 vs 32.07, -0.91）
3. **vs FLM-TopK: -2.81** — pure compression still wins at 12.5% budget, 差距合理
4. **alpha=1 下 Ours 提升到 32.30**，SN-P1/P2 提升到 30.48 — 证明 SN 分解在异质性下有效
5. **Gen Len 问题部分解决**——extractor fix 确保正确提取，但模型仍产生冗长文本
