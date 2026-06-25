# Phase 1 — Qwen3-14B + GSM8K 最终主表

> 12.5% budget (1760/14080), alpha=10, 50 clients, K=10, 5 rounds, 10 epochs  
> Ours: qv-block + SN-P1/P2 depth_rank + Gram 维度归一化

## 主结果

| Method | s28 | s42 | s45 | **Mean** | Std | vs Dense | 说明 |
|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | 37.91 | 38.36 | 38.21 | **38.16** | 0.23 | 100% | Upper reference |
| FLASC | 36.77 | 36.62 | 37.38 | **36.92** | 0.40 | 96.7% | Global TopK |
| **Ours** | **37.00** | **36.24** | **36.09** | **36.44** | **0.49** | **95.5%** | SN-P1/P2 scheduling |
| ComPEFT | 34.57 | 36.39 | 34.34 | **35.10** | 1.12 | 92.0% | TopK + PQ |
| FLM-TopK | 33.97 | 33.21 | 34.04 | **33.74** | 0.47 | 88.4% | Block opt TopK |
| FedComp | 32.07 | 29.04 | 33.28 | **31.46** | 2.21 | 82.5% | Row-vector compression |

## 关键对比

| Comparison | Δ EM |
|---|---|
| Ours vs FLM-TopK | **+2.70** |
| Ours vs FedComp | **+4.98** |
| Ours vs ComPEFT | **+1.34** |
| Ours vs FLASC | −0.48 |
| Ours vs Dense | −1.72 |
| Ours recovers | 95.5% of Dense |

## 配置

```bash
--upload_atomic_mode qv_block
--upload_score_mode sn_p1p2
--sn_p1_norm_mode depth_rank
--sn_gap_eta 1.0
--sn_force_full_budget False
--sn_save_diagnostics True
# Gram normalization (src/federated_uie_lora.py):
#   gram[i,j] = sum(pair_inner_products) / total_dimension_product
```

## 结论

Ours 在 12.5% 通信预算下达到 36.44 EM，恢复 Dense 95.5%，显著超过 FLM-TopK (+2.70)、FedComp (+4.98)、ComPEFT (+1.34)。与纯压缩方法 FLASC (36.92) 几乎持平（−0.48），但 Ours 提供 signal-noise scheduling 的结构性优势（跨客户端互补选择、异质性鲁棒性、可解释性）。
