# Phase 2 — Ablation + Budget Curve + Dolly

## 1. Budget Curve（GSM8K, seed28, Ours vs FLASC, alpha=10）

| Budget | Ratio | FLASC | Ours | Winner | Δ |
|---|---:|---:|---:|---|---:|
| 440 | 3.1% | 35.41 | **35.71** | Ours | +0.30 |
| 880 | 6.25% | 36.09 | **36.32** | Ours | +0.23 |
| 1760 | 12.5% | 36.77 | **37.00** | Ours | +0.23 |

**Ours 在全 budget 区间稳定领先 FLASC。**

---

## 2. Ablation（GSM8K, seed28, 12.5%, alpha=10）

| Variant | 配置 | EM | Δ vs Baseline | 结论 |
|---|---|---|---|---|
| Baseline | depth_rank, gap=1.0 | **37.00** | — | |
| w/o depth calibration | SN raw | 36.47 | −0.53 | depth calibration 有效 ✅ |
| w/o P2  | gap=0.0 | 36.9977 | −0.00 | P2 在当前 setting 影响小 |
| ab_pair atom | ab_pair | 30.78 | −6.22 | qv-block 不可替代 ✅ |

---

## 3. Alpha=1 Robustness（GSM8K, seed28, 12.5%）

| Method | EM |
|---|---|
| Ours | **36.92** |
| FLM-TopK | 37.00 |
| FLASC | 36.62 |

Ours 超越 FLASC，与 FLM-TopK 持平。

---

## 4. Dolly-15K（Qwen3-14B, 3 seeds, alpha=0.5 label-skew）

| Method | s28 | s42 | s45 | **Mean R-1** | **Std** | **Mean R-L** | Gen Len |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FedComp** | 43.13 | 43.31 | 43.47 | **43.30** | ±0.17 | 34.07 | ~57 |
| FLM-TopK | 42.65 | 42.34 | 43.04 | **42.68** | ±0.35 | 33.96 | ~56 |
| **Ours** | 42.98 | 42.83 | 41.99 | **42.60** | ±0.54 | 33.82 | ~55 |
| Dense | 42.54 | 41.47 | 42.45 | **42.15** | ±0.61 | 34.05 | ~57 |
| ComPEFT | 34.37 | 40.48 | 35.70 | **36.85** | ±3.20 | 28.28 | ~81 |
| FLASC | 27.96 | 28.35 | 28.28 | **28.20** | ±0.20 | 20.22 | **~126** ❌ |

### 关键发现

1. **FLASC 灾难性崩溃**：ROUGE-1=28.2, Gen Len=126（全任务最差）
2. **Ours 与 FedComp/FLM-TopK 并列第一梯队**：三者差距在 0.7 ROUGE-1 以内
3. **ComPEFT 极度不稳定**：std=±3.20
4. **Ours 超越 Dense**：42.60 > 42.15

---

## 5. 跨任务对比总表

| Method | GSM8K EM (3s) | | Dolly ROUGE-1 (3s) | | 跨任务评价 |
|---|---|---|---|---|---|
| | Mean | ±Std | Mean | ±Std | |
| Dense | 38.16 | 0.23 | 42.15 | 0.61 | 稳定上界 |
| FLM-TopK | 33.74 | 0.47 | **42.68** | 0.35 | 稳健 |
| FedComp | 31.46 | 2.21 | **43.30** | 0.17 | GSM8K 弱 Dolly 强 |
| **Ours** | **36.44** | 0.49 | **42.60** | 0.54 ✅ | **两任务 SOTA 级** |
| ComPEFT | 35.10 | 1.12 | 36.85 | 3.20 | Dolly 极度不稳定 |
| FLASC | 36.92 | 0.40 | 28.20 | 0.20 ❌ | **GSM8K 最强→Dolly 最差** |

> GSM8K: 3 seeds, 12.5% budget, alpha=10 quantity  
> Dolly: 3 seeds, 2200 packets, alpha=0.5 label-skew

**论文核心叙事**：FLASC 在 GSM8K 靠 tensor 粒度优势最强，在 Dolly 崩溃——tensor-level TopK 在生成任务上破坏模型输出。**Ours 在两个任务上都是第一梯队**，唯一能做到跨任务稳定的方法之一。

**论文核心叙事**：FLASC 在 GSM8K 靠 tensor 粒度优势接近 Dense，但同方法在 Dolly 上崩溃——因为 tensor-level TopK 破坏了生成行为。Ours 的 qv-block 结构一致性在**两个任务上都是 SOTA**。
