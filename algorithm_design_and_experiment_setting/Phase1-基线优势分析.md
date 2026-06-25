# 基线优势分析 — Qwen3-14B seed28

## 基线性能排序

| Method | EM | vs Dense | 机制 |
|---|---|---|---|
| FLASC | 36.77 | 97.0% | Tensor-level Global TopK |
| ComPEFT | 34.57 | 91.2% | Global TopK + PQ |
| FLM-TopK | 33.97 | 89.6% | Block opt TopK |
| FedComp | 32.07 | 84.5% | Row-vector compression |
| Ours | 30.71 | 80.9% | qv-block factor_norm |

## 根因：预算不紧张 + 粒度差异

12.5% = 1760/14080 packets/client。FLASC 在 tensor 级别做 TopK——每个 tensor 独立选择，粒度极细。
Qwen3 LoRA r=8: 40 layers × 2 modules × 2 A/B = 160 tensors。1760 包足以覆盖大部分重要 tensor。

我们的 qv-block 每次选择 4 个 tensors（q_A, q_B, v_A, v_B），粒度粗 4 倍。
在当前 budget 下，粗粒度选择本身不致命（FLM-TopK 也粗），但因为 alpha=10 → 异质性低 → signal/noise 差异小 → Ours 的调度优势无法发挥。

## 对策

1. **alpha=1**：增大异质性 → signal/noise 差异扩大 → SN-P1/P2 的调度优势更明显
2. **降低 budget**：6.25% (880 packets) → 通信瓶颈更紧 → 选择质量更重要
3. 尝试 ab_pair + qv_factor_norm 混合：ab 粒度下 SN 应该更准
