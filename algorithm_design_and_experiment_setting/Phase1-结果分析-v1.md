# Phase 1 实验结果 — Qwen3-14B + GSM8K Main Table

> 实验时间：2026-06-22  
> 设置：Qwen3-14B-Instruct, GSM8K, 12.5% budget (1760/14080), 50 clients, 10/round, 5 rounds, 10 local epochs, LoRA r=8, α=10

## 完整结果

| Method | Seed 28 | Seed 42 | Seed 45 | **Mean EM** | Std | Gen Len | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| **Dense** | 37.91 | 38.36 | 38.21 | **38.16** | 0.23 | 3.31 | ✅ |
| **FLM-TopK** | 33.97 | 33.21 | 34.04 | **33.74** | 0.47 | 3.43 | ✅ |
| **FedComp** | 32.07 | 29.04 | 33.28 | **31.46** | 2.21 | 3.97 | ✅ |
| **Ours** | 16.07 | 10.84 | 18.95 | **15.29** | 4.16 | 20.22 | ⚠️ |
| **FLASC** | 7.96 | 8.49 | 7.73 | **8.06** | 0.39 | 50.0 | ❌ |
| **ComPEFT** | 6.67 | 6.82 | 5.91 | **6.47** | 0.49 | 49.8 | ❌ |

## 问题分类

### 🔴 致命 Bug：FLASC / ComPEFT 生成异常（Gen Len = 50）

Gen Len 全部触顶 `generation_max_length=50`，说明这两个方法的生成 pipeline 有问题。Qwen3 生成的 output 包含大量无关文本（可能是 Qwen3 的 thinking/explanation tokens），GSM8K answer extractor 无法从中提取正确答案。

**对比**：FLM-TopK 和 FedComp 的 Gen Len 正常（~3-4），说明这个问题只影响 FLASC 和 ComPEFT 两个方法的代码路径。

**诊断方向**：检查 `generation_config` 在 FLASC/ComPEFT 路径下的同步逻辑。可能是在 `baseline_compressors.py` 或 `federated_uie_lora.py` 中，生成配置未正确传递给 tokenizer/model。

### 🟡 Ours 效果不达预期（EM=15.29）

Gen Len 不稳定（9~34），说明 generation 质量波动大。EM 远低于 FLM-TopK（33.74）和 FedComp（31.46）。

可能原因：
1. Generation_config 同步不完整（类似 FLASC 问题但程度较轻）
2. Qwen3 attention pattern 与 Qwen2.5 不同 → qv-block 选择偏好的层不一样
3. depth_balanced 1:1:2 ratio 可能需要针对 Qwen3 重新校准
4. `sn_save_diagnostics=True` 输出需检查选择分布是否合理

### ✅ FLM-TopK 和 FedComp 非常强

| vs Qwen2.5-14B (seed 42) | FLM-TopK | FedComp |
|---|---|---|
| Qwen2.5 | ~22-23 | 32.68 |
| Qwen3 | 33.97 | 32.07 |

Qwen3 的 Dense baseline (38.16) 本身就比 Qwen2.5 (34.45) 高 ~3.7 EM。FLM-TopK 在 Qwen3 上达到 33.74，recover 了 Dense 的 88.4%。

## 下一步

1. 修复 FLASC/ComPEFT generation bug
2. 修复 Ours generation + 调度问题
3. 重跑修复后的 FLASC/ComPEFT/Ours
