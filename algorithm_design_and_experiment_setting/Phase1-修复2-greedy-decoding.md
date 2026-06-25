# Phase 1 修复 #2 — 强制 Greedy Decoding + Extractor 适配

> 修复时间：2026-06-23

## 修复概述

两个独立修复，协同解决 Qwen3 的生成与评估问题。

### 修复 1：强制 Greedy Decoding（`src/uie_trainer_lora.py`）

**问题**：`prediction_step` 中创建 `GenerationConfig(**gen_kwargs)` 并设置 `do_sample=False`，但 `model.generate()` 在 transformers 4.51 中会自动用模型原生配置覆盖——Qwen3 的 `generation_config.json` 有 `do_sample=True, temperature=0.6`，导致 evaluation 时实际用了随机采样。

```
WARNING: `generation_config` default values have been modified to match 
model-specific defaults: {'do_sample': True, 'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}
```

**修复**：`GenerationConfig(**gen_kwargs)` 之后显式再设 `do_sample=False`：

```python
generation_config = GenerationConfig(**gen_kwargs)
generation_config.do_sample = False
generation_config.temperature = 1.0
```

### 修复 2：GSM8K Extractor 适配（`src/gsm8k/gsm8k_metrics.py`）

**问题**：`_EXPLANATION_MARKER_RE` 只匹配 `explanation|reasoning|rationale|solution`，不匹配 Qwen3 的 "Okay, let's...", "Let me...", "First,..." 等 thinking 模式。

**修复**：扩展正则，新增 `okay`, `ok`, `let's`, `let me`, `first,`, `now,`。

## 修复后预期

- FLASC：Gen Len 大幅下降，EM 恢复
- ComPEFT：同
- Ours：Gen Len 稳定且下降，EM 提升
- Dense/FLM-TopK/FedComp：基本不受影响（它们本就是 greedy 输出）
