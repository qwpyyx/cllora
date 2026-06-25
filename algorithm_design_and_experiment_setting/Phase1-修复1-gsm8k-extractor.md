# Phase 1 修复 #1 — GSM8K Extractor 适配 Qwen3

> 修复时间：2026-06-23

## 问题诊断

### 根因：Qwen3 默认 `do_sample=true, temperature=0.6`

```json
{
    "do_sample": true,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95
}
```

虽然 `prediction_step` 设置了 `do_sample=False`，但 Qwen3 的 verbose 生成行为来自其训练本身的倾向——即使 greedy decoding，训练不足的 LoRA 权重下模型仍会生成 "Okay, let's see..." 这类 thinking text。

### Extractor 不认识 Qwen3 的思考模式

`_EXPLANATION_MARKER_RE` 原 regex 只匹配 `explanation|reasoning|rationale|solution`，不匹配 Qwen3 的 "Okay, let's see", "Let me...", "First,..." 等模式。

→ Extractor 退到 `last-number` fallback → 从解释文本中取到错误数字。

### 为什么 FLM-TopK / FedComp 没受影响

这两方法训练的 LoRA 质量足够好，模型能简洁输出（如 "10"），不触发 verbose 模式。

### 为什么 Ours 也不稳定

Ours 的 LoRA 质量中等，有些 case 输出简洁（Gen Len ~9），有些输出 verbose（Gen Len ~34），extractor 有时能取对有时不能。

## 修复内容

**文件**：`src/gsm8k/gsm8k_metrics.py`

**改动**：扩展 `_EXPLANATION_MARKER_RE` 正则，增加 Qwen3 常见思考模式：

```python
_EXPLANATION_MARKER_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:explanation|reasoning|rationale|solution|解析|解释"
    r"|okay\b|ok\b|let(?:'s|\s+me)\b|first[,，]|now[,，]"
    r")\s*[:：]?"
)
```

新增匹配项：`okay`, `ok`, `let's`, `let me`, `first,`, `now,`

**原理**：当 prediction = "540\n\nOkay, let's see. James runs..." 时，extractor 现在能识别 "Okay" 作为解释起始标记，优先提取前缀 "540" 作为 answer。

**副作用风险评估**：
- 低：GSM8K 正确答案几乎不会出现在 "Okay" 之前作为错误前缀
- 安全：最终仍有 `_ANSWER_CUE_RE` 和 `####` 等更精确的提取路径

## 验证方法

重跑 flasc/compeft/ours 的 seed28，检查 Gen Len 不变但 EM 是否上升。
