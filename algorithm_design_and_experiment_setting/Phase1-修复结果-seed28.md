# Phase 1 修复结果 — Seed28 快速验证

> 修复：强制 greedy decoding + GSM8K extractor 适配 Qwen3

## 结果

| Method | 修复前 EM | 修复后 EM | 变化 | vs Dense (37.91) |
|---:|---:|---:|---:|---:|
| FLASC | 7.96 | **36.77** | +28.81 ✅ | 97.0% of Dense |
| ComPEFT | 6.67 | **34.57** | +27.90 ✅ | 91.2% of Dense |
| Ours | 16.07 | **18.27** | +2.20 ⚠️ | 48.2% of Dense |

## 分析

### FLASC / ComPEFT — 修复成功

修复前 Gen Len=50 时 extractor 从解释文本取到错误数字。强制 greedy + extractor 适配后，即使生成长度仍接近 50，但"答案优先"格式（如 "10.\n\nOkay, let's verify..."）被正确提取首个数字，EM 恢复正常。

### Ours — 仍不达预期

EM 仅从 16.07 → 18.27，提升有限。Gen Len=17.8（改善但不稳定）。说明 Ours 的问题不在生成/评估，而在模型质量本身——SN-P1/P2 选择的 qv-block 不适合 Qwen3。

**可能根因**：
1. Qwen3（40 层）vs Qwen2.5（48 层）的 layer depth 分布不同，`depth_balanced 1:1:2` 配比可能不适合
2. Qwen3 的 attention pattern 不同，Q/V block 的 signal/noise 估计不准确
3. SN statistics estimation 可能在 Qwen3 上需要调整（sketch dimension、norm estimation）

## 下一步

1. 重跑完整 15 job（flasc/compeft 用修复后代码）→ 更新主表
2. 诊断 Ours 的 qv-block 选择分布 → 调整 depth ratio 或 SN params
3. Ours 可能需要单独的 hyperparameter tuning
