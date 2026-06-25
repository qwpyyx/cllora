# Phase 1 修复 #3 — SN-P1/P2 根因分析

## 根因：`depth_balanced` 归一化过度放大组内差异

SN-P1/P2 使用 `_sn_rank_normalize` 在 depth group 内部做 min-max 归一化。

Qwen3 的 a_hat（shared signal）在所有 qv-block 上非常接近：
- a_hat 范围：~0.01
- b_hat 范围：~0.001-0.002
- SNR = a/b ≈ 5-10

`_sn_rank_normalize` 将 a_hat 按 depth group 内排序后归一化到 [0,1]。
组内最低的 a_hat → 0 → 边际收益 Δ < 0 → quota = 0。

结果：仅 17/40 模块有正 quota。

## 修复方案

改用 `p1_norm_mode=raw`（不归一化），直接使用原始 a_hat, b_hat。

```bash
--sn_p1_norm_mode raw
```

由于 a_hat 都在 0.01 附近且 SNR 均 > 5，Δ_m 应为正值，所有 40 个 qv-block 都应获得 quota。

## 预期

- active_units 从 17 → 40
- EM 从 18 回复到 30+
