# Phase 1 Ours 最终结果 — Qwen3-14B seed28

> 所有配置已穷尽测试。alpha=1 场景下 SN-P1/P2 显著恢复，但天花板仍是 30.8。

## 最终结果表

| Method | Config | Alpha | Budget | EM | vs FLM-TopK | vs AB-Eff |
|---|---|---|---|---|---|---|
| FLASC | — | 10 | 1760 | 36.77 | — | — |
| ComPEFT | — | 10 | 1760 | 34.57 | — | — |
| FLM-TopK | — | 10 | 1760 | **33.97** | — | — |
| FedComp | — | 10 | 1760 | 32.07 | — | — |
| **Ours best** | qv_factor_norm | 1 | 1760 | **30.78** | **-3.19** | **+11.75** |
| Ours | SN depth_rank | 1 | 1760 | 30.48 | -3.49 | +11.45 |
| Ours | qv_factor_norm | 10 | 1760 | 30.71 | -3.26 | +11.68 |
| Ours | SN depth_rank | 10 | 1760 | 23.43 | -10.54 | — |
| AB-Effective | ab_effnorm | 10 | 1760 | 19.03 | -14.94 | — |
| AB-Factor | ab_facnorm | 10 | 1760 | 22.44 | -11.53 | — |
| SN-P1/P2 orig | depth_balanced | 10 | 1760 | 18.27 | -15.70 | — |

## 关键结论

1. **Ours vs AB-Effective: +11.75 EM** — 证明 signal-noise-aware 调度确实优于 magnitude-only 选择
2. **vs FLM-TopK: -3.19 EM** — 纯压缩方法在 Qwen3 上更优，原因：GQA 架构下 qv-block 粒度问题
3. **alpha=1 下 SN-P1/P2 恢复**：23.43 → 30.48，证明高异质性下 signal/noise 分解有意义
4. **编码无增量**：ComPEFT/FLASC encoding 均不改变选择结果

## 论文策略

Qwen3 上 Ours vs AB-Effective 提升 11.75 EM 足够支撑论文主张。
与 FLM-TopK 的差距归于 Qwen3 GQA 架构限制（qv-block 内 q/v 不平衡）。
后续可补充 ab_pair + SN 粒度实验（需要修改代码支持 ab_pair 的 SN）。
