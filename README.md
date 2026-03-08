# RieSelect

The theory analyzes an idealized RieSelect update along the Fisher-preconditioned direction. In implementation, we adopt AdamW for numerical stability in federated LoRA training, while preserving the Fisher-based conflict detection, adaptive step-size gating, and budget-constrained layer selection. Therefore, the implementation should be viewed as an engineering instantiation of RieSelect rather than a theorem-level line-by-line equivalent optimizer.

