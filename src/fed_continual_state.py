import os
from dataclasses import dataclass, field
from typing import Dict, List
import torch
import logging
@dataclass
class ContinualState:
    """State object to store Fisher information and optimal parameters for past tasks."""
    bar_F: Dict[str, torch.Tensor] = field(default_factory=dict)  # 历史Fisher（已归一化）
    bar_B: Dict[str, torch.Tensor] = field(default_factory=dict)
    theta_last: Dict[str, torch.Tensor] = field(default_factory=dict)# 历史Fisher-参数乘积（已归一化）
    gamma: float = 1.0
    eta_history: Dict[str, List[float]] = field(default_factory=dict)
    # 归一化配置（类内统一管理，避免硬编码）
    norm_min_mean: float = 1e-6  # Fisher均值小于此值时才归一化
    norm_max_scale: float = 100.0  # 最大放大倍数，避免极端值


    def update(self, F_task: Dict[str, torch.Tensor], theta_star: Dict[str, torch.Tensor]) -> None:
        """
        无归一化的更新：直接用原始F_task和F_task*theta_star累积到bar_F和bar_B
        Args:
            F_task: 当前任务的Fisher（原始未归一化）
            theta_star: 当前任务的最优参数（原始未归一化）
        """
        # 1. 生成当前任务的B_task = F_task ⊙ theta_star（逐元素乘，无归一化）
        B_task = {}
        for name in F_task:
            if name not in theta_star:
                raise ValueError(f"F_task包含层{name}，但theta_star中无该层，无法计算B_task")
            B_task[name] = F_task[name] * theta_star[name]  # 原始未归一化的B_task

        # 2. 直接累积更新到bar_F和bar_B（无归一化步骤）
        for name in F_task:
            F_raw = F_task[name]
            B_raw = B_task[name]
            if name not in self.bar_F:
                # 新层：直接用原始数据初始化
                self.bar_F[name] = F_raw.clone()
                self.bar_B[name] = B_raw.clone()
            else:
                # 旧层：指数累积（gamma为历史权重，保留原始量级）
                self.bar_F[name] = self.gamma * self.bar_F[name] + F_raw
                self.bar_B[name] = self.gamma * self.bar_B[name] + B_raw

        self.theta_last = theta_star

    def get_f_past(self, layer_name: str) -> torch.Tensor:
        """
        外部获取历史Fisher（bar_F），返回原始未归一化数据
        Args:
            layer_name: 层名
        Returns:
            原始未归一化的历史Fisher（bar_F）
        """
        if layer_name not in self.bar_F:
            raise KeyError(f"层{layer_name}不在bar_F中，无历史Fisher数据")
        return self.bar_F[layer_name].clone()



    def _normalize_single_layer(self, F: torch.Tensor, B: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单一层的Fisher和关联B的同步归一化（核心工具方法）
        Args:
            F: 当前层的Fisher（如F_task的某一层）
            B: 当前层的Fisher-参数乘积（如F_task*theta_star，可选）
        Returns:
            归一化后的F和B（若B为None，仅返回归一化后的F）
        """
        f_mean = torch.mean(F)
        # 仅当Fisher均值过小且为正时，才进行归一化（避免无意义操作）
        if f_mean > 0 and f_mean < self.norm_min_mean:
            # scale = min(1.0 / f_mean, self.norm_max_scale)  # 限制最大放大倍数
            scale = 1.0 / f_mean
            F_norm = F * scale
            B_norm = B * scale if B is not None else None  # B与F同比例放大
        else:
            F_norm = F.clone()
            B_norm = B.clone() if B is not None else None
        return F_norm, B_norm

    def update_with_normalization(self, F_task: Dict[str, torch.Tensor], theta_star: Dict[str, torch.Tensor]) -> None:
        """
        带归一化的更新：先归一化F_task和F_task*theta_star，再累积到bar_F和bar_B
        （外部代码只需调用此方法，无需关心归一化细节）
        Args:
            F_task: 当前任务的Fisher（未归一化）
            theta_star: 当前任务的最优参数（未归一化）
        """
        # 1. 生成当前任务的B_task = F_task ⊙ theta_star（逐元素乘）
        B_task = {}
        for name in F_task:
            if name not in theta_star:
                raise ValueError(f"F_task包含层{name}，但theta_star中无该层，无法计算B_task")
            B_task[name] = F_task[name] * theta_star[name]  # 未归一化的B_task

        # 2. 对每一层的F_task和B_task同步归一化
        F_task_norm = {}
        B_task_norm = {}
        for name in F_task:
            F_norm, B_norm = self._normalize_single_layer(F_task[name], B_task[name])
            F_task_norm[name] = F_norm
            B_task_norm[name] = B_norm

        # 3. 累积更新到bar_F和bar_B（与原update逻辑一致，但用归一化后的数据）
        for name in F_task_norm:
            F_norm = F_task_norm[name]
            B_norm = B_task_norm[name]
            if name not in self.bar_F:
                # 新层：直接用归一化后的数据初始化
                self.bar_F[name] = F_norm
                self.bar_B[name] = B_norm
            else:
                # 旧层：指数累积（gamma为历史权重）
                self.bar_F[name] = self.gamma * self.bar_F[name] + F_norm
                self.bar_B[name] = self.gamma * self.bar_B[name] + B_norm

        self.theta_last = theta_star



    def get_normalized_f_past(self, layer_name: str) -> torch.Tensor:
        """
        外部获取归一化后的f_past（即bar_F），确保训练时用的是已归一化的数据
        Args:
            layer_name: 层名
        Returns:
            归一化后的f_past（bar_F）
        """
        if layer_name not in self.bar_F:
            raise KeyError(f"层{layer_name}不在bar_F中，无历史Fisher数据")
        # bar_F已在update时归一化，直接返回
        return self.bar_F[layer_name].clone()

    # ------------------------------ 原有方法保留（无需修改）------------------------------
    def save(self, path: str) -> None:
        torch.save({'bar_F': self.bar_F, 'bar_B': self.bar_B, 'eta_history': self.eta_history, "theta_last":self.theta_last}, path)

    @classmethod
    def load(cls, path: str) -> "ContinualState":
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
            state = cls()
            state.bar_F = data.get('bar_F', {})
            state.bar_B = data.get('bar_B', {})
            state.theta_last = data.get('theta_last', {})
            # state.eta_history = data.get('eta_history', {})
            return state
        return cls()

    def has_valid_history(self):
        return hasattr(self, 'bar_F') and self.bar_F is not None and len(self.bar_F) > 0

    def append_eta_history(self, eta_batch: Dict[str, float]) -> None:
        for layer_name, eta_val in eta_batch.items():
            if layer_name not in self.eta_history:
                self.eta_history[layer_name] = []
            self.eta_history[layer_name].append(eta_val)