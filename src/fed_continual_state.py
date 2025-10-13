import os
from dataclasses import dataclass, field
from typing import Dict, List
import torch

@dataclass
class ContinualState:
    """State object to store Fisher information and optimal parameters for past tasks."""
    bar_F: Dict[str, torch.Tensor] = field(default_factory=dict)
    bar_B: Dict[str, torch.Tensor] = field(default_factory=dict)
    gamma: float = 1.0

    def save(self, path: str) -> None:
        torch.save({'bar_F': self.bar_F, 'bar_B': self.bar_B}, path)

    def update(self, F_task: Dict[str, torch.Tensor], theta_star: Dict[str, torch.Tensor], gamma: float=1.0) -> None:
        for name, F in F_task.items():
            if name not in self.bar_F:
                self.bar_F[name] = F.clone()
                self.bar_B[name] = (F*theta_star[name]).clone()
            else:
                self.bar_F[name] = gamma*self.bar_F[name] + F
                self.bar_B[name] = gamma*self.bar_B[name] + F*theta_star[name]

    @classmethod
    def load(cls, path: str) -> "ContinualState":
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
            state = cls()
            state.bar_F = data.get('bar_F', {})
            state.bar_B = data.get('bar_B', {})
            return state
        return cls()

    def has_valid_history(self):
        """判断是否包含有效的历史信息（如 Fisher 矩阵）"""
        return hasattr(self, 'bar_F') and self.bar_F is not None and len(self.bar_F) > 0

