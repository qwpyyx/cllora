import torch
from typing import Dict, List, Tuple


class ClientState:
    """Holds running estimates for Fisher information and parameter means."""

    def __init__(self):
        self.bar_F: Dict[str, torch.Tensor] = {}
        self.bar_B: Dict[str, torch.Tensor] = {}
        self._fisher_sum: Dict[str, torch.Tensor] = {}

    def bar_theta(self, name: str) -> torch.Tensor:
        F = self.bar_F.get(name)
        B = self.bar_B.get(name)
        if F is None or B is None:
            return None
        return B / (F + 1e-12)

    def accumulate_fisher(self, fisher: Dict[str, torch.Tensor]) -> None:
        """Add the current round's Fisher estimates to the running task sum."""
        for name, F in fisher.items():
            if name not in self._fisher_sum:
                self._fisher_sum[name] = F.clone()
            else:
                self._fisher_sum[name] += F

    def finalize_task(
        self,
        theta_star: Dict[str, torch.Tensor],
        num_rounds: int,
        gamma: float = 1.0,
    ) -> None:
        """Update historical statistics after finishing a task.

        Args:
            theta_star: Final model parameters for each LoRA layer.
            num_rounds: Total number of global rounds in the task.
            gamma: Decay factor for old task importance.
        """
        if num_rounds <= 0:
            return
        for name, theta in theta_star.items():
            F_sum = self._fisher_sum.get(name, torch.zeros_like(theta))
            F_avg = F_sum / num_rounds
            self.bar_F[name] = gamma * self.bar_F.get(name, torch.zeros_like(F_avg)) + F_avg
            B = F_avg * theta
            self.bar_B[name] = gamma * self.bar_B.get(name, torch.zeros_like(B)) + B
        self._fisher_sum.clear()

def knapsack_select(values: List[float], costs: List[int], budget: int) -> List[int]:
    """Dynamic programming 0-1 knapsack solver.

    Args:
        values: Utility for each item.
        costs: Integer cost for each item.
        budget: Total budget.
    Returns:
        Indices of selected items.
    """
    n = len(values)
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]
    keep = [[False] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost = costs[i - 1]
        val = values[i - 1]
        for b in range(budget + 1):
            if cost <= b and dp[i - 1][b - cost] + val > dp[i - 1][b]:
                dp[i][b] = dp[i - 1][b - cost] + val
                keep[i][b] = True
            else:
                dp[i][b] = dp[i - 1][b]
    b = budget
    chosen: List[int] = []
    for i in range(n, 0, -1):
        if keep[i][b]:
            chosen.append(i - 1)
            b -= costs[i - 1]
    return chosen[::-1]


class ContinualFisherClient:
    """Client implementing Fisher-based continual FL with communication limits.

    This class follows the algorithmic description supplied by the user. It
    operates on LoRA parameters of the provided model and performs adaptive
    learning-rate updates along with a knapsack-based layer selection strategy
    to respect a communication budget.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        state: ClientState,
        lr: float = 1e-3,
        alpha: float = 0.9,
        radius: float = 1.0,
        beta: float = 1.0,
        sigma: float = 0.0,
        tau: float = 0.0,
        lam: float = 1.0,
        comm_budget: int = 1,
        layer_costs: Dict[str, int] | None = None,
    ) -> None:
        self.model = model
        self.state = state
        self.lr = lr
        self.alpha = alpha
        self.radius = radius
        self.beta = beta
        self.sigma = sigma
        self.tau = tau
        self.lam = lam
        self.comm_budget = comm_budget
        self.layer_costs = layer_costs or {}
        self._lora_params = [
            (n, p) for n, p in model.named_parameters() if "lora" in n and p.requires_grad
        ]

    def _init_layer_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        stats: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, p in self._lora_params:
            F_old = self.state.bar_F.get(name, torch.zeros_like(p.data))
            self.state.bar_F.setdefault(name, F_old.clone())
            self.state.bar_B.setdefault(name, torch.zeros_like(p.data))
            theta_old = self.state.bar_theta(name)
            if theta_old is None:
                theta_old = torch.zeros_like(p.data)
            r2 = ((p.data - theta_old) ** 2 * F_old).sum()
            stats[name] = {
                "F_curr": torch.zeros_like(p.data),
                "r2": r2,
                "r2_start": r2.clone() if torch.is_tensor(r2) else torch.tensor(r2),
                "B_round": torch.tensor(0.0),
                "conf": 0.0,
            }
        return stats

    def client_update(self, loader: torch.utils.data.DataLoader, epochs: int = 1) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.model.train()
        stats = self._init_layer_stats()
        initial_params = {n: p.data.clone() for n, p in self._lora_params}
        total_steps = 0
        device = next(self.model.parameters()).device
        for _ in range(epochs):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                total_steps += 1
                loss = self.model(**batch).loss
                loss.backward()
                for name, p in self._lora_params:
                    # 获取当前batch的梯度
                    g = p.grad.detach()
                    st = stats[name]
                    # 在线近似fisher
                    F_batch = g * g
                    st["F_curr"] = self.alpha * st["F_curr"] + (1 - self.alpha) * F_batch
                    # step 2
                    v_B = g / (st["F_curr"] + 1e-12)
                    bar_F = self.state.bar_F[name]
                    theta_old = self.state.bar_theta(name)
                    a_B = (v_B * bar_F * v_B).sum()
                    b_B = (v_B * bar_F * (p.data - theta_old)).sum()
                    delta = self.radius ** 2 - st["r2"]
                    # step 2判断学习率
                    if b_B.item() < 0:
                        num = b_B - self.sigma
                        truncated_diff = torch.clamp(delta - self.tau, min=0.0)
                        radicand = (b_B - self.sigma) ** 2 + self.beta * a_B * truncated_diff
                        eta = (num + torch.sqrt(radicand)) / (self.beta * a_B + 1e-12)
                        eta = torch.minimum(eta, torch.tensor(self.lr, device=eta.device))
                        st["conf"] += 1
                    else:
                        # 非冲突分支：新增“可行硬帽”（上根）逻辑
                        eta_0 = torch.tensor(self.lr, device=p.data.device)  # 原来的基础步长 η₀

                        # 计算 max{0, Δ}，Δ = self.radius² - st["r2"]
                        max_delta = torch.clamp(delta, min=0.0)
                        # 计算 η_max 公式：[b + sqrt(b² + a * max{0, Δ})] / a
                        sqrt_term = torch.sqrt(b_B ** 2 + a_B * max_delta)
                        eta_max = (b_B + sqrt_term) / (a_B + 1e-12)  # 分母加小量防除零

                        # 取 η = min{η₀, η_max}，保证步长不越界
                        eta = torch.minimum(eta_0, eta_max)
                    Q_B = (g * g / (st["F_curr"] + 1e-12)).sum()
                    st["B_round"] += torch.maximum(torch.tensor(0.0), (eta - 0.5 * eta ** 2) * Q_B)
                    st["r2"] = st["r2"] - 2 * eta * b_B + (eta ** 2) * a_B
                    p.data -= eta * v_B
                    p.grad.zero_()
        F_round = {}
        values = []
        costs = []
        selected_layers: List[str] = []
        mu_full = {n: initial_params[n] - p.data for n, p in self._lora_params}
        for name, p in self._lora_params:
            st = stats[name]
            F_round[name] = torch.maximum(torch.tensor(0.0), 0.5 * (st["r2"] - st["r2_start"]))
            p_round = st["conf"] / max(total_steps, 1)
            value = st["B_round"] - self.lam * p_round * F_round[name]
            values.append(value.item())
            costs.append(int(self.layer_costs.get(name, 1)))
        chosen_idx = knapsack_select(values, costs, self.comm_budget)
        for idx, (name, _) in enumerate(self._lora_params):
            if idx in chosen_idx:
                selected_layers.append(name)
            else:
                mu_full[name].zero_()
        return mu_full, stats

    def accumulate_fisher(self, fisher: Dict[str, torch.Tensor]) -> None:
        """Proxy to accumulate Fisher info for the current round."""
        self.state.accumulate_fisher(fisher)

    def finalize_task(
        self, theta_star: Dict[str, torch.Tensor], num_rounds: int, gamma: float = 1.0
    ) -> None:
        """Proxy to update state statistics once a task finishes."""
        self.state.finalize_task(theta_star, num_rounds, gamma)

