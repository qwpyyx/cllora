import torch
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
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

    def __init__(self, model, state, lr, alpha,
                 cid, radius, beta, sigma, tau, lam, comm_budget, layer_costs, use_history: bool = True,):
        self.model = model
        self.state = state
        self.lr = lr
        self.alpha = alpha
        self.cid = cid
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
        self.logger = logging.getLogger("federated_training")  # 与run_federated_training共用同一个logger
        self.packet_size = 1500
        self.use_history = use_history



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

    def client_update(self, loader: torch.utils.data.DataLoader, epochs: int = 1) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.model.train()
        stats = self._init_layer_stats()
        initial_params = {n: p.data.clone() for n, p in self._lora_params}
        total_steps = 0
        device = next(self.model.parameters()).device

        total_batches = epochs * len(loader)
        # 客户端训练开始日志（复用全局logger，带客户端ID）
        self.logger.info(f"[客户端 {self.cid}] 开始本地训练：{epochs}个epoch，共{total_batches}个batch")

        for epoch in range(epochs):
            self.logger.info(f"[客户端 {self.cid}] ====== Epoch {epoch + 1}/{epochs} ======")
            epoch_pbar = tqdm(loader, desc=f"[客户端 {self.cid}] Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(epoch_pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                total_steps += 1

                # 前向传播与损失计算
                loss = self.model(**batch).loss
                loss.backward()

                # 1) 统计每个batch中 A/B 的梯度总量
                if batch_idx in (0, 1, 2):  # 只看前几个batch
                    sumA = sum(float(p.grad.detach().abs().sum())
                               for n, p in self._lora_params if "lora_A" in n)
                    sumB = sum(float(p.grad.detach().abs().sum())
                               for n, p in self._lora_params if "lora_B" in n)
                    print(f"[客户端 {self.cid}] batch={batch_idx} | |g|_A={sumA:.3e} | |g|_B={sumB:.3e}")

                # 2) 验证我们确实在更新权重（更新前后范数是否变化）
                if batch_idx in (0, 1):
                    for n, p in self._lora_params:
                        if "lora_B" in n:
                            print(f"[{batch_idx}] before step ||{n}||={p.data.norm().item():.3e}")
                    # ……执行 p.data -= eta * v_B 之后……
                    for n, p in self._lora_params:
                        if "lora_B" in n:
                            print(f"[{batch_idx}]  after step ||{n}||={p.data.norm().item():.3e}")

                # 3) 注意你每个参数更新后都在本地清梯度
                # p.grad.zero_() 已存在：见源码 L210

                if epoch == 0 and batch_idx == 0:
                    with torch.no_grad():
                        valid_label_tokens = (batch["labels"] != -100).sum().item() if "labels" in batch else -1
                    print(
                        f"[客户端 {self.cid}] 首个batch: loss={loss.item():.6f}, 有效label数={valid_label_tokens}")


                # 记录每10个batch的损失（避免日志过多）
                if (batch_idx + 1) % 2 == 0:
                    self.logger.info(
                        f"[客户端 {self.cid}] Epoch {epoch + 1} Batch {batch_idx + 1}/{len(loader)} | 损失: {loss.item():.4f}")

                all_zero = True
                for _n, _p in self._lora_params:
                    if _p.grad is not None and _p.grad.detach().abs().sum() > 0:
                        all_zero = False
                        break
                if all_zero:
                    print(
                        f"[客户端 {self.cid}] 警告：本batch所有LoRA梯度均为0。极可能是labels被全mask或LoRA未参与前向。")

                # 遍历LoRA层处理
                for name, p in self._lora_params:
                    # 获取当前batch的梯度
                    g = p.grad.detach()
                    st = stats[name]

                    # 在线近似fisher
                    F_batch = g * g
                    st["F_curr"] = self.alpha * st["F_curr"] + (1 - self.alpha) * F_batch

                    if not self.use_history:
                        p.data -= self.lr * g
                        p.grad.zero_()
                        continue

                    # step 2: 计算关键参数
                    v_B = g / (st["F_curr"] + 1e-12)
                    bar_F = self.state.bar_F[name]
                    theta_old = self.state.bar_theta(name)
                    a_B = (v_B * bar_F * v_B).sum()
                    b_B = (v_B * bar_F * (p.data - theta_old)).sum()
                    delta = self.radius ** 2 - st["r2"]

                    # step 2: 判断学习率（冲突层/非冲突层）
                    is_conflict = b_B.item() < 0
                    if is_conflict:
                        # 冲突层逻辑
                        num = b_B - self.sigma
                        truncated_diff = torch.clamp(delta - self.tau, min=0.0)
                        radicand = (b_B - self.sigma) ** 2 + self.beta * a_B * truncated_diff
                        eta = (num + torch.sqrt(radicand)) / (self.beta * a_B + 1e-12)
                        eta = torch.minimum(eta, torch.tensor(self.lr, device=eta.device))
                        st["conf"] += 1
                    else:
                        # 非冲突层逻辑
                        eta_0 = torch.tensor(self.lr, device=p.data.device)
                        # max_delta = torch.clamp(delta, min=0.0)
                        # sqrt_term = torch.sqrt(b_B ** 2 + a_B * max_delta)
                        # eta_max = (b_B + sqrt_term) / (a_B + 1e-12)
                        # eta = torch.minimum(eta_0, eta_max)
                        eta = eta_0

                    # 记录冲突层信息（每20个batch输出一次，避免刷屏）
                    if is_conflict and (total_steps % 20 == 0):
                        self.logger.info(
                            f"[客户端 {self.cid}] 冲突层检测 | 层: {name} | "
                            f"b_B: {b_B.item():.4f} | 学习率: {eta.item():.6f} | "
                            f"累计冲突次数: {st['conf']}"
                        )

                    # 更新统计量
                    Q_B = (g * g / (st["F_curr"] + 1e-12)).sum()
                    st["B_round"] += torch.maximum(torch.tensor(0.0), (eta - 0.5 * eta ** 2) * Q_B)
                    st["r2"] = st["r2"] - 2 * eta * b_B + (eta ** 2) * a_B

                    # 参数更新
                    p.data -= eta * v_B
                    p.grad.zero_()

                # 更新进度条信息
                epoch_pbar.set_postfix({
                    "损失": f"{loss.item():.4f}",
                    "总冲突次数": sum(st["conf"] for st in stats.values())
                })

            # Epoch结束日志
            self.logger.info(
                f"[客户端 {self.cid}] Epoch {epoch + 1} 结束 | "
                f"最终损失: {loss.item():.4f} | "
                f"累计冲突层次数: {sum(st['conf'] for st in stats.values())}"
            )

        if not self.use_history:
            mu_full = {n: initial_params[n] - p.data for n, p in self._lora_params}
            return mu_full, stats

        # 训练结束后：计算上传层并记录日志
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

        # 记录背包问题输入
        self.logger.info(
            f"[客户端 {self.cid}] 层选择开始 | 通信预算: {self.comm_budget} 包 | "
            f"总LoRA层数: {len(self._lora_params)}"
        )

        chosen_idx = knapsack_select(values, costs, self.comm_budget)
        for idx, (name, _) in enumerate(self._lora_params):
            if idx in chosen_idx:
                selected_layers.append(name)
            else:
                mu_full[name].zero_()

        # 记录选中的层
        self.logger.info(
            f"[客户端 {self.cid}] 层选择完成 | 选中层数: {len(selected_layers)} | "
            f"总消耗数据包: {sum(costs[idx] for idx in chosen_idx)} | "
            f"选中层: {[name.split('.')[-2] for name in selected_layers]}"  # 简化层名称显示
        )

        return mu_full, stats

    def accumulate_fisher(self, fisher: Dict[str, torch.Tensor]) -> None:
        """Proxy to accumulate Fisher info for the current round."""
        self.state.accumulate_fisher(fisher)

    def finalize_task(
        self, theta_star: Dict[str, torch.Tensor], num_rounds: int, gamma: float = 1.0
    ) -> None:
        """Proxy to update state statistics once a task finishes."""
        self.state.finalize_task(theta_star, num_rounds, gamma)

