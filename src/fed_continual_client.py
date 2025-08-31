from typing import Dict, List, Tuple
import torch
from uie_trainer_lora import UIETrainer
from fed_continual_state import ContinualState


def _knapsack(values: List[float], costs: List[int], budget: int) -> List[bool]:
    """0/1 knapsack dynamic programming."""
    n = len(values)
    dp = [[0]*(budget+1) for _ in range(n+1)]
    for i in range(1, n+1):
        v, c = values[i-1], costs[i-1]
        for b in range(budget+1):
            dp[i][b] = dp[i-1][b]
            if c <= b:
                dp[i][b] = max(dp[i][b], dp[i-1][b-c] + v)
    sel = [False]*n
    b = budget
    for i in range(n,0,-1):
        if dp[i][b] != dp[i-1][b]:
            sel[i-1] = True
            b -= costs[i-1]
    return sel


def compute_fisher(model, dataloader, alpha: float = 0.9, engine=None):
    """Compute EMA-based diagonal Fisher information for model parameters."""
    device = next(model.parameters()).device
    F_curr = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    model.train()
    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        if engine is not None:
            engine.zero_grad()
        else:
            model.zero_grad()
        outputs = model(**batch)
        loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs['loss']
        if engine is not None:
            engine.backward(loss)
        else:
            loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                F_batch = param.grad * param.grad
                F_curr[name] = alpha * F_curr[name] + (1 - alpha) * F_batch
    return {k: v.detach().cpu() for k, v in F_curr.items()}


class ContinualTrainer(UIETrainer):
    """Trainer implementing continual federated algorithm."""
    def __init__(self, *args, state: ContinualState, radius: float=1.0, sigma: float=0.0,
                 tau: float=0.0, beta: float=1.0, alpha: float=0.9,
                 comm_budget: int=0, layer_costs: Dict[str,int]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.radius = radius
        self.sigma = sigma
        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.comm_budget = comm_budget
        self.layer_costs = layer_costs or {}

    def train(self, base_params: Dict[str, torch.Tensor], **kwargs) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:  # type: ignore[override]
        model = self.model
        dataloader = self.get_train_dataloader()
        self.create_optimizer_and_scheduler(num_training_steps=len(dataloader) * int(self.args.num_train_epochs))
        device = next(model.parameters()).device
        model.train()

        F_curr = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
        bar_F = self.state.bar_F
        bar_B = self.state.bar_B
        bar_theta = {k: (bar_B[k] / bar_F[k]) if k in bar_F else torch.zeros_like(p, device=device)
                     for k, p in model.named_parameters() if p.requires_grad}
        r2 = {}
        r2_start = {}
        B_round = {}
        conf = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            f = bar_F.get(name, torch.zeros_like(p, device=device))
            r2[name] = ((p - bar_theta.get(name, torch.zeros_like(p))).pow(2) * f).sum()
            r2_start[name] = r2[name].clone()
            B_round[name] = torch.tensor(0., device=device)
            conf[name] = 0

        for epoch in range(int(self.args.num_train_epochs)):
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if self.deepspeed:
                    self.deepspeed.zero_grad()
                else:
                    self.optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs['loss']
                if self.deepspeed:
                    self.deepspeed.backward(loss)
                else:
                    loss.backward()
                for name, p in model.named_parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    g = p.grad
                    F_batch = g * g
                    F_curr[name] = self.alpha * F_curr[name] + (1 - self.alpha) * F_batch
                    v = g / (F_curr[name] + 1e-8)
                    a = (v * bar_F.get(name, 0)).view(-1).dot(v.view(-1)) if name in bar_F else torch.tensor(0.,
                                                                                                             device=device)
                    diff = (p - bar_theta.get(name, 0))
                    b_val = (v * bar_F.get(name, 0) * diff).sum() if name in bar_F else torch.tensor(0., device=device)
                    delta = self.radius ** 2 - r2[name]
                    if b_val.item() < 0:
                        eta = (b_val - self.sigma)
                        disc = (b_val - self.sigma) ** 2 + self.beta * a * (delta - self.tau)
                        disc = torch.clamp(disc, min=0)
                        eta = (eta + torch.sqrt(disc)) / (self.beta * a + 1e-8)
                        eta = min(eta.item(), self.args.learning_rate)
                        conf[name] += 1
                    else:
                        eta = self.args.learning_rate
                    Q = (g * g / (F_curr[name] + 1e-8)).sum()
                    B_round[name] = B_round[name] + max(0, (eta - 0.5 * eta ** 2) * Q)
                    r2[name] = r2[name] - 2 * eta * b_val + (eta ** 2) * a
                    p.data = p.data - eta * v
                if self.deepspeed:
                    self.deepspeed.step()
                else:
                    self.optimizer.step()

        F_round = {n: torch.clamp(0.5 * (r2[n] - r2_start[n]), min=0) for n in r2}
        p_round = {n: conf[n] / max(len(dataloader), 1) for n in conf}
        values, costs, names = [], [], []
        for name in B_round:
            values.append((B_round[name] - p_round[name] * F_round[name]).item())
            costs.append(self.layer_costs.get(name, 1))
            names.append(name)
        selected = _knapsack(values, costs, self.comm_budget)
        delta = {}
        state_dict = model.state_dict()
        for name, sel in zip(names, selected):
            local_param = state_dict[name].detach().cpu()
            delta[name] = (base_params[name] - local_param) if sel else torch.zeros_like(local_param)
        return delta, {k: v.detach().cpu() for k, v in F_curr.items()}