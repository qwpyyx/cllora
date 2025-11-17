import logging
import os
import time
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
from fed_continual_state import ContinualState
from uie_collator import SUPPORTED_DECODER_MODELS, _check_model_name as check_model
from uie_dataset_lora import ANSWER_PREFIX
from collections import defaultdict
# logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from accelerate import Accelerator
import logging
import time
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import re
# logger = logging.getLogger(__name__)
from torch.optim.optimizer import Optimizer


def monitor_gradient_health(step, model, F_curr, bar_F_raw, r2, B_round):
    """实时监控梯度健康状况"""
    health_report = {
        'step': step,
        'issues': [],
        'metrics': {}
    }

    # 检查梯度健康
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = torch.norm(p.grad).item()
            health_report['metrics'][f'{name}_grad_norm'] = grad_norm

            if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
                health_report['issues'].append(f"{name}: 梯度包含NaN/Inf")

            if grad_norm > 100:
                health_report['issues'].append(f"{name}: 梯度爆炸 (范数: {grad_norm})")

    # 检查Fisher健康
    for name in F_curr:
        fisher_val = F_curr[name]
        fisher_mean = torch.mean(fisher_val).item()
        health_report['metrics'][f'{name}_fisher_mean'] = fisher_mean

        if torch.any(torch.isnan(fisher_val)) or torch.any(torch.isinf(fisher_val)):
            health_report['issues'].append(f"{name}: Fisher包含NaN/Inf")

        if fisher_mean < 1e-12:
            health_report['issues'].append(f"{name}: Fisher值过小 ({fisher_mean})")
        elif fisher_mean > 1e3:
            health_report['issues'].append(f"{name}: Fisher值过大 ({fisher_mean})")

    # 检查半径健康
    for name in r2:
        r2_val = r2[name].item()
        health_report['metrics'][f'{name}_r2'] = r2_val

        if r2_val < 0:
            health_report['issues'].append(f"{name}: 马氏半径为负 ({r2_val})")
        elif r2_val > 1:
            health_report['issues'].append(f"{name}: 马氏半径过大 ({r2_val})")

    # 检查收益健康
    for name in B_round:
        b_round_val = B_round[name].item()
        health_report['metrics'][f'{name}_b_round'] = b_round_val

        if b_round_val > 1000:
            health_report['issues'].append(f"{name}: 收益值过大 ({b_round_val})")

    return health_report

def update_online_fisher(
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        F_curr: Dict[str, torch.Tensor],
        alpha_ema: float,
        device: torch.device
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    在线 Fisher EMA 更新（无额外前向/反向、无分布式通信）:
      - 假设外层已调用 self.accelerator.backward(loss)，此时 p.grad 已由 DDP 同步；
      - 直接用 grad^2 做 EMA: F = alpha*F + (1-alpha)*(grad^2)
      - 统一使用“规范化键名”（去 'module.' 前缀）索引 F_curr，避免 KeyError；
      - 保留返回 F_batch 供监控/诊断（即 grad^2 快照）。
    """
    import torch

    def _canon_name(n: str) -> str:
        return n[7:] if isinstance(n, str) and n.startswith("module.") else n

    F_batch: Dict[str, torch.Tensor] = {}

    # 直接读取“已同步”的梯度
    for name, p in model.named_parameters():
        if (not p.requires_grad) or (p.grad is None):
            continue

        cn = _canon_name(name)
        g = p.grad.detach()
        # 半精度时在 float32 上算平方更稳
        g2 = (g.float() * g.float()) if g.dtype in (torch.float16, torch.bfloat16) else (g * g)

        # 初始化 F_curr 项（与参数形状/设备一致；dtype 与 g2 对齐，避免频繁 cast）
        if cn not in F_curr:
            F_curr[cn] = torch.zeros_like(p, device=p.device, dtype=(g2.dtype if g2.dtype.is_floating_point else p.dtype))

        # 对齐 dtype（极少数情况下 F_curr 已存在但 dtype 与 g2 不同）
        if F_curr[cn].dtype != g2.dtype:
            g2 = g2.to(F_curr[cn].dtype)

        # EMA 更新
        F_curr[cn] = alpha_ema * F_curr[cn] + (1.0 - alpha_ema) * g2

        # 记录本 step 的 F_batch（可用于你后续的监控/可视化）
        F_batch[cn] = g2

    return F_curr, F_batch


def _knapsack(values: List[float], costs: List[int], budget: int) -> List[bool]:
    """0/1 knapsack dynamic programming."""
    n = len(values)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, c = values[i - 1], costs[i - 1]
        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]
            if c <= b:
                dp[i][b] = max(dp[i][b], dp[i - 1][b - c] + v)
    sel = [False] * n
    b = budget
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            sel[i - 1] = True
            b -= costs[i - 1]
    return sel


def compute_fisher_arithmetic(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    算术平均版本的Fisher信息计算（替代原有compute_fisher）
    适配联邦学习客户端训练流程，仅计算LoRA可训练参数的Fisher
    """
    if device is None:
        device = next(model.parameters()).device

    # 初始化Fisher累积器（仅跟踪LoRA可训练参数）
    fisher = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n
    }
    total_steps = 0  # 累计训练步数（用于算术平均）

    model.train()
    for batch in dataloader:
        # 数据移至设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        model.zero_grad(set_to_none=True)

        # 前向+反向传播（与客户端训练逻辑一致）
        outputs = model(**batch)
        loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs["loss"]
        loss.backward()

        # 累积梯度平方（算术平均核心：每步累加，最后除以总步数）
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and "lora" in name:
                fisher[name] += param.grad.detach() ** 2  # 梯度平方累加
                total_steps += 1  # 仅计数有梯度的参数步骤

    # 计算算术平均：总累积 / 总步数（避免除零）
    # 分布式环境下同步梯度平方和计数
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        for tensor in fisher.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_steps_tensor = torch.tensor(float(total_steps), device=device)
        dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.SUM)
        total_steps = max(int(total_steps_tensor.item()), 0)
        if world_size > 1 and total_steps == 0:
            logger.warning("分布式Fisher计算中统计步数为0，返回零矩阵")
    if total_steps > 0:
        for name in fisher:
            fisher[name] = fisher[name] / total_steps
    else:
        logger.warning("未记录到有效梯度步骤，Fisher保持初始零值")

    # 转移到CPU并与base_params对齐（兼容原有返回格式）
    return {k: v.detach().cpu() for k, v in fisher.items()}


def compute_fisher(model, dataloader, alpha: float = 0.5, engine=None):
    """Compute EMA-based diagonal Fisher information for model parameters."""
    device = next(model.parameters()).device
    F_curr = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
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
        loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs["loss"]
        if engine is not None:
            engine.backward(loss)
        else:
            loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                F_batch = param.grad * param.grad
                F_curr[name] = alpha * F_curr[name] + (1 - alpha) * F_batch

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        for tensor in F_curr.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size

    return {k: v.detach().cpu() for k, v in F_curr.items()}

def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)
    # 将预测的 ID 序列解码为字符串
    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:

            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class AdaptiveAdamW(Optimizer):
    """
    一个 PyTorch 优化器，它封装了你的自适应逻辑 (adaLR) 和 AdamW。
    所有计算都在 C++ 后端执行，以获得最大性能。
    """

    def __init__(self, params,
                 # 历史状态 (从 Trainer 传入, p -> tensor 的映射)
                 bar_F_tensors: Dict[torch.Tensor, torch.Tensor],
                 bar_theta_tensors: Dict[torch.Tensor, torch.Tensor],
                 # AdamW 超参
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 # 你的 Adaptive 超参 (从 Trainer 传入)
                 radius=1.0, sigma=0.0, tau=0.0, beta=10.0, alpha_ema=0.9,
                 eta_shrink=1.0, trust_radius_shrink=1e-3, beta_mul=1.0,
                 # Fisher 修正超参
                 fisher_floor_quantile=0.02, fisher_floor_min=1e-12,
                 fisher_floor_mix=0.7, precond_power=0.5,
                 # 内部状态 EMA 超参 (可选)
                 eta_scale_rho=0.1, eta_smin=0.1, eta_smax=10.0
                 ):

        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"无效的 epsilon 值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta 参数 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta 参数 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaptiveAdamW, self).__init__(params, defaults)

        # --------- 1. 存储历史状态 (必须是 GPU 张量) ---------
        # self.bar_F_tensors 和 self.bar_theta_tensors 是 {p: tensor} 的映射
        self.bar_F_tensors = bar_F_tensors
        self.bar_theta_tensors = bar_theta_tensors

        # --------- 2. 存储 Adaptive 超参 ---------
        self.radius_sq = radius ** 2
        self.sigma = sigma
        self.tau = tau
        self.beta = beta
        self.alpha_ema = alpha_ema
        self.eta_shrink = eta_shrink
        self.trust_radius_shrink_sq = trust_radius_shrink ** 2
        self.beta_mul = beta_mul
        self.fisher_floor_quantile = fisher_floor_quantile
        self.fisher_floor_min = fisher_floor_min
        self.fisher_floor_mix = fisher_floor_mix
        self.precond_power = precond_power

        # --------- 3. 用于缩放的内部状态 (可选) ---------
        self._eta_scale = {} # 这个仍然是 {p: float} 映射，开销很小
        self._eta_scale_rho = eta_scale_rho
        self._eta_smin = eta_smin
        self._eta_smax = eta_smax

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # 遍历所有参数组 (例如，一组有 wd，一组没有 wd)
        for group in self.param_groups:
            # AdamW 超参
            beta1, beta2 = group['betas']
            lr_cap = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            # 遍历该组中的每个参数 (p)
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdaptiveAdamW 不支持稀疏梯度")

                state = self.state[p]

                # 1. 初始化状态 (只在第一步执行)
                if len(state) == 0:
                    state['step'] = 0
                    # F_curr (EMA of g^2)
                    state['F_curr'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Adam 状态
                    state['adam_m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['adam_v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # 历史状态 (从 __init__ 获取)
                    f_past_gpu = self.bar_F_tensors.get(p, torch.zeros_like(p))
                    theta_past_gpu = self.bar_theta_tensors.get(p, torch.zeros_like(p))

                    # Adaptive 状态
                    hist_mean = torch.clamp(f_past_gpu.mean(), min=1e-12)
                    f_past_eff = f_past_gpu / hist_mean

                    state['f_past_eff'] = f_past_eff
                    state['theta_past_gpu'] = theta_past_gpu
                    state['r2'] = ((p.detach() - theta_past_gpu).pow(2) * f_past_eff).sum()
                    state['r2_start'] = state['r2'].clone()
                    state['B_round'] = torch.tensor(0.0, device=p.device)
                    state['conf'] = 0

                state['step'] += 1
                t = state['step']

                # --- 获取状态 ---
                F_curr = state['F_curr']
                f_past_eff = state['f_past_eff']
                theta_past_gpu = state['theta_past_gpu']
                r2 = state['r2']
                adam_m = state['adam_m']
                adam_v = state['adam_v']

                g_float = g.float()
                g2 = g_float * g_float
                F_curr.mul_(self.alpha_ema).add_(g2, alpha=1.0 - self.alpha_ema)

                F_hat = F_curr / (1.0 - (self.alpha_ema ** t) + eps)
                scale = 1.0
                curr_mean = F_hat.mean()
                if curr_mean > 0: # f_past_eff.mean() 约等于 1
                    prev_scale = self._eta_scale.get(p, 1.0)
                    ratio = torch.clamp(curr_mean, min=1e-5) # 假设 hist_mean=1
                    scale = (1 - self._eta_scale_rho) * prev_scale + self._eta_scale_rho * ratio.item() # .item() 开销小
                    scale = max(self._eta_smin, min(self._eta_smax, scale))
                    self._eta_scale[p] = scale

                # [Fisher 修正 (softfloor)]
                floor_min = torch.as_tensor(self.fisher_floor_min, device=F_hat.device, dtype=F_hat.dtype)
                pos = F_hat[F_hat > 0]
                if pos.numel() > 0:
                    floor_q = torch.quantile(pos, self.fisher_floor_quantile)
                    floor = torch.maximum(floor_q, floor_min)
                else:
                    floor = floor_min
                F_soft = torch.where(F_hat < floor,
                                     (1.0 - self.fisher_floor_mix) * F_hat + self.fisher_floor_mix * floor, F_hat)

                preconditioner = (F_soft.pow(self.precond_power)) / scale
                v = g_float / (preconditioner + eps)

                # [v 缩放] (保留你的逻辑)
                mean_g_abs = torch.mean(torch.abs(g_float))
                median_v_abs = torch.median(torch.abs(v))
                scale_v = mean_g_abs / (median_v_abs + eps)
                v_scaled = v * scale_v
                v_alg = v_scaled  # 你的代码中 v_alg 和 v_scaled 相同

                # --- eta (自适应步长) ---
                delta_theta = p.detach() - theta_past_gpu

                v2 = (v_alg * v_alg)  # (v_alg 已经是 float)
                cap = torch.quantile(v2, 0.99)
                v2_clip = torch.clamp(v2, max=cap)

                a_local = (v2_clip * f_past_eff).sum()
                b_local = (v_alg * f_past_eff * delta_theta).sum()

                Delta_local = self.radius_sq - r2

                eta = torch.tensor(0.0, device=p.device, dtype=p.dtype)  # 默认为 0

                if Delta_local > self.tau:
                    a_safe = torch.clamp(a_local, min=v2_clip.mean() * v2_clip.numel() * 1e-4)
                    Delta_eff = torch.clamp(Delta_local - self.tau, min=0.0)
                    Delta_eff = self.trust_radius_shrink_sq * Delta_eff
                    beta_eff = self.beta * self.beta_mul

                    if b_local < 0.0:
                        term_u = b_local - self.sigma
                        discriminant = torch.clamp(term_u.pow(2) + beta_eff * a_safe * Delta_eff, min=0.0)
                        eta_closed = (term_u + torch.sqrt(discriminant)) / (beta_eff * a_safe + eps)
                        eta = torch.clamp(eta_closed * self.eta_shrink, max=lr_cap)
                        state['conf'] += 1
                    else:
                        eta_trust = torch.sqrt(Delta_eff / (a_safe + eps))
                        eta = torch.clamp(eta_trust * self.eta_shrink, max=lr_cap)

                    if torch.isnan(eta):
                        eta = torch.tensor(0.0, device=p.device, dtype=p.dtype)

                    # --- AdamW 状态更新 (在 v_scaled 上) ---
                adam_m.mul_(beta1).add_(v_scaled, alpha=1.0 - beta1)
                adam_v.mul_(beta2).addcmul_(v_scaled, v_scaled, value=1.0 - beta2)

                m_hat = adam_m / (1.0 - (beta1 ** t))
                v_hat = adam_v / (1.0 - (beta2 ** t))
                adam_dir = m_hat / (torch.sqrt(v_hat) + eps)

                # --- 状态更新 (B_round, r2) ---
                Q_alg = (g_float * v_alg).sum()
                gain_local = torch.clamp((eta - 0.5 * eta.pow(2)) * Q_alg, min=0.0)
                state['B_round'] = state['B_round'] + gain_local
                state['r2'] = r2 - 2.0 * eta * b_local + (eta.pow(2)) * a_local

                # --------- 5. [执行参数更新] ---------
                # Decoupled Weight Decay
                if wd > 0.0:
                    p.add_(p, alpha=-eta * wd)  # p = p - eta * wd * p

                # AdamW 步长 (使用 adam_dir)
                p.add_(adam_dir, alpha=-eta)  # p = p - eta * adam_dir

        return loss



class UIETrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        radius: float = 1,
        sigma: float = 0.0,
        tau: float = 0.0,
        beta: float = 10.0,
        alpha: float = 0.9,
        comm_bandwidth: float = 1.0,
        comm_fixed_cost: float = 0.0,
        comm_budget: int = 0,
        layer_costs: Dict[str, int] = None,
        accelerator: Accelerator = None,
        **kwargs,
    ):
        self.continual_state = kwargs.pop("state", None)  # 这里直接赋值给 continual_state
        super().__init__(*args, **kwargs)  # 此时 kwargs 中已无 state，父类不会报错

        base_accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            self.accelerator = accelerator
        elif base_accelerator is not None:
            self.accelerator = base_accelerator
        else:
            try:
                from accelerate import Accelerator
                report_to = list(self.args.report_to) if getattr(self.args, "report_to", None) else []
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    log_with=report_to if len(report_to) > 0 else None,
                    project_dir=self.args.output_dir if len(report_to) > 0 else None,
                )
            except Exception:
                import torch, torch.distributed as _dist
                class _Shim:
                    def __init__(self, device):
                        self.device = device

                    def unwrap_model(self, m):
                        return m

                    def prepare(self, obj):
                        return obj

                    def backward(self, loss):
                        loss.backward()

                    def wait_for_everyone(self):
                        pass

                    @property
                    def is_main_process(self):
                        try:
                            return (not _dist.is_available()) or (not _dist.is_initialized()) or _dist.get_rank() == 0
                        except Exception:
                            return True

                    @property
                    def state(self):
                        class S: num_processes = 1

                        return S()

                dev = self.args.device if hasattr(self.args, "device") else (
                    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                )
                self.accelerator = _Shim(dev)



        self.radius = radius
        self.comm_bandwidth = comm_bandwidth  # MB/s
        self.comm_fixed_cost = comm_fixed_cost  # 固定开销（秒）
        self.sigma = sigma
        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.comm_budget = comm_budget
        self.layer_costs = layer_costs or {}
        self.deepspeed = None  # 明确禁用 DS 句柄，避免 Base 类分支误判
        self.deepspeed_engine = None
        self.lambda_conf = 1.0  # 可选：背包里的 λ
        self.method = self.args.method
        self._force_sgd = False
        self.fisher_floor = getattr(self.args, "fisher_floor", 1e-5)
        self.eta_shrink = float(getattr(self.args, "eta_shrink", 1.0))
        self.trust_radius_shrink = float(getattr(self.args, "trust_radius_shrink", 1e-3))
        self.beta_mul = float(getattr(self.args, "beta_mul", 1.0))

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to.
        """
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """添加详细调试信息的 training_step"""

        # 调试信息
        if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            logger.info(f"[DEBUG] training_step 开始, 进程: {self.accelerator.process_index}")

        try:
            loss = super().training_step(model, inputs)
        except Exception as exc:  # pragma: no cover - debug safeguard
            logger.error(f"training_step 错误: {exc}")
            return torch.tensor(
                0.0,
                device=self.accelerator.device if hasattr(self, "accelerator") else None,
                requires_grad=False,
            )

        if hasattr(self, "accelerator") and self.accelerator.is_main_process:
            try:
                loss_value = loss.item()
            except Exception:
                loss_value = float("nan")
            print(f"[DEBUG] 反向传播完成, loss: {loss_value:.6f}")

        return loss



    def train(
        self,
        task_id: int = 1,
        base_params: Dict[str, torch.Tensor] = None,
        cid: int = -1,
        **kwargs,
    ):
        if self.method == "lora_origin" or (self.method == "adaptive" and task_id == 1):
            logger.info(f"[Task {task_id}] 调用标准 super().train() (lora_origin 或 task 1)")
            return super().train(**kwargs)

        elif self.method == "adaptive" and task_id > 1:

            def _canon_name(name: str) -> str:
                return name[7:] if name.startswith("module.") else name

            state_has_history = self.continual_state.has_valid_history() if self.continual_state is not None else False

            # ========== 冷启动：该 client 没有历史 -> 走一次常规训练 + Fisher ==========
            if self.continual_state is None or base_params is None or (
                    self.continual_state is not None and not state_has_history
            ):
                logger.info(f"[Task {task_id}] 调用标准 super().train() (冷启动)")
                output = super().train(**kwargs)

                model_plain = self.accelerator.unwrap_model(self.model)
                sdict = model_plain.state_dict()

                # delta: 使用 base_params 的键集合（已是 LoRA 子集）
                delta = {k: (base_params[k] - sdict[k].detach().cpu()) for k in base_params}

                # Fisher：仅 main 进程用“全量 client 数据”计算，其它进程置零
                if self.accelerator.is_main_process:
                    from torch.utils.data import DataLoader, SequentialSampler
                    full_dl = DataLoader(
                        dataset=self.train_dataset,
                        sampler=SequentialSampler(self.train_dataset),
                        batch_size=self.args.per_device_train_batch_size,
                        collate_fn=self.data_collator,
                        drop_last=False,
                    )
                    F_raw = compute_fisher(model_plain, full_dl)
                else:
                    F_raw = {k: torch.zeros_like(v) for k, v in base_params.items()}

                # 统一规范化 + 放 CPU
                F_client = {_canon_name(k): v.detach().cpu() for k, v in F_raw.items()}
                theta_last = {_canon_name(k): sdict[k].detach().cpu() for k in F_client.keys()}

                self.accelerator.wait_for_everyone()
                return delta, F_client, theta_last

            # ==================== 有历史：自适应训练路径 ====================
            use_adaptive_logic = True
            logger.info(f"===== 开始训练 | 模式：{'自适应' if use_adaptive_logic else '基线'} | Task {task_id} =====")

            # HF Trainer 已经包好 DDP/Accelerate：训练使用 self.model；读写权重用 unwrap 后的 model_plain
            train_dataloader = self.get_train_dataloader()

            # 2) 确保模型被 Accelerate/DDP 包装（避免二次包装）
            from torch.nn.parallel import DistributedDataParallel as DDP
            _wrapped = isinstance(self.model, DDP) or hasattr(self.model, "module")
            if getattr(self, "accelerator", None) is not None:
                if not _wrapped:
                    self.model, train_dataloader = self.accelerator.prepare(self.model, train_dataloader)
                else:
                    train_dataloader = self.accelerator.prepare(train_dataloader)


            device = self.accelerator.device
            model = self.model
            model_plain = self.accelerator.unwrap_model(self.model)

            model.train()
            logger.info(f"Moving historical tensors (F_past, theta_past) to device: {device}...")
            name_to_p = {_canon_name(n): p for n, p in model_plain.named_parameters() if p.requires_grad}
            p_to_hist_F = {}
            p_to_hist_theta = {}

            for cn, p in name_to_p.items():
                if cn in self.continual_state.bar_F:
                    p_to_hist_F[p] = self.continual_state.get_f_past(cn).to(device)
                if cn in self.continual_state.theta_last:
                    p_to_hist_theta[p] = self.continual_state.theta_last[cn].to(device)

            logger.info("Historical tensors moved to GPU.")

            # [创建新的 AdaptiveAdamW 优化器]
            decay_parameters = self.get_decay_parameter_names(model_plain)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               p.requires_grad and _canon_name(n) in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               p.requires_grad and _canon_name(n) not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = AdaptiveAdamW(
                optimizer_grouped_parameters,
                bar_F_tensors=p_to_hist_F,
                bar_theta_tensors=p_to_hist_theta,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                # 传入你的 adaptive 超参
                radius=self.radius, sigma=self.sigma, tau=self.tau, beta=self.beta,
                alpha_ema=self.alpha, eta_shrink=self.eta_shrink,
                trust_radius_shrink=self.trust_radius_shrink, beta_mul=self.beta_mul,
                fisher_floor_quantile=float(getattr(self.args, "fisher_floor_quantile", 0.02)),
                fisher_floor_min=float(getattr(self.args, "fisher_floor_min", 1e-12)),
                fisher_floor_mix=float(getattr(self.args, "fisher_floor_mix", 0.7)),
                precond_power=float(getattr(self.args, "precond_power", 0.5))
            )

            self.optimizer = self.accelerator.prepare(self.optimizer)
            num_epochs = int(self.args.num_train_epochs)
            steps_per_epoch = len(train_dataloader)
            actual_steps = 0
            diag_loss = []  # (用于日志)

            for epoch in range(num_epochs):
                for step, batch in enumerate(train_dataloader):
                    actual_steps += 1
                    model.train()


                    loss = self.training_step(model, batch)


                    if actual_steps % self.args.gradient_accumulation_steps == 0:

                        self.optimizer.step()
                        self.optimizer.zero_grad()


                    if self.accelerator.is_main_process:
                        # (我们保持你之前的日志逻辑，每 5 个 epoch 打印一次)
                        if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
                            # 仅在需要打印时才执行 .item()
                            raw_loss = loss.detach().item() * self.args.gradient_accumulation_steps
                            logger.info(
                                f"Task {task_id} | Epoch [{epoch + 1}/{num_epochs}] | "
                                f"Batch [{step + 1}/{steps_per_epoch}] | Batch Loss: {raw_loss:.6f}"
                            )
                            diag_loss.append(raw_loss)  # 仅记录打印的 loss
            F_client = {}
            theta_last = {}
            delta = {}
            B_round = {}
            F_round = {}
            conf = {}
            name_to_p = {_canon_name(n): p for n, p in model_plain.named_parameters() if p.requires_grad}

            for name, p in name_to_p.items():
                if p in self.optimizer.state:  # 检查优化器是否跟踪了此参数
                    state = self.optimizer.state[p]
                    F_client[name] = state['F_curr'].detach().cpu()
                    B_round[name] = state['B_round'].detach().cpu()
                    conf[name] = state['conf']  # 这是一个 int
                    r2_start_val = state['r2_start'].detach().cpu()
                    r2_end_val = state['r2'].detach().cpu()
                    F_round[name] = 0.5 * torch.clamp(r2_end_val - r2_start_val, min=0.0)
                else:

                    F_client[name] = torch.zeros_like(p).cpu()
                    B_round[name] = torch.tensor(0.0)
                    conf[name] = 0
                    F_round[name] = torch.tensor(0.0)

                # 这些总是被更新
                theta_last[name] = p.detach().cpu()
                delta[name] = base_params[name] - theta_last[name]

            p_round = {n: conf[n] / max(actual_steps, 1) for n in conf}

            values, costs, names = [], [], []

            for name in B_round:
                # 确保 F_round[name] 也是 tensor
                val = (B_round[name] - self.lambda_conf * p_round[name] * F_round[name]).item()
                values.append(val)
                costs.append(max(int(self.layer_costs.get(name, 1)), 1))
                names.append(name)

            total_cost = sum(costs)
            budget = self.comm_budget if self.comm_budget is not None else total_cost
            if budget >= total_cost:
                selected_flags = [True] * len(names)
            else:
                selected_flags = _knapsack(values, costs, budget)

            if self.accelerator.num_processes > 1:
                import torch.distributed as dist
                obj = [selected_flags] if self.accelerator.is_main_process else [None]
                dist.broadcast_object_list(obj, src=0)
                selected_flags = obj[0]

            selection_map = {n: f for n, f in zip(names, selected_flags)}
            adaptive_delta = {}

            for name in delta:  # 使用已经计算好的 delta
                if not selection_map.get(name, True):  # 默认是 True (发送)
                    adaptive_delta[name] = torch.zeros_like(delta[name])
                else:
                    adaptive_delta[name] = delta[name]

            delta = adaptive_delta

            return delta, F_client, theta_last


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, # inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs
        gen_kwargs["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            input_ids=generation_inputs,
            generation_config=generation_config
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    # 跳到lora的forward
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
