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
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
import logging
import time
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import re
try:
    # 新版 Transformers (4.40+) 路径
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    try:
        # 旧版 Transformers 路径
        from transformers.deepspeed import is_deepspeed_zero3_enabled
    except ImportError:
        # 兜底：如果没有安装 DeepSpeed 或版本太老
        def is_deepspeed_zero3_enabled():
            return False
from lorm_utils import LoRMTracker
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


def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    # [修改] 增加 skip_special_tokens=True，防止特殊字符干扰 split
    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    # 只有 Decoder 模型 (Llama) 需要切分，T5 不需要
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:
            # [关键修复] 使用正确的分割符
            if ANSWER_PREFIX in pred:
                # 取最后一个分割符之后的内容，防止 Input 里也有 "Output:"
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                # [兜底逻辑] 如果没找到分割符（模型没生成 Output:），
                # 不要返回空字符串，而是返回原始预测，万一模型直接输出了答案呢？
                # 或者仅仅是打印出来方便排查
                # print(f"[DEBUG Warning] Prefix '{ANSWER_PREFIX}' not found in: {pred[:50]}...")
                final_predictions.append(pred.strip())
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
                 ablation_no_adaptive_lr=False,
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

        # [新增] 存储消融标志
        self.ablation_no_adaptive_lr = ablation_no_adaptive_lr
        # --------- 3. 用于缩放的内部状态 (可选) ---------
        self._eta_scale = {} # 这个仍然是 {p: float} 映射，开销很小
        self._eta_scale_rho = eta_scale_rho
        self._eta_smin = eta_smin
        self._eta_smax = eta_smax
        self._step_idx = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_idx += 1
        debug_eta_min = None
        debug_eta_max = None
        debug_eta_sum = 0.0
        debug_eta_cnt = 0

        # 遍历所有参数组
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
                    state['F_curr'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['adam_m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['adam_v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    f_past_gpu = self.bar_F_tensors.get(p, torch.zeros_like(p))
                    theta_past_gpu = self.bar_theta_tensors.get(p, torch.zeros_like(p))

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

                # gongcheng
                F_hat = F_curr / (1.0 - (self.alpha_ema ** t) + eps)
                scale = 1.0
                curr_mean = F_hat.mean()
                if curr_mean > 0:
                    prev_scale = self._eta_scale.get(p, 1.0)
                    ratio = torch.clamp(curr_mean, min=1e-5)
                    scale = (1 - self._eta_scale_rho) * prev_scale + self._eta_scale_rho * ratio.item()
                    scale = max(self._eta_smin, min(self._eta_smax, scale))
                    self._eta_scale[p] = scale

                # [Fisher 修正]
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

                # [v 缩放]
                mean_g_abs = torch.mean(torch.abs(g_float))
                median_v_abs = torch.median(torch.abs(v))
                scale_v = mean_g_abs / (median_v_abs + eps)
                v_scaled = v * scale_v
                v_alg = v_scaled

                # --- eta (自适应步长) 计算 ---
                delta_theta = p.detach() - theta_past_gpu

                v2 = (v_alg * v_alg)
                cap = torch.quantile(v2, 0.99)
                v2_clip = torch.clamp(v2, max=cap)

                a_local = (v2_clip * f_past_eff).sum()
                b_local = (v_alg * f_past_eff * delta_theta).sum()

                Delta_local = self.radius_sq - r2

                # 默认 eta 为 0
                eta = torch.tensor(0.0, device=p.device, dtype=p.dtype)

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

                # ==================== [消融实验逻辑: 确定 Step Size] ====================
                if self.ablation_no_adaptive_lr:
                    # 如果开启消融：强制使用 AdamW 的固定学习率
                    step_size = torch.tensor(lr_cap, device=p.device, dtype=p.dtype)
                else:
                    # 正常模式：使用自适应计算出的 eta
                    step_size = eta
                # ======================================================================

                # 记录实际使用的步长用于日志
                eta_scalar = float(step_size.detach())
                state['eta_last'] = eta_scalar

                # 统计
                debug_eta_sum += eta_scalar
                debug_eta_cnt += 1
                if debug_eta_min is None or eta_scalar < debug_eta_min:
                    debug_eta_min = eta_scalar
                if debug_eta_max is None or eta_scalar > debug_eta_max:
                    debug_eta_max = eta_scalar

                # --- AdamW 状态更新 (在 v_scaled 上) ---
                adam_m.mul_(beta1).add_(v_scaled, alpha=1.0 - beta1)
                adam_v.mul_(beta2).addcmul_(v_scaled, v_scaled, value=1.0 - beta2)

                m_hat = adam_m / (1.0 - (beta1 ** t))
                v_hat = adam_v / (1.0 - (beta2 ** t))
                adam_dir = m_hat / (torch.sqrt(v_hat) + eps)

                # --- 状态更新 (B_round, r2) ---
                # [关键] 所有的计算全部基于 step_size
                Q_alg = (g_float * v_alg).sum()

                # 计算收益：如果用 step_size 走这一步，收益是多少
                gain_local = torch.clamp((step_size - 0.5 * step_size.pow(2)) * Q_alg, min=0.0)
                state['B_round'] = state['B_round'] + gain_local

                # 更新半径
                state['r2'] = r2 - 2.0 * step_size * b_local + (step_size.pow(2)) * a_local

                # --------- 5. [执行参数更新] ---------
                # Decoupled Weight Decay
                if wd > 0.0:
                    p.add_(p, alpha=-step_size * wd)  # 使用 step_size

                # AdamW 步长 (使用 adam_dir)
                p.add_(adam_dir, alpha=-step_size)  # 使用 step_size

        if debug_eta_cnt > 0:
            try:
                is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                is_main = True

            if is_main and self._step_idx <= 20:
                eta_mean = debug_eta_sum / debug_eta_cnt
                logger.info(
                    f"[AdaptiveAdamW] step {self._step_idx}: "
                    f"eta_min={debug_eta_min:.3e}, eta_max={debug_eta_max:.3e}, eta_mean={eta_mean:.3e}, "
                    f"count={debug_eta_cnt}"
                )

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
        ewc_fisher: Dict[str, torch.Tensor] = None,
        ewc_params: Dict[str, torch.Tensor] = None,
        gem_dataset=None,
        **kwargs,
    ):
        self.continual_state = kwargs.pop("state", None)  # 这里直接赋值给 continual_state
        super().__init__(*args, **kwargs)  # 此时 kwargs 中已无 state，父类不会报错



        # 🤗 Transformers 在 ``Trainer`` 基类内部已经会根据 ``TrainingArguments``
        # 初始化一个 ``Accelerator`` 实例，并在数据加载、梯度累积和同步等
        # 关键路径中复用它。如果我们在子类里无条件地再创建一个新的
        # ``Accelerator``，就会触发第二套分布式状态 —— 这在 ``accelerate
        # launch`` 的场景下会造成进程间握手的不一致，从而出现首轮训练
        # 卡住的现象。这里通过读取父类已经准备好的 ``accelerator``（若存在）
        # 并在用户显式传入时才覆写，确保我们始终引用同一个分布式上下文。
        base_accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            self.accelerator = accelerator
        elif base_accelerator is not None:
            # Trainer 可能通过属性而非实例属性暴露 accelerator，这里将其缓存
            # 到实例字典中，方便子类内部统一访问。
            self.accelerator = base_accelerator
        else:
            # 兼容某些 Transformers 版本：__init__ 尚未设置 .accelerator
            try:
                from accelerate import Accelerator
                report_to = list(self.args.report_to) if getattr(self.args, "report_to", None) else []
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    log_with=report_to if len(report_to) > 0 else None,
                    project_dir=self.args.output_dir if len(report_to) > 0 else None,
                )
            except Exception:
                # 极限兜底（无 accelerate 环境也不崩）
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
        print(f"DEBUG: UIETrainer initialized with radius = {self.radius}")
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

        self.ewc_fisher = ewc_fisher
        self.ewc_params = ewc_params
        self.ewc_lambda = getattr(self.args, "ewc_lambda", 0.0)

        # [Safety Check] 仅在 GEM 模式下初始化相关组件
        self.gem_dataset = gem_dataset
        self.gem_loader = None
        self.gem_iterator = None

        # 严格隔离：只有明确指定 method 为 gem 时才激活
        if self.args.method == "gem" and self.gem_dataset is not None:
            self.init_gem_loader()

    def init_gem_loader(self):
        """A-GEM 专用：初始化记忆数据加载器"""
        if self.accelerator.is_main_process:
            logger.info(
                f"🛠️ [Trainer Internal] Initializing GEM DataLoader with batch size {self.args.per_device_train_batch_size}...")

        sampler = RandomSampler(self.gem_dataset, replacement=True)
        # 使用更小的 batch size 以减少显存开销，或者与训练 batch size 一致
        bs = self.args.per_device_train_batch_size

        self.gem_loader = DataLoader(
            self.gem_dataset,
            batch_size=bs,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False
        )
        self.gem_iterator = iter(self.gem_loader)

    def get_gem_batch(self):
        try:
            batch = next(self.gem_iterator)
        except StopIteration:
            self.gem_iterator = iter(self.gem_loader)
            batch = next(self.gem_iterator)
        return self._prepare_inputs(batch)



    def compute_loss(self, model, inputs, return_outputs=False):
        # 还原为只调用父类，不要在这里算 EWC
        return super().compute_loss(model, inputs, return_outputs)


    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to.
        """
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        1. 执行标准的前向+反向传播 (Super Class Logic)
        2. [EWC核心修复] 在反向传播结束后，手动计算并叠加 EWC 梯度
        """

        # 1) 兼容 decoder-only（LLaMA）多出来的字段，防止 model(**inputs) 报错
        if "input_ids_wo_label" in inputs:
            # 拷一份，避免修改到 Trainer 外部的原始 dict
            inputs = dict(inputs)
            inputs.pop("input_ids_wo_label", None)

        # 2) 交给父类做真正的前向 + 反向（包括 Deepspeed / Accelerate 逻辑）
        try:
            loss = super().training_step(model, inputs)
        except Exception as exc:  # 防御性兜底，避免训练直接崩溃
            logger.error(f"training_step 错误: {exc}")
            return torch.tensor(
                0.0,
                device=self.accelerator.device if hasattr(self, "accelerator") else None,
                requires_grad=False,
            )

        if self.args.method == "gem" and self.gem_loader is not None:
            # (A) 筛选可训练参数 (LoRA)
            trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                # (B) 备份当前任务梯度 (g_curr)
                # 必须 clone，因为后续计算参考梯度会清空 grad
                grad_curr = []
                for p in trainable_params:
                    if p.grad is not None:
                        grad_curr.append(p.grad.detach().clone())
                    else:
                        grad_curr.append(torch.zeros_like(p))

                # (C) 清空梯度，计算记忆数据的梯度 (g_ref)
                model.zero_grad()
                gem_batch = self.get_gem_batch()

                if "input_ids_wo_label" in gem_batch:
                    gem_batch.pop("input_ids_wo_label")

                # 使用上下文管理器确保混合精度(AMP)等配置正确
                with self.compute_loss_context_manager():
                    loss_gem = self.compute_loss(model, gem_batch)

                if self.args.n_gpu > 1:
                    loss_gem = loss_gem.mean()

                # 反向传播 (A-GEM 参考梯度)
                if hasattr(self, "accelerator"):
                    self.accelerator.backward(loss_gem)
                else:
                    loss_gem.backward()

                # (D) 提取参考梯度 g_ref
                grad_ref = []
                for p in trainable_params:
                    if p.grad is not None:
                        grad_ref.append(p.grad.detach().clone())
                    else:
                        grad_ref.append(torch.zeros_like(p))

                # (E) 恢复 g_curr 到 p.grad (准备原地修改)
                model.zero_grad()
                for p, g_c in zip(trainable_params, grad_curr):
                    if p.grad is None:
                        p.grad = g_c
                    else:
                        p.grad.copy_(g_c)

                # (F) 投影计算 (Projection)
                # dot = g_curr · g_ref
                dot_prod = sum(torch.sum(gc * gr) for gc, gr in zip(grad_curr, grad_ref))

                # 只有当方向冲突 (点积 < 0) 时才修正
                if dot_prod < 0:
                    ref_mag = sum(torch.sum(gr * gr) for gr in grad_ref)
                    # g_new = g_curr - (dot / mag) * g_ref
                    scale = dot_prod / (ref_mag + 1e-12)

                    with torch.no_grad():
                        for p, gr in zip(trainable_params, grad_ref):
                            p.grad.add_(gr, alpha=-scale)


        if self.args.method == "ewc" and getattr(self, "ewc_fisher", None) is not None:
            ewc_lambda = getattr(self.args, "ewc_lambda", 0.0)

            # 用于 Log 显示的 Loss 值 (仅记录，不参与反向传播)
            ewc_loss_val = 0.0

            # 使用 no_grad 确保不建立任何计算图，这是解决 DDP 报错的关键
            with torch.no_grad():
                for name, param in model.named_parameters():
                    # 去除 DDP 可能加上的 module. 前缀以匹配 key
                    clean_name = name.replace("module.", "")

                    if param.requires_grad and clean_name in self.ewc_fisher:
                        # 确保 tensor 在同一设备
                        f_val = self.ewc_fisher[clean_name].to(param.device)
                        star_val = self.ewc_params[clean_name].to(param.device)

                        # --- A. 计算梯度 ---
                        # EWC Loss 公式: L = (lambda/2) * F * (theta - theta*)^2
                        # 对 theta 求导: Grad = lambda * F * (theta - theta*)
                        ewc_grad = ewc_lambda * f_val * (param.data - star_val)

                        # --- B. 叠加梯度 ---
                        # 直接加到现有的 Task Gradient 上
                        if param.grad is not None:
                            param.grad.add_(ewc_grad)

                        # --- C. 计算 Loss 值用于打印 (可选) ---
                        # 仅做记录，方便你看 Log 确认 EWC 生效了
                        current_ewc_loss = (f_val * (param.data - star_val).pow(2)).sum()
                        ewc_loss_val += current_ewc_loss.item()

            # [Log] 打印调试信息 (仅主进程，且每 10 个 micro steps 打一次，避免刷屏)
            if hasattr(self, "accelerator") and self.accelerator.is_main_process:
                # 随便用个计数器判断一下，或者直接复用下面的 logging 逻辑
                # 这里简单处理，每次调 training_step 都累加，取模打印
                if not hasattr(self, "_ewc_log_step"): self._ewc_log_step = 0
                self._ewc_log_step += 1
                if self._ewc_log_step % 10 == 0:
                    final_ewc_loss_display = (ewc_lambda / 2.0) * ewc_loss_val
                    logger.info(f"[DEBUG EWC] Task Loss: {loss.item():.4f} | EWC Term: {final_ewc_loss_display:.4f}")

            # 注意：我们不需要修改返回的 'loss' 变量，因为 backward 已经结束了。
            # 修改返回的 loss 只会影响 Tensorboard 上的显示，不会影响训练。
            # 如果你想在 Tensorboard 上看到总 loss，可以这样加（不影响梯度）：
            loss += (ewc_lambda / 2.0) * ewc_loss_val

        # ============================================================
        # 4) 原有的 Logging 逻辑 (保持不变)
        # ============================================================



        # 3) 只在主进程上做 logging
        if hasattr(self, "accelerator") and self.accelerator.is_main_process:
            # 维护一个 micro-step 计数器，用来和 gradient_accumulation_steps 对齐
            if not hasattr(self, "_micro_step"):
                self._micro_step = 0
            self._micro_step += 1

            # 当前梯度累积配置
            try:
                ga = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
            except Exception:
                ga = 1

            # 是否完成了一个梯度累积周期（即将进行一次 optimizer.step）
            is_block_end = (self._micro_step % ga == 0)

            if is_block_end:
                # Trainer 外层在 optimizer.step 之后才会 self.state.global_step += 1
                # 这里先预估一下“下一个 global_step”
                current_step = getattr(self.state, "global_step", 0)
                next_step = current_step + 1

                # logging_steps 配置（保证 >=1）
                try:
                    log_every = max(1, int(getattr(self.args, "logging_steps", 1)))
                except Exception:
                    log_every = 1

                # 只在满足 logging_steps 的时候打一次 log
                if next_step % log_every == 0:
                    # 参考 Trainer 内部的 nan/inf 过滤逻辑
                    if (not getattr(self.args, "logging_nan_inf_filter", False)) or (
                            loss is not None and torch.isfinite(loss)
                    ):
                        try:
                            loss_value = loss.item()
                        except Exception:
                            loss_value = float("nan")
                        logger.info(f"[global_step {next_step}] 反向传播完成, loss: {loss_value:.6f}")

        return loss

    def _pad_across_processes(self, tensor, pad_index=-100):
        return self.accelerator.pad_across_processes(tensor, dim=1, pad_index=pad_index)

    def train(
        self,
        task_id: int = 1,
        base_params: Dict[str, torch.Tensor] = None,
        cid: int = -1,
        **kwargs,
    ):  # type: ignore[override]

        if self.method == "lorm":
            # 1. 正常训练 (Standard SGD/AdamW)
            # 这里的 train 只是更新本轮“未冻结”的参数 (A 或 B)
            if self.accelerator.is_main_process:
                logger.info(f"[Client {cid}] LoRM: Running standard training...")

            train_output = super().train(**kwargs)


            # 2. Post-Hoc Gram Matrix Computation
            # 训练结束后，跑一遍数据计算 Gram 矩阵
            if self.accelerator.is_main_process:
                logger.info(f"[Client {cid}] LoRM: Computing Gram matrices...")

            tracker = LoRMTracker(self.model, self.accelerator.device)
            tracker.register_hooks()  # 只 Hook lora_A 的输入

            train_dataloader = self.get_train_dataloader()
            self.model.eval()

            with torch.no_grad():
                for step, inputs in enumerate(train_dataloader):
                    inputs = self._prepare_inputs(inputs)
                    # 移除不兼容字段
                    if "input_ids_wo_label" in inputs:
                        inputs.pop("input_ids_wo_label", None)

                    # 前向传播触发 Hook
                    try:
                        _ = self.model(**inputs)
                    except Exception as e:
                        logger.exception(
                            f"[Client {cid}] LoRM Gram forward failed at step={step}. "
                            f"Keys={list(inputs.keys())}. This must be fixed (do NOT ignore), "
                            f"otherwise grams may be empty and LoRM aggregation will become no-op."
                        )
                        raise

            # 保存 Grams 到 Trainer 实例，供联邦聚合使用
            grams = tracker.get_grams()
            tracker.remove_hooks()

            if len(grams) == 0:
                raise RuntimeError(
                    f"[Client {cid}] LoRM grams is EMPTY. "
                    "This usually means your forward never reached lora_A hooks. "
                    "Fix the forward inputs instead of ignoring exceptions."
                )


            # 多卡：把各 rank 的 gram_diag 相加得到“全数据”的对角向量
            if self.accelerator.num_processes > 1:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    for k in sorted(grams.keys()):  # 关键：固定 all_reduce 顺序
                        t = grams[k].to(self.accelerator.device)
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        grams[k] = t.cpu()

            # 只让主进程把 grams 暴露给 federated 端；其他进程置空，避免重复使用
            if self.accelerator.is_main_process:
                self.lorm_grams = grams
            else:
                self.lorm_grams = {}

            if self.accelerator.is_main_process:
                logger.info(f"[Client {cid}] LoRM: Grams computed. Count: {len(self.lorm_grams)}")

            return train_output


        if self.method in ["lora_origin", "ewc", "replay", "gem"] or (self.method == "adaptive" and task_id == 1):
            logger.info(f"[Task {task_id}] Method '{self.method}': 调用标准 super().train()")
            return super().train(**kwargs)

        elif self.method == "adaptive" and task_id > 1:

            def _canon_name(name: str) -> str:
                return name[7:] if name.startswith("module.") else name

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
            name_to_p = {_canon_name(n): p for n, p in model.named_parameters() if p.requires_grad}
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
                precond_power=float(getattr(self.args, "precond_power", 0.5)),
                ablation_no_adaptive_lr=getattr(self.args, "ablation_no_adaptive_lr", False)
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

                    # 1. 前向 (使用父类的 training_step)
                    # training_step 内部处理了 autocast, loss 计算, 和 gradient_accumulation
                    loss = self.training_step(model, batch)

                    # 2. 反向 (training_step 已经处理了)
                    # self.accelerator.backward(loss) # (已在 training_step 中完成)

                    # 3. [核心] 优化器步骤
                    # 检查是否需要执行优化器步骤（处理梯度累积）
                    if actual_steps % self.args.gradient_accumulation_steps == 0:

                        # (可选：梯度裁剪)
                        # if self.args.max_grad_norm is not None:
                        #     self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        # 这 ONE line 替换了你所有的 Python 循环！
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # 4. [日志] (现在可以安全地高频调用 .item() 了)
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
            name_to_p = {_canon_name(n): p for n, p in model.named_parameters() if p.requires_grad}
            eta_vals = []

            for name, p in name_to_p.items():
                if p in self.optimizer.state:  # 检查优化器是否跟踪了此参数
                    state = self.optimizer.state[p]
                    F_client[name] = state['F_curr'].detach().cpu()
                    B_round[name] = state['B_round'].detach().cpu()
                    conf[name] = state['conf']  # 这是一个 int
                    r2_start_val = state['r2_start'].detach().cpu()
                    r2_end_val = state['r2'].detach().cpu()
                    F_round[name] = 0.5 * torch.clamp(r2_end_val - r2_start_val, min=0.0)
                    if 'eta_last' in state:
                        eta_vals.append(float(state['eta_last']))
                else:
                    # (参数可能没有被优化，例如没有梯度)
                    F_client[name] = torch.zeros_like(p).cpu()
                    B_round[name] = torch.tensor(0.0)
                    conf[name] = 0
                    F_round[name] = torch.tensor(0.0)

                # 这些总是被更新
                theta_last[name] = p.detach().cpu()
                delta[name] = base_params[name] - theta_last[name]

            if eta_vals and self.accelerator.is_main_process:
                eta_min = min(eta_vals)
                eta_max = max(eta_vals)
                eta_mean = sum(eta_vals) / len(eta_vals)
                logger.info(
                    f"[Task {task_id}] AdaptiveAdamW 最后一轮各层 eta_last 统计: "
                    f"min={eta_min:.3e}, max={eta_max:.3e}, mean={eta_mean:.3e}, count={len(eta_vals)}"
                )


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
            # ========== 修改开始：随机层选择逻辑 ==========
            if getattr(self.args, "random_layer_selection", False):
                import random

                # 核心修改：利用现有的 seed 构造局部随机源，确保复现性
                # self.args.seed 是全局种子
                # task_id 区分不同任务
                # actual_steps 区分同一任务中的不同轮次 (Train Loop中的计数器)
                # 这样既利用了你的 seed，又避免了不同轮次选一样的层
                local_seed = self.args.seed + task_id + actual_steps
                rng = random.Random(local_seed)

                # 创建索引并随机打乱
                indices = list(range(len(names)))
                rng.shuffle(indices)

                selected_flags = [False] * len(names)
                current_cost = 0

                # 贪心填充：打乱后依次尝试放入，直到塞满 Budget
                for idx in indices:
                    cost = costs[idx]
                    if current_cost + cost <= budget:
                        selected_flags[idx] = True
                        current_cost += cost

                # 仅在主进程打印日志，避免刷屏
                if self.accelerator.is_main_process:
                    # 偶尔打印一下（比如每100步），或者只打印简单的 info
                    logger.info(
                        f"[Task {task_id} | Step {actual_steps}] Random Selection: Budget {budget}, Used {current_cost}")

            else:
                # 原有的背包算法 (Ours)
                if budget >= total_cost:
                    selected_flags = [True] * len(names)
                else:
                    selected_flags = _knapsack(values, costs, budget)
            # ========== 修改结束 ==========

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

        if not self.args.predict_with_generate or prediction_loss_only:
            # 移除不兼容参数，防止调用 super().forward 报错
            if "input_ids_wo_label" in inputs:
                inputs.pop("input_ids_wo_label")
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # 配置 Generation
        gen_max_new_tokens = getattr(self.args, "generation_max_length", None)
        if gen_max_new_tokens is None:
            # 向前兼容：如果 args 里（将来）也挂了 max_target_length，就用它；
            # 再不行就退回一个安全默认值 50（对应 DataTrainingArguments 的默认）
            gen_max_new_tokens = getattr(self.args, "max_target_length", 50)

        gen_kwargs = {
            "max_new_tokens": gen_max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
        }

        # 针对 Llama 的特殊配置
        if inputs.get("input_ids_wo_label", None) is not None:
            gen_kwargs.update({
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            })
        else:
            # T5 配置
            gen_kwargs.update({
                "decoder_start_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 0,
            })

        generation_config = GenerationConfig(**gen_kwargs)


        is_enc_dec = bool(getattr(model.config, "is_encoder_decoder", False))

        # 生成逻辑
        if is_enc_dec:
            # [T5]
            generation_inputs = inputs[self.model.encoder.main_input_name]
            generated_tokens = self.model.generate(
                input_ids=generation_inputs,
                generation_config=generation_config,
            )
        else:
            # [Llama] 使用 input_ids_wo_label 进行生成
            input_ids_wo_label = inputs.get("input_ids_wo_label", inputs[self.model.main_input_name])

            generated_tokens = self.model.generate(
                input_ids=input_ids_wo_label,
                generation_config=generation_config,
            )

            # [截断] 别人代码里可能在 collator 处理了，或者这里处理
            # 标准 AutoModel 生成会包含 input，必须截断
            input_len = input_ids_wo_label.shape[1]
            generated_tokens = generated_tokens[:, input_len:]




        # 计算 Loss (如果需要)
        # 在计算 Loss 前必须把 input_ids_wo_label 移除，否则标准模型会报错
        if "input_ids_wo_label" in inputs:
            inputs.pop("input_ids_wo_label")

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = loss.mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        # Padding 对齐逻辑 (保持不变)
        if generated_tokens.shape[-1] < gen_kwargs["max_new_tokens"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"])

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
