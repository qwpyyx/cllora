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
from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX
from collections import defaultdict
# logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from accelerate import Accelerator
import logging
import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Union
from collections import defaultdict

# logger = logging.getLogger(__name__)

def analyze_tensor(tensor, tensor_name, expected_range=None):
    """
    分析单个张量的分布和异常值
    tensor: 待分析的张量（PyTorch）
    tensor_name: 张量名称（用于打印和绘图）
    expected_range: 业务预期的合理范围，如 (0, 1)（可选）
    """
    # 转移到CPU并展平（方便处理多维张量）
    tensor_cpu = tensor.cpu().flatten()
    tensor_np = tensor_cpu.numpy()

    # 1. 计算核心统计量
    stats = {
        "max": tensor_cpu.max().item(),
        "min": tensor_cpu.min().item(),
        "mean": tensor_cpu.mean().item(),
        "std": tensor_cpu.std().item(),
        "median": torch.median(tensor_cpu).item(),
        "count": len(tensor_cpu)  # 元素总数
    }

    print(f"\n===== 张量 {tensor_name} 分析结果 =====")
    print(f"元素总数: {stats['count']}")
    print(f"最大值: {stats['max']:.6f} | 最小值: {stats['min']:.6f}")
    print(f"均值: {stats['mean']:.6f} | 标准差: {stats['std']:.6f} | 中位数: {stats['median']:.6f}")

    # 2. 用“均值±3倍标准差”筛选异常值（统计意义上的离群点）
    upper_threshold = stats["mean"] + 3 * stats["std"]
    lower_threshold = stats["mean"] - 3 * stats["std"]
    outliers_upper = tensor_cpu[tensor_cpu > upper_threshold]
    outliers_lower = tensor_cpu[tensor_cpu < lower_threshold]
    print(f"\n统计异常值（均值±3倍标准差）:")
    print(f"上界: {upper_threshold:.6f} | 超出上界的异常值数量: {len(outliers_upper)}")
    if len(outliers_upper) > 0:
        print(f"  上界异常值示例: {outliers_upper[:5]}")  # 打印前5个
    print(f"下界: {lower_threshold:.6f} | 超出下界的异常值数量: {len(outliers_lower)}")
    if len(outliers_lower) > 0:
        print(f"  下界异常值示例: {outliers_lower[:5]}")

    # 3. 结合业务预期范围筛选（若提供）
    if expected_range is not None:
        lower_exp, upper_exp = expected_range
        invalid = tensor_cpu[(tensor_cpu < lower_exp) | (tensor_cpu > upper_exp)]
        print(f"\n业务范围外的值（预期 {lower_exp} ~ {upper_exp}）:")
        print(f"  数量: {len(invalid)} | 占比: {len(invalid) / stats['count']:.2%}")
        if len(invalid) > 0:
            print(f"  示例: {invalid[:5]}")

    # 4. 可视化：直方图 + 箱线图（直观看分布和离群点）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 直方图（看整体分布）
    ax1.hist(tensor_np, bins=50, alpha=0.7)
    ax1.axvline(stats["mean"], color='r', linestyle='--', label=f"均值: {stats['mean']:.2f}")
    ax1.axvline(upper_threshold, color='g', linestyle=':', label=f"上界: {upper_threshold:.2f}")
    ax1.set_title(f"{tensor_name} 直方图")
    ax1.set_xlabel("值")
    ax1.set_ylabel("频数")
    ax1.legend()

    # 箱线图（看离群点）
    ax2.boxplot(tensor_np)
    ax2.set_title(f"{tensor_name} 箱线图")
    ax2.set_ylabel("值")

    plt.tight_layout()
    plt.show()

    return stats  # 返回统计量，方便后续对比

class _ActivationProbe:
    def __init__(self, topk=5, ratio_trigger=30.0, print_limit=10):
        self.topk = topk
        self.ratio_trigger = ratio_trigger  # p99/median 超过该比值就报警
        self.print_limit = print_limit
        self._handles = []
        self._store = {}   # mname -> dict(stats)

    @staticmethod
    def _flatten_last(x):
        if x is None:
            return None
        if x.dim() == 2:
            return x
        return x.view(-1, x.size(-1))

    @torch.no_grad()
    def _quantiles(self, v: torch.Tensor):
        v = v.float()
        med = torch.median(v)
        p95 = torch.quantile(v, 0.95)
        p99 = torch.quantile(v, 0.99)
        return float(med), float(p95), float(p99), float(p99 / max(med.item(), 1e-8))

    def register(self, model, named_modules: dict, lora_param_names: list[str]):
        # 只在含 lora_A / lora_B 的父 Linear 上挂钩子
        targets = set()
        for pname in lora_param_names:
            if (".lora_A" not in pname) and (".lora_B" not in pname):
                continue
            mname = pname.split(".lora_")[0]
            if mname in named_modules:
                targets.add(mname)

        self._store = {}
        for mname in sorted(targets):
            mod = named_modules[mname]
            self._store[mname] = {}

            def fwd_hook(m, inp, out, mname=mname):
                try:
                    x = inp[0].detach()
                    xf = self._flatten_last(x)
                    tok_l2 = xf.norm(dim=-1)  # 逐 token L2
                    (med, p95, p99, ratio) = self._quantiles(tok_l2)
                    rec = self._store[mname]
                    rec["x_med"], rec["x_p95"], rec["x_p99"], rec["x_ratio"] = med, p95, p99, ratio

                    # 可选：估算 LoRA 通道内的幅度（若能取到权重）
                    try:
                        A = getattr(m, "lora_A").get("default").weight  # peft>=0.7
                        B = getattr(m, "lora_B").get("default").weight
                        scaling = float(getattr(m, "scaling", 1.0))
                        xA = xf @ A    # [N, r]
                        xAB = xA @ B   # [N, out]
                        rec["xA_med"], _, rec["xA_p99"], _ = self._quantiles(xA.norm(dim=-1))
                        rec["xAB_med"], _, rec["xAB_p99"], _ = self._quantiles((xAB * scaling).norm(dim=-1))
                    except Exception:
                        pass
                except Exception:
                    pass

            def bwd_hook(m, gin, gout, mname=mname):
                try:
                    dy = gout[0]
                    if dy is None:
                        return
                    dy = dy.detach()
                    dyf = self._flatten_last(dy)
                    (med, p95, p99, ratio) = self._quantiles(dyf.norm(dim=-1))
                    rec = self._store[mname]
                    rec["dy_med"], rec["dy_p95"], rec["dy_p99"], rec["dy_ratio"] = med, p95, p99, ratio
                except Exception:
                    pass

            self._handles.append(mod.register_forward_hook(fwd_hook))
            self._handles.append(mod.register_full_backward_hook(bwd_hook))

    def begin_batch(self):
        for k in self._store:
            self._store[k].clear()

    def report_if_needed(self, model, logger, global_grad_norm, clip_norm=1.0):
        # 触发条件：全局梯度过大，或任一层的 x_ratio / dy_ratio 超过阈值
        trigger = (global_grad_norm is not None and global_grad_norm > 20 * clip_norm)
        hot = []
        for mname, rec in self._store.items():
            xr = rec.get("x_ratio", 0.0)
            dyr = rec.get("dy_ratio", 0.0)
            if xr > self.ratio_trigger or dyr > self.ratio_trigger:
                trigger = True
                hot.append((max(xr, dyr), mname, rec))
        if not trigger:
            return

        # 排序后打印 top-k 可疑层
        hot.sort(reverse=True)
        hot = hot[: self.print_limit]

        # 辅助：取该层 LoRA 参数梯度范数
        named_params = dict(model.named_parameters())

        for _, mname, rec in hot:
            # 尝试拿同名前缀的 A/B 权重的 grad 范数
            def _gn(pn):
                t = named_params.get(pn, None)
                if (t is not None) and (t.grad is not None):
                    return float(t.grad.detach().norm().item())
                return None

            pa = f"{mname}.lora_A.default.weight"
            pb = f"{mname}.lora_B.default.weight"
            ga = _gn(pa)
            gb = _gn(pb)

            logger.warning(
                f"[ACT-PROBE] {mname} | "
                f"x_med={rec.get('x_med')} x_p99={rec.get('x_p99')} x_ratio={rec.get('x_ratio')} | "
                f"dy_med={rec.get('dy_med')} dy_p99={rec.get('dy_p99')} dy_ratio={rec.get('dy_ratio')} | "
                f"xA_med={rec.get('xA_med', None)} xA_p99={rec.get('xA_p99', None)} | "
                f"xAB_med={rec.get('xAB_med', None)} xAB_p99={rec.get('xAB_p99', None)} | "
                f"||grad(A)||={ga} ||grad(B)||={gb}"
            )

    def unregister(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
# === [新增结束] ============================================================

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
    修复版的在线Fisher更新函数，确保只对需要梯度的参数计算Fisher
    """
    # 保存当前梯度状态
    original_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            original_grads[name] = param.grad.clone()

    try:
        # 确保模型处于训练模式
        model.train()

        # 数据移至设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch.get("labels", None)

        if labels is None:
            return F_curr, {name: torch.zeros_like(v) for name, v in F_curr.items()}

        # 只选择需要梯度的参数
        params_to_compute = []
        param_names = []
        for name, param in unwrap_model.named_parameters():
            if name in F_curr and param.requires_grad:
                params_to_compute.append(param)
                param_names.append(name)

        if not params_to_compute:

            return F_curr, {}

        # 前向计算（需要计算梯度）
        model.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # 初始化batch级梯度平方存储
        batch_grad_sum = {
            name: torch.zeros_like(param, device=device)
            for name, param in F_curr.items()
        }

        batch_size = inputs["input_ids"].shape[0]
        valid_sample_count = 0

        # 逐样本计算梯度
        for i in range(batch_size):
            sample_label = labels[i]
            valid_mask = sample_label != -100
            if not valid_mask.any():
                continue
            valid_sample_count += 1

            # 计算有效位置的log概率总和
            seq_len = log_probs[i].shape[0]
            sample_log_prob = log_probs[i, torch.arange(seq_len), sample_label]
            sample_log_prob = sample_log_prob[valid_mask].sum()

            # 计算梯度
            grads = torch.autograd.grad(
                sample_log_prob,
                params_to_compute,
                create_graph=False,  # 不需要创建计算图
                retain_graph=(i < batch_size - 1),
                allow_unused=True
            )

            # 累积梯度平方
            for idx, name in enumerate(param_names):
                if idx < len(grads) and grads[idx] is not None:
                    batch_grad_sum[name] += grads[idx].detach() ** 2

        # 计算当前batch的Fisher
        F_batch = {}
        if valid_sample_count > 0:
            F_batch = {name: grad_sum / valid_sample_count for name, grad_sum in batch_grad_sum.items()}
        else:
            F_batch = {name: torch.zeros_like(v) for name, v in batch_grad_sum.items()}


        # EMA更新
        updated_F_curr = {
            name: alpha_ema * F_curr[name] + (1 - alpha_ema) * F_batch[name]
            for name in F_curr
        }


        return updated_F_curr, F_batch

    finally:
        # 恢复原始梯度状态
        for name, param in model.named_parameters():
            if name in original_grads:
                param.grad = original_grads[name]

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


        # Accelerator 集成
        if accelerator is None:
            # 如果没有传入 accelerator，创建一个默认的
            self.accelerator = Accelerator()
        else:
            self.accelerator = accelerator



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


        self.adam_states = {}  # 存储Adam状态：key为参数名，value为字典(m, v, t)
        # 初始化Adam超参数（可根据需要调整）
        self.beta1 = 0.9  # 一阶矩动量系数
        self.beta2 = 0.999  # 二阶矩动量系数
        self.eps = 1e-8  # 数值稳定项


    def _ema_fisher_from_synced_grads(self, unwrapped_model, F_curr: Dict[str, torch.Tensor], alpha: float):
        """
        用 DDP 同步后的 p.grad 估计 Fisher 对角（EMA 版本）：
          F[name] = alpha * F[name] + (1 - alpha) * (grad^2)
        说明：
          - 必须在 self.accelerator.backward(loss) 之后调用；
          - 不需要再做 dist.all_reduce，DDP 已经把 grad 同步为一致值；
          - 仅对 requires_grad 且存在 grad 的参数更新。
        """
        for name, p in unwrapped_model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g2 = (p.grad.detach() ** 2)
            if name not in F_curr:
                F_curr[name] = torch.zeros_like(p, device=p.device)
            F_curr[name] = alpha * F_curr[name] + (1.0 - alpha) * g2
        return F_curr

    # ===== [NEW] Fisher EMA（DDP/Accelerate 已同步梯度）=====
    def _synchronized_ema_fisher(
            self,
            model,
            F_curr: Dict[str, torch.Tensor],
            alpha: float,
            *,
            ddp_avg_of_squares: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        要求：必须在 self.accelerator.backward(loss) 之后调用。
        默认用 (E[g])^2；若 ddp_avg_of_squares=True，则改用 E[g^2]（多一次 all_reduce）。
        """
        import torch
        import torch.distributed as dist

        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(model)
        world_size = max(1, getattr(self.accelerator.state, "num_processes", 1))

        for name, p in unwrapped.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue
            g = p.grad.detach()
            need_upcast = (g.dtype in (torch.float16, torch.bfloat16))
            g2 = (g.float() * g.float()) if need_upcast else (g * g)

            if ddp_avg_of_squares and world_size > 1:
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(g2, op=dist.ReduceOp.SUM)
                    g2 = g2 / float(world_size)

            if name not in F_curr:
                F_curr[name] = torch.zeros_like(
                    p, device=p.device, dtype=(torch.float32 if need_upcast else p.dtype)
                )

            if F_curr[name].dtype != g2.dtype:
                g2 = g2.to(F_curr[name].dtype)
            F_curr[name] = alpha * F_curr[name] + (1.0 - alpha) * g2

        return F_curr

    # ===== [NEW] 规约/广播工具（可复用）=====
    def _reduce_scalar(self, value, reduction: str = "mean"):
        import torch
        t = torch.as_tensor(value, device=self.accelerator.device, dtype=torch.float32)
        t = self.accelerator.reduce(t, reduction=reduction)
        return float(t.item())

    def _reduce_tensor(self, t: "torch.Tensor", reduction: str = "mean"):
        return self.accelerator.reduce(t, reduction=reduction)

    def _broadcast_state_dict_(self, *dicts: dict, src: int = 0):
        import torch, torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        device = self.accelerator.device
        for d in dicts:
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    dist.broadcast(v.data, src=src)
                else:
                    t = torch.tensor(v, device=device)
                    dist.broadcast(t, src=src)
                    d[k] = int(t.item()) if isinstance(v, int) else float(t.item())

    def _assert_equal_across_processes(self, value, name="var"):
        import torch
        v = torch.tensor(float(value), device=self.accelerator.device)
        v_min = self.accelerator.reduce(v, reduction="min")
        v_max = self.accelerator.reduce(v, reduction="max")
        if self.accelerator.is_main_process and abs(v_max.item() - v_min.item()) > 1e-8:
            print(f"[WARN] {name} mismatch across ranks: min={v_min.item():.6g}, max={v_max.item():.6g}")

    def _synchronized_preconditioner(self, F_curr_safe: torch.Tensor, scale: float, device: torch.device) -> torch.Tensor:
        """
        输入：当前步 EMA Fisher 对角（同步梯度得到，天然一致）+ 层级缩放因子 scale
        输出：用于 v = g / preconditioner 的对角预条件器
        做法：bias-correct -> 分位数软抬升 -> 幂次（RMS风格） -> / scale
        """
        eps = 1e-12
        F_hat = F_curr_safe
        # 软抬升 floor
        q = float(getattr(self.args, "fisher_floor_quantile", 0.02))
        lam = float(getattr(self.args, "fisher_floor_mix", 0.7))
        floor_min = torch.as_tensor(getattr(self.args, "fisher_floor_min", 1e-12), device=F_hat.device, dtype=F_hat.dtype)
        pos = F_hat[F_hat > 0]
        if pos.numel() > 0:
            floor_q = torch.quantile(pos, q)
            floor = torch.maximum(floor_q, floor_min)
        else:
            floor = floor_min
        F_soft = torch.where(F_hat < floor, (1.0 - lam) * F_hat + lam * floor, F_hat)
        gamma = float(getattr(self.args, "precond_power", 0.5))
        precond = F_soft.pow(gamma)
        precond = precond / max(float(scale), 1e-8)
        return precond.clamp_min(1e-12).to(device)


    def _init_deepspeed(self):
        """初始化DeepSpeed引擎（修复属性名+配置格式转换）"""
        if self.args.deepspeed and not self.deepspeed:  # 先判断是否启用DeepSpeed
            from deepspeed import initialize

            # 1. 获取模型可训练参数
            model_parameters = [p for p in self.model.parameters() if p.requires_grad]

            # 2. 修复属性名：用 hf_deepspeed_config 替代 deepspeed_config，且转为字典
            deepspeed_config = self.args.hf_deepspeed_config.to_dict() if self.args.hf_deepspeed_config else None

            # 3. 初始化DeepSpeed引擎
            self.deepspeed_engine, optimizer, _, _ = initialize(
                model=self.model,
                model_parameters=model_parameters,
                config_params=deepspeed_config,  # 传入转换后的字典配置
                dist_init_required=False  # 联邦学习中已提前初始化分布式（若未初始化则设为True）
            )

            # 4. 对齐Trainer的优化器和DeepSpeed属性
            self.optimizer = optimizer
            self.deepspeed = self.deepspeed_engine  # 让Trainer识别已启用DeepSpeed

    @torch.no_grad()
    def _tripwire(self, name, g, v, d, f_past, theta, theta_bar, a, b, eta, R):
        eps = 1e-12
        # 一阶下降性：g·d 必须 <= 0
        dot_gd = (g * d).sum().item()
        if dot_gd > 1e-8:
            print(f"[TRIPWIRE] {name}: g·d={dot_gd:.3e} > 0  -> 方向可能反了/尺度错。")
        # 信任域内：r_new^2 <= R^2
        u = theta - theta_bar
        r2_now = (f_past * (u * u)).sum()
        r2_new = (f_past * ((u + d) * (u + d))).sum()
        if r2_new > (R * R + 1e-8):
            print(f"[TRIPWIRE] {name}: r2_new={r2_new.item():.6e} 超界 R^2={R * R:.3e}")
        # 条件数健康：a>=0；且别被少量坐标支配
        if a.item() < -1e-12:
            print(f"[TRIPWIRE] {name}: a<0 (数值错误)")
        v2 = (v * v).flatten()
        p99 = torch.quantile(v2, 0.99).item()
        med = torch.median(v2).item()
        if p99 > 1e4 * max(med, 1e-12):
            print(f"[TRIPWIRE] {name}: v^2 重尾非常严重 (p99/med={p99 / max(med, 1e-12):.2e})")
        # eta 合理：不是 NaN/Inf，也不远大于 LR 上限
        if not torch.isfinite(eta):
            print(f"[TRIPWIRE] {name}: eta 非有限")

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """添加详细调试信息的 training_step"""

        # 调试信息
        if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            logger.info(f"[DEBUG] training_step 开始, 进程: {self.accelerator.process_index}")

        try:
            # 原有的训练逻辑
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.accelerator.autocast():
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # 梯度累积
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # 反向传播
            self.accelerator.backward(loss)

            # 调试信息
            if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
                print(f"[DEBUG] 反向传播完成, loss: {loss.item():.6f}")

            # 关键：等待所有进程完成
            self.accelerator.wait_for_everyone()

            if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
                print(f"[DEBUG] 所有进程同步完成")

            return loss.detach()

        except Exception as e:
            logger.error(f"training_step 错误: {e}")
            # 返回一个虚拟的损失值避免卡死
            return torch.tensor(0.0, requires_grad=False)

    def train(
        self,
        task_id: int = 1,
        base_params: Dict[str, torch.Tensor] = None,
        cid: int = -1,
        **kwargs,
    ):  # type: ignore[override]

        if self.method == "lora_origin":
            return super().train(**kwargs)


        ########################adaLR###########################
        if not hasattr(self, "_eta_scale"):
            self._eta_scale = {}  # per-parameter EMA 的 s
        self._eta_scale_rho = getattr(self, "eta_scale_rho", 0.1)  # EMA 系数，0.05~0.2
        self._eta_smin = getattr(self, "eta_smin", 0.1)  # 缩放下界
        self._eta_smax = getattr(self, "eta_smax", 10.0)  # 缩放上界


        if self.method == "adaptive" and task_id == 1:
            dataloader = self.get_train_dataloader()
            dataloader = self.accelerator.prepare(dataloader)
            # 验证：不同进程的数据集长度应不同（分片后）
            print(f"进程 {self.accelerator.process_index} 数据加载器长度: {len(dataloader)}")
            return super().train(**kwargs)

        elif self.method == "adaptive" and task_id > 1:
            state_has_history = self.continual_state.has_valid_history() if self.continual_state is not None else False
            # 如果该节点之前从未被训练过（无历史）：走一次普通训练 + Fisher
            if self.continual_state is None or base_params is None or (
                    self.continual_state is not None and not state_has_history
            ):
                super().train(**kwargs)
                state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                delta = {k: base_params[k] - state_dict[k].detach().cpu() for k in base_params}
                F_client = compute_fisher(self.model, self.get_train_dataloader())
                F_client = {k: F_client.get(k, torch.zeros_like(base_params[k])) for k in base_params}
                theta_last = {k: state_dict[k].detach().cpu() for k in F_client.keys()}
                return delta, F_client, theta_last

            # --------------Train---------------
            use_adaptive_logic = True
            logger.info(f"===== 开始训练 | 模式：{'自适应' if use_adaptive_logic else '基线'} | Task {task_id} =====")

            # model = self.model
            model = self.accelerator.unwrap_model(self.model)
            dataloader = self.get_train_dataloader()
            dataloader = self.accelerator.prepare(dataloader)
            device = self.accelerator.device

            # device = next(model.parameters()).device
            model.train()
            self.eta_per_layer = defaultdict(list)
            # unwrapped_model.train()

            # 启用激活探针
            self.enable_activation_probe = bool(getattr(self.args, "enable_activation_probe", True))
            # === [新增] 在进入自适应训练循环前安装激活探针 ===
            if self.accelerator.is_main_process and getattr(self.args, "enable_activation_probe", True):
                self._act_probe = _ActivationProbe(
                    topk=5,
                    ratio_trigger=float(getattr(self.args, "actprobe_ratio_trigger", 30.0)),
                    print_limit=int(getattr(self.args, "actprobe_print_limit", 10)),
                )
                named_modules = dict(model.named_modules())
                lora_param_names = [n for (n, p) in model.named_parameters() if ("lora_A" in n or "lora_B" in n)]
                self._act_probe.register(model, named_modules, lora_param_names)


            # 工程
            F_curr = None
            bar_F_raw = None
            bar_theta = {}
            r2, r2_start = {}, {}
            B_round, conf = {}, {}

            if use_adaptive_logic:

                F_curr = {
                    n: torch.zeros_like(p, device=device)
                    for n, p in model.named_parameters()
                    if p.requires_grad
                }

                # Load old task
                bar_F_raw = {}
                bar_theta = {}
                # 从ContinualState获取归一化后的bar_F和bar_B
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue

                    if self.continual_state.has_valid_history():
                        f_past_norm = self.continual_state.get_f_past(name)
                        bar_F_raw[name] = f_past_norm.to(device)
                    else:
                        bar_F_raw[name] = torch.zeros_like(p, device=device)

                    # # 2. 计算bar_theta（bar_B已归一化，直接用）
                    # if name in self.continual_state.bar_B and name in bar_F_raw:
                    #     bar_theta[name] = self.continual_state.bar_B[name].to(device) / (bar_F_raw[name] + 1e-8)
                    # else:
                    #     bar_theta[name] = torch.zeros_like(p, device=device)

                    bar_theta[name] = self.continual_state.theta_last[name].to(device)

                # for name in bar_theta:
                #     # 获取两个张量（确保设备一致）
                #     theta_last = self.continual_state.theta_last[name].to(device)
                #     current_theta = bar_theta[name]
                #
                #     # 检查是否所有元素数值完全相同（atol和rtol设为0表示严格相等）
                #     if torch.allclose(theta_last, current_theta, atol=1e-6, rtol=1e-6):
                #         print(f"{name}: yizhi")
                #     else:
                #         print(f"{name}: bu yi zhi")

                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.adam_states[name] = {
                            'm': torch.zeros_like(p, device=p.device),  # 一阶矩（动量）
                            'v': torch.zeros_like(p, device=p.device),  # 二阶矩（平方梯度累积）
                            't': 0  # 时间步（用于偏差修正）
                        }

                # 初始化半径和收益状态（关键：需要跨进程同步）
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue

                    # r²_ℓ = ‖θ^(ℓ) - bar_theta^(ℓ)‖²_{bar_F^(ℓ)}
                    f_past = bar_F_raw[name]
                    hist_mean = torch.clamp(f_past.mean(), min=1e-12)
                    f_past_eff = f_past / hist_mean

                    theta_curr = p  # 当前层参数θ^(ℓ)
                    theta_past = bar_theta[name]

                    # 计算核心变量
                    # delta_theta = theta_curr - theta_past  # 偏移量：θ - bar_θ
                    # delta_sq_mean = torch.mean(delta_theta.pow(2)).item()  # 偏移平方的均值（反映整体偏移程度）
                    # bar_F_mean = torch.mean(f_past).item()  # bar_F的均值（反映历史累积量级）
                    # r_sq = ((delta_theta.pow(2) * f_past).sum()).item()  # 当前r²_ℓ的值
                    # r_sq_threshold = self.radius ** 2  # R²（半径阈值）

                    # 打印关键信息（使用logger或print，建议用logger）
                    # logger.info(
                    #     f"任务{task_id} | 层{name} | "
                    #     f"偏移平方均值: {delta_sq_mean:.6f} | "
                    #     f"bar_F均值: {bar_F_mean:.6f} | "
                    #     f"r²_ℓ: {r_sq:.6f} | "
                    #     f"R²阈值: {r_sq_threshold:.6f} | "
                    #     f"Delta: {r_sq_threshold - r_sq:.6f}"  # 直接显示Delta值
                    # )
                    if name == 'base_model.model.encoder.block.10.layer.0.SelfAttention.q.lora_B.default.weight':
                        pass
                    r2[name] = ((theta_curr - theta_past).pow(2) * f_past_eff).sum()
                    # r2[name] = ((theta_curr - theta_past).pow(2) * f_past).sum()

                    # logger.info(f"任务3 初始状态 | 层{name} | "
                    #             f"delta_theta均值: {torch.mean(torch.abs(theta_curr - theta_past)).item():.6f} | "
                    #             f"bar_F均值: {torch.mean(f_past).item():.6f} | "
                    #             f"初始r²_ℓ: {r2[name].item():.6f} | "
                    #             f"R²阈值: {self.radius ** 2:.6f} | "
                    #             f"初始Delta: {self.radius ** 2 - r2[name].item():.6f}")

                    # r²_ℓ,start = r²_ℓ（记录初始半径）
                    r2_start[name] = r2[name].clone()
                    # B^round_ℓ = 0（收益累计）
                    B_round[name] = torch.tensor(0.0, device=device)
                    # conf_ℓ = 0（冲突次数统计）
                    conf[name] = 0

            else:
                # 基线模式：仅初始化“手动更新必要参数”（无多余代码）
                lr_base = self.args.learning_rate  # 基线用固定学习率（与自适应lr_cap一致）
                # 基线无需bar_F/bar_B，仅记录初始参数用于计算delta
                for k, p in model.named_parameters():
                    if p.requires_grad:
                        bar_theta[k] = p.detach().clone()

            if getattr(self.accelerator.state, "num_processes", 1) > 1:
                try:
                    self._broadcast_state_dict_(bar_theta, r2, r2_start, B_round, conf)
                    if isinstance(F_curr, dict) and len(F_curr) > 0:
                        self._broadcast_state_dict_(F_curr)
                    if isinstance(bar_F_raw, dict) and len(bar_F_raw) > 0:
                        self._broadcast_state_dict_(bar_F_raw)
                except Exception as _:
                    pass

            num_epochs = int(self.args.num_train_epochs)
            steps_per_epoch = len(dataloader)
            actual_steps = 0  # 真实已处理的 mini-batch 数（用于 p_round 分母）
            #eta_save = []
            round_s = []
            Delta_s = []
            #eta_ori = []
            # 在训练循环开始前添加监控变量
            gradient_monitor = {
                'grad_norms': [],
                'fisher_values': [],
                'preconditioner_values': [],
                'v_norms': [],
                'eta_values': [],
                'loss_values': [],
                'delta_values': []
            }

            # /*-------------Begin train------------
            for epoch in range(num_epochs):
                for step, batch in enumerate(dataloader):

                    eta_save_step, eta_ori_step = [], []
                    eta_d = []
                    # batch_start = time.perf_counter()
                    actual_steps += 1

                    if getattr(self, "_act_probe", None) is not None:
                        self._act_probe.begin_batch()

                    batch = self._prepare_inputs(batch)

                    # 清梯度
                    model.zero_grad(set_to_none=True)


                    # 前向
                    with self.compute_loss_context_manager(), self.accelerator.autocast():
                        outputs = model(**batch)
                        loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs["loss"]

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    # 反向传播
                    # loss.backward()
                    self.accelerator.backward(loss)

                    # 打印loss结果
                    raw_loss = loss.detach().item() * self.args.gradient_accumulation_steps
                    # 输出日志：包含epoch、step、当前batch的loss
                    if self.accelerator.is_main_process:
                        logger.info(
                            f"Task {task_id} | Epoch [{epoch + 1}/{num_epochs}] | Batch [{step + 1}/{steps_per_epoch}] | "
                            f"Batch Loss: {raw_loss:.6f}"
                        )


                    # before_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

                    ###########################################
                    # 第二步：调用独立函数，在线更新Fisher
                    ###########################################
                    # 调用函数：传入模型、当前batch、F_curr、EMA系数、设备
                    # F_curr, F_batch = update_online_fisher(
                    #     model=model,
                    #     batch=batch,
                    #     F_curr=F_curr,
                    #     alpha_ema=self.alpha,
                    #     device=device
                    # )
                    F_curr = self._synchronized_ema_fisher(model, F_curr, alpha=self.alpha)
                    # global_norm = None
                    # if any(p.grad is not None for p in unwrapped_model.parameters()):
                    #     s = 0.0
                    #     for p in unwrapped_model.parameters():
                    #         if p.grad is not None:
                    #             s += float(p.grad.detach().float().norm().item()) ** 2
                    #     global_norm = (s ** 0.5)
                    #
                    # # === [新增] 如有需要，打印可疑层的详细激活/反传统计 ===
                    # # if getattr(self, "_act_probe", None) is not None:
                    # #     clip_norm = float(getattr(self.args, "clip_grad_norm", 1.0))
                    # #     self._act_probe.report_if_needed(model, logger, global_norm, clip_norm)



                    # 监控点1：梯度范数检查
                    grad_norms = {}
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            grad_norm = torch.norm(p.grad).item()
                            grad_norms[name] = grad_norm
                            if grad_norm > 1000:  # 梯度爆炸阈值
                                logger.warning(f"梯度爆炸检测: {name}, 梯度范数: {grad_norm}")

                    # 监控点2：Fisher值检查
                    fisher_values = {}
                    for name in F_curr:
                        fisher_mean = torch.mean(F_curr[name]).item()
                        fisher_values[name] = fisher_mean
                        if fisher_mean < 1e-12 or fisher_mean > 1e6:
                            logger.warning(f"Fisher值异常: {name}, 均值: {fisher_mean}")

                    # 将监控数据保存
                    if self.accelerator.is_main_process:
                        gradient_monitor['grad_norms'].append(grad_norms)
                        gradient_monitor['fisher_values'].append(fisher_values)
                        gradient_monitor['loss_values'].append(raw_loss)
                    # after_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                    # # 验证梯度是否一致
                    # for name in before_grads:
                    #     if not torch.allclose(before_grads[name], after_grads[name]):
                    #         print(f"梯度不一致: {name}")
                    #     else:
                    #         print(f"梯度一致: {name}")

                    #_tripwire()

                    ###########################################
                    # 第三步：自适应学习率计算
                    ###########################################
                    # 自适应学习率
                    with torch.no_grad():
                        if use_adaptive_logic:
                            # F_norm_per_param = {}
                            for name, p in model.named_parameters():
                                if not p.requires_grad or p.grad is None:
                                    continue

                                # Step 1
                                g = p.grad

                                # # 添加梯度异常检测
                                # if torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)):
                                #     logger.warning(f"梯度包含NaN或Inf: {name}")
                                #     continue
                                #
                                # # 检查梯度范数是否异常
                                # grad_norm = torch.norm(g)
                                # if grad_norm > 10:  # 设置合理的阈值
                                #     logger.warning(f"梯度范数异常: {name}, 范数: {grad_norm:.6f}")
                                #     # 应用梯度裁剪
                                #     g = torch.clamp(g, -10.0, 10.0)
                                #     p.grad = g  # 更新梯度




                                # F_batch = g * g
                                # F_curr[name] = self.alpha * F_curr[name] + (1 - self.alpha) * F_batch
                                F_curr_safe = F_curr[name]
                                # Step 2
                                # v = g / F_curr_safe
                                # 2. 过去任务Fisher（bar_F^(ℓ)）
                                f_past = bar_F_raw.get(name, torch.zeros_like(p))
                                if torch.all(f_past <= 0) or f_past.mean() <= 1e-12:
                                    f_past_eff = torch.ones_like(f_past)  # ← 关键：未见层按单位权重
                                else:
                                    f_past_eff = f_past / f_past.mean()


                                # Fisher 信息的绝对量级不影响 “参数重要性的区分”，仅相对比例有意义

                                # if torch.any(f_past > 0):
                                #     f_past_mean = torch.mean(f_past)
                                #     if f_past_mean > 0:
                                #         f_past = f_past / f_past_mean  # 归一化，让f_past均值为1

                                scale = 1.0
                                if torch.is_tensor(f_past_eff) and torch.numel(f_past_eff) > 0 and torch.any(f_past_eff > 0):
                                    curr_mean = torch.mean(F_curr_safe).item()
                                    hist_mean = torch.mean(f_past_eff).item()
                                    if hist_mean > 0:
                                        prev_scale = self._eta_scale.get(name, 1.0)
                                        ratio = curr_mean / hist_mean
                                        ratio = max(self.fisher_floor, ratio)

                                        scale = (1 - self._eta_scale_rho) * prev_scale + self._eta_scale_rho * ratio
                                        scale = max(self._eta_smin, min(self._eta_smax, scale))
                                        self._eta_scale[name] = scale

                                # --- Robust preconditioning (bias-correct + adaptive floor + power) ---
                                t_eff = max(1, actual_steps)
                                eps = 1e-12
                                F_hat = F_curr_safe / (1.0 - (self.alpha ** t_eff) + eps)  # bias-correct EMA

                                # 2) 用“目标分位数”确定 floor，保证只有底部少量坐标被抬升
                                #    - fisher_floor_quantile: 目标抬升比例（默认 0.02 = 2%）
                                #    - fisher_floor_min: 全局极小兜底，避免完全为 0（推荐 1e-12~1e-10，而不是 1e-5）
                                q = float(getattr(self.args, "fisher_floor_quantile", 0.02))
                                floor_min = torch.as_tensor(getattr(self.args, "fisher_floor_min", 1e-12),
                                                            device=F_hat.device, dtype=F_hat.dtype)
                                pos = F_hat[F_hat > 0]
                                if pos.numel() > 0:
                                    floor_q = torch.quantile(pos, q)
                                    floor = torch.maximum(floor_q, floor_min)
                                else:
                                    floor = floor_min

                                # 3) 软抬升（mix），而不是硬 clamp：保留次序与差异
                                #    fisher_floor_mix ∈ [0,1]：0=不抬升，1=等价硬 clamp；建议 0.5~0.8
                                lam = float(getattr(self.args, "fisher_floor_mix", 0.7))
                                F_soft = torch.where(F_hat < floor, (1.0 - lam) * F_hat + lam * floor, F_hat)

                                # 4) 幂次预条件（RMS 风格），继续抑制长尾
                                gamma = float(getattr(self.args, "precond_power", 0.5))  # 0.5~0.75
                                # preconditioner = (F_soft.pow(gamma)) / scale

                                preconditioner = self._synchronized_preconditioner(F_curr_safe, scale, device)
                                v = g / (preconditioner + eps)

                                # 计算v的均值和标准差（统计分布特征）
                                v_mean = torch.mean(v).item()
                                v_std = torch.std(v).item()

                                # 定义“合理范围”为 均值 ± 3倍标准差（倍数可根据场景调整，如2或4）
                                upper_bound = v_mean + 3 * v_std
                                lower_bound = v_mean - 3 * v_std

                                # 对超出范围的部分进行“软缩放”：超出的部分乘以衰减系数（如0.5）
                                v = torch.where(v > upper_bound, upper_bound + 0.5 * (v - upper_bound), v)
                                v = torch.where(v < lower_bound, lower_bound + 0.5 * (v - lower_bound), v)


                                # （可选）轻量监控：抬升比例与 floor 大小
                                if (t_eff % 50 == 0) and (F_hat.numel() >= 16):
                                    lifted_frac = (F_hat < floor).float().mean().item()
                                    try:
                                        med = F_hat.median().item()
                                    except Exception:
                                        med = float('nan')
                                    print(f"[Fisher-softfloor] q={q:.3f}, floor={float(floor):.3e}, mix={lam:.2f}, "
                                          f"lifted={lifted_frac:.3f}, min/med/max={F_hat.min().item():.3e}/{med:.3e}/{F_hat.max().item():.3e}")

                                ####################################################
                                # 动态缩放v的量级：与g的标准差对齐.保留 Fisher 预条件对 “不同参数重要性” 的区分（v 的元素间比例不变），
                                # 仅调整 v 的整体量级，使其与 g 的更新幅度匹配。
                                # std_g = torch.std(g)
                                # std_v = torch.std(v)
                                # std_v = torch.clamp(std_v, min=1e-8)  # 防止除零
                                # scale_v = std_g / std_v  # 计算缩放因子
                                mean_g = torch.mean(torch.abs(g))
                                # mean_v = torch.mean(torch.abs(v))
                                mean_v = torch.median(torch.abs(v))
                                mean_v = torch.clamp(mean_v, min=1e-8)  # 防止除零
                                scale_v = mean_g / mean_v  # 用均值比计算缩放因子
                                v_scaled = v * scale_v  # 缩放后的v，量级与g一致
                                ####################################################

                                # scale_tensor = torch.as_tensor(scale, device=device, dtype=p.dtype)
                                # v_alg 对应“算法中的 v_B^(ℓ)”，后续 a/b/Q 等均基于该无缩放版本计算，
                                # 只在最终更新时再与 scale_tensor 结合，保持量纲一致。
                                # v_alg = v_scaled / scale_tensor
                                v_alg = v_scaled

                                # 3. 当前参数与过去最优参数的差：Δθ = θ^(ℓ) - bar_theta^(ℓ)
                                delta_theta = p - bar_theta.get(name, torch.zeros_like(p))

                                # 4. a_B^(ℓ) = (v_B^(ℓ))^T * bar_F^(ℓ) * v_B^(ℓ)
                                # a_raw_alg = (v_alg * f_past * v_alg).sum()

                                # winsorized a using capped v^2 to suppress heavy tails
                                v2 = (v_alg * v_alg).detach()
                                try:
                                    cap = torch.quantile(v2, 0.99)
                                except Exception:
                                    cap = v2.mean() + 3.0 * v2.std()
                                v2_clip = torch.clamp(v2, max=cap)


                                # a_raw_alg = (v2_clip * f_past_eff).sum()
                                # b_raw_alg = (v_alg * f_past_eff * delta_theta).sum()
                                a_local = (v2_clip * f_past_eff).sum()  # v^T \bar F v
                                b_local = (v_alg * f_past_eff * delta_theta).sum()


                                # a_raw_alg = (v2_clip * f_past).sum()
                                # b_raw_alg = (v_alg * f_past * delta_theta).sum()
                                # 6. 半径余量 Δ_ℓ = R² - r²_ℓ
                                if name == 'base_model.model.encoder.block.10.layer.0.SelfAttention.q.lora_B.default.weight':
                                    pass
                                # Delta = (self.radius ** 2) - r2[name]
                                Delta_local = (self.radius ** 2) - r2[name]
                                Q_local = (g * v_alg).sum()

                                a_raw_alg = torch.as_tensor(self._reduce_scalar(a_local, reduction="mean"),
                                                            device=device, dtype=p.dtype)
                                b_raw_alg = torch.as_tensor(self._reduce_scalar(b_local, reduction="mean"),
                                                            device=device, dtype=p.dtype)
                                Delta = torch.as_tensor(self._reduce_scalar(Delta_local, reduction="mean"),
                                                        device=device, dtype=p.dtype)
                                Q_alg = torch.as_tensor(self._reduce_scalar(Q_local, reduction="mean"), device=device,
                                                        dtype=p.dtype)



                                Delta_s.append(Delta)

                                if Delta.item() <= self.tau:
                                    eta_alg = torch.zeros(1, device=device, dtype=p.dtype).squeeze()
                                    eta = eta_alg
                                else:
                                    # a_safe = torch.clamp(a_raw_alg, min=1e-12)
                                    a_safe = torch.clamp(a_raw_alg, min=v2_clip.mean() * v2_clip.numel() * 1e-4)
                                    Delta_eff = torch.clamp(Delta - self.tau, min=0.0)

                                # 4. a_B^(ℓ) = (v_B^(ℓ))^T * bar_F^(ℓ) * v_B^(ℓ)
                                # a_raw_alg = (v_alg * f_past * v_alg).sum()

                                # winsorized a using capped v^2 to suppress heavy tails
                                # v2 = (v_alg * v_alg).detach()
                                # try:
                                #     cap = torch.quantile(v2, 0.99)
                                # except Exception:
                                #     cap = v2.mean() + 3.0 * v2.std()
                                # v2_clip = torch.clamp(v2, max=cap)
                                # a_raw_alg = (v2_clip * f_past).sum()
                                #
                                #
                                # # 5. b_B^(ℓ) = (v_B^(ℓ))^T * bar_F^(ℓ) * Δθ
                                # b_raw_alg = (v_alg * f_past * delta_theta).sum()
                                # # 6. 半径余量 Δ_ℓ = R² - r²_ℓ
                                # Delta = (self.radius ** 2) - r2[name]
                                # Delta_s.append(Delta)
                                #
                                # if Delta.item() <= self.tau:
                                #     eta_alg = torch.zeros(1, device=device, dtype=p.dtype).squeeze()
                                #     eta = eta_alg
                                # else:
                                #     a_safe = torch.clamp(a_raw_alg, min=1e-12)
                                #     Delta_eff = torch.clamp(Delta - self.tau, min=0.0)

                                    # --------------------------
                                    # 算法步骤8：冲突判断与步长计算
                                    # --------------------------
                                    if b_raw_alg.item() < 0.0:  # 冲突层：b_B^(ℓ) < 0（新旧任务方向冲突）
                                        # 闭式步长公式（严格按算法）：
                                        #term_u = b_raw_alg - self.sigma
                                        term_u = b_raw_alg - torch.as_tensor(self.sigma, device=device, dtype=p.dtype)
                                        discriminant = term_u ** 2 + self.beta * a_safe * Delta_eff
                                        discriminant = torch.clamp(discriminant, min=0.0)  # 确保开方非负
                                        xx = (term_u + torch.sqrt(discriminant))
                                        yy = (self.beta * a_safe)
                                        eta_closed = xx / yy
                                        eta_ori_step.append(eta_closed.detach().float().mean().cpu().clone())
                                        # 步长上限：不超过初始学习率 η₀
                                        eta_alg = torch.minimum(
                                            eta_closed,
                                            torch.tensor(self.args.learning_rate, device=device, dtype=p.dtype),
                                        )
                                        if eta_alg.item() > 0:
                                            conf[name] += 1
                                    else:
                                        eta_trust = torch.sqrt(Delta_eff / a_safe)
                                        eta_save_step.append(eta_trust.detach().float().mean().cpu().clone())
                                        eta_alg = torch.clamp(
                                            eta_trust,
                                            max=torch.as_tensor(self.args.learning_rate, device=device, dtype=p.dtype),
                                        )

                                    # 确保 eta 为标量并非 NaN
                                    if torch.isnan(eta_alg):
                                        eta_alg = torch.zeros(1, device=device, dtype=p.dtype).squeeze()

                                    eta = eta_alg
                                    eta_d.append(eta)

                                if step % 1 == 0:
                                    self._assert_equal_across_processes(eta_alg.item(), name="eta_alg")

                                # --------------------------
                                # 算法步骤9：累计收益 B^round_ℓ
                                # --------------------------
                                # 二阶收益 Q_B^(ℓ) = (g_B^(ℓ))^T * (F_curr^(ℓ))^(-1) * g_B^(ℓ)（等价于 g·v）
                                # Q_alg = torch.sum(g * v_alg)
                                # 收益计算：max{0, (η - 0.5η²) * Q}
                                gain_local = (eta_alg - 0.5 * eta_alg * eta_alg) * Q_alg
                                gain_local = torch.clamp(gain_local, min=0.0)
                                gain = torch.as_tensor(self._reduce_scalar(gain_local, reduction="mean"), device=device,
                                                       dtype=p.dtype)

                                B_round[name] = B_round[name] + gain
                                if "model.encoder.block.1.layer.0.SelfAttention.v.lora_A.default" in name:
                                    round_s.append(B_round[name])
                                # --------------------------
                                # 算法步骤10：更新马氏半径 r²_ℓ
                                # --------------------------
                                if "base_model.model.decoder.block.3.layer.0.SelfAttention.q.lora_B.default.weight" in name:
                                    pass

                                if "base_model.model.encoder.block.3.layer.0.SelfAttention.q.lora_A.default.weight" in name:
                                    pass
                                r2[name] = r2[name] - 2.0 * eta_alg * b_raw_alg + (eta_alg * eta_alg) * a_raw_alg
                                if r2[name] > 1:
                                    pass
                                # --------------------------
                                # 算法步骤11：更新参数 θ^(ℓ) = θ^(ℓ) - η·v_B^(ℓ)
                                # --------------------------
                                if eta.item() != 0.0:
                                    with self.accelerator.no_sync(model):
                                        p.add_(-eta * v_scaled)  # 修正：使用缩放后的v_scaled
                                    # p.add_(-self.args.learning_rate * g)
                                    # state = self.adam_states[name]
                                    # state['t'] += 1
                                    # t = state['t']
                                    # state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * v_scaled
                                    # state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (v_scaled ** 2)
                                    # # 偏差修正（抵消初始时刻动量和平方和接近0的影响）
                                    # m_hat = state['m'] / (1 - self.beta1 ** t)  # 修正后的一阶矩
                                    # v_hat = state['v'] / (1 - self.beta2 ** t)  # 修正后的二阶矩
                                    # update = eta * (m_hat / (torch.sqrt(v_hat) + self.eps))
                                    # p.add_(-update)
                                # p.add_(-self.args.learning_rate * g)

                                if "lora_" in name:
                                    self.eta_per_layer[name].append(eta.item())


                        else:
                            # --------------------------
                            # 基线模式：仅保留“手动SGD更新”（无任何自适应逻辑）
                            # --------------------------
                            for name, p in unwrapped_model.named_parameters():
                                if not p.requires_grad or p.grad is None:
                                    continue
                                # 基线：无Fisher归一化，v=原始梯度；无动态步长，eta=固定学习率
                                v_baseline = p.grad
                                eta_baseline = lr_base
                                # 与自适应模式相同的手动更新方式（保证公平性）
                                p.add_(-eta_baseline * v_baseline)

                    self.accelerator.wait_for_everyone()

                    s = 0
                    for i in eta_d:
                        if i == self.args.learning_rate:
                            s += 1
                    if len(eta_d) == s:
                        print('这个batch所有层都是统一学习率')

                    if step % 1 == 0:
                        health_report = monitor_gradient_health(step, model, F_curr, bar_F_raw, r2, B_round)

                        if health_report['issues']:
                            logger.warning(f"步骤 {step} 健康检查发现问题:")
                            for issue in health_report['issues']:
                                logger.warning(f"  - {issue}")

                        # 记录关键指标
                        # logger.info(f"步骤 {step} 关键指标:")
                        # for metric, value in health_report['metrics'].items():
                        #     if 'grad_norm' in metric or 'b_round' in metric:
                        #         logger.info(f"  {metric}: {value:.6f}")

            # 辅助函数：计算通信数据量（字节）和时间（秒）
            def calculate_comm_metrics(param_dict: Dict[str, torch.Tensor]) -> Tuple[float, int]:
                total_bytes = 0
                for param in param_dict.values():
                    # 计算参数总字节数（元素数 × 每个元素字节数，float32为4字节）
                    total_bytes += param.numel() * param.element_size()
                if total_bytes == 0:
                    return 0.0, 0
                # 转换带宽单位：MB/s → 字节/秒（1MB = 1024×1024字节）
                bandwidth_bytes_per_sec = comm_bandwidth * 1024 * 1024
                # 通信时间 = 数据传输时间 + 固定开销
                comm_time = (total_bytes / bandwidth_bytes_per_sec) + comm_fixed_cost
                return comm_time, total_bytes

            if use_adaptive_logic:
                # ====== Step-3：选层与通信计算 ======
                F_round = {n: 0.5 * torch.clamp(r2[n] - r2_start[n], min=0.0) for n in r2}
                p_round = {n: conf[n] / max(actual_steps, 1) for n in conf}  # 用真实步数计算冲突占比

                # 背包算法选层
                values, costs, names = [], [], []
                for name in B_round:
                    val = (B_round[name] - self.lambda_conf * p_round[name] * F_round[name]).item()
                    values.append(val)
                    costs.append(max(int(self.layer_costs.get(name, 1)), 1))
                    names.append(name)
                #selected = _knapsack(values, costs, self.comm_budget)

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


                # Test
                # selected_flags = [True] * len(names)
                selection_map = {n: f for n, f in zip(names, selected_flags)}




                # 1) 自适应模式：仅传输选中层的delta
                state_dict = model.state_dict()
                adaptive_delta = {}
                for name in base_params:
                    local_param = state_dict[name].detach().cpu()
                    # 仅选中的层传输真实delta，未选中的层传输零（不实际传输，仅占位）
                    #adaptive_delta[name] = (base_params[name] - local_param) if sel else torch.zeros_like(local_param)
                    if selection_map.get(name, True):
                        adaptive_delta[name] = base_params[name] - local_param
                    else:
                        adaptive_delta[name] = torch.zeros_like(local_param)

                delta = adaptive_delta

                # # 计算自适应模式通信时间
                # adaptive_comm_time, adaptive_bytes = calculate_comm_metrics(
                #     {k: v for k, v in delta.items() if torch.count_nonzero(v).item() > 0}
                # )
                #
                # # 2) 计算基线模式通信时间（传输所有可训练参数的delta）
                # baseline_delta = {}
                # for name, p in model.named_parameters():
                #     if name in base_params and p.requires_grad:
                #         local_param = state_dict[name].detach().cpu()
                #         baseline_delta[name] = base_params[name] - local_param
                # baseline_comm_time, baseline_bytes = calculate_comm_metrics(baseline_delta)
                #
                # # 计算节省的通信时间
                # save_time = baseline_comm_time - adaptive_comm_time
                # save_ratio = (save_time / baseline_comm_time * 100.0) if baseline_comm_time > 0 else 0.0
                #
                # # 日志输出通信统计
                # logger.info(
                #     f"通信统计 - 自适应模式: {adaptive_bytes / (1024 * 1024):.2f} MB, 时间: {adaptive_comm_time:.4f}秒\n"
                #     f"通信统计 - 基线模式: {baseline_bytes / (1024 * 1024):.2f} MB, 时间: {baseline_comm_time:.4f}秒\n"
                #     f"通信节省时间: {save_time:.4f}秒 (节省比例: {save_ratio:.2f}%)"
                # )

                # 3) 本轮Fisher计算
                F_client = {
                    k: F_curr.get(k, torch.zeros_like(state_dict[k])).detach().cpu()
                    for k in base_params
                }
            else:
                # 基线模式：传输所有可训练参数的delta
                state_dict = unwrapped_model.state_dict()
                baseline_delta = {}
                for name, p in unwrapped_model.named_parameters():
                    if name in base_params and p.requires_grad:
                        local_param = state_dict[name].detach().cpu()
                        baseline_delta[name] = base_params[name] - local_param
                delta = baseline_delta

                # # 计算基线模式通信时间（自适应模式未启用，节省时间为0）
                # baseline_comm_time, baseline_bytes = calculate_comm_metrics(baseline_delta)
                # adaptive_comm_time = 0.0
                #
                # logger.info(
                #     f"通信统计 - 基线模式: {baseline_bytes / (1024 * 1024):.2f} MB, 时间: {baseline_comm_time:.4f}秒"
                # )

                F_client = {k: torch.zeros_like(v) for k, v in base_params.items()}

            # 4) 本轮结束时该客户端的θ*（只取可训练/LoRA参数）
            theta_last = {k: state_dict[k].detach().cpu() for k in base_params.keys()}

            # 返回值新增通信节省时间
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
