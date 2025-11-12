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
        self.adamw_weight_decay = getattr(self.args, "weight_decay", 0.01)
        self.eta_shrink = float(getattr(self.args, "eta_shrink", 1.0))
        self.trust_radius_shrink = float(getattr(self.args, "trust_radius_shrink", 1e-3))
        self.beta_mul = float(getattr(self.args, "beta_mul", 1.0))

    # ---- LoRA ΔW_eff 工具：把同名前缀的 lora_A / lora_B 配对并计算 ||B@A - B_bar@A_bar||_F ----
    @torch.no_grad()
    def _compute_delta_weff_norm(self, model_plain: nn.Module, bar_theta: Dict[str, torch.Tensor]) -> float:
        named_params = dict(model_plain.named_parameters())
        named_modules = dict(model_plain.named_modules())

        def _pref(n: str) -> str:
            # 形如 "...SelfAttention.q.lora_A.default.weight" -> 去掉 ".lora_A.default.weight"
            return n.split(".lora_")[0] if ".lora_" in n else ""

        # 收集 A/B
        A_map, B_map, scale_map = {}, {}, {}
        for n, p in named_params.items():
            if ".lora_A." in n and n.endswith(".weight"):
                A_map[_pref(n)] = p.detach().float()
            elif ".lora_B." in n and n.endswith(".weight"):
                B_map[_pref(n)] = p.detach().float()
        # 取 scaling
        for base in set(list(A_map.keys()) + list(B_map.keys())):
            m = named_modules.get(base, None)
            s = 1.0
            if m is not None and hasattr(m, "scaling"):
                try:
                    s = float(getattr(m, "scaling"))
                except Exception:
                    s = 1.0
            scale_map[base] = s
        # 相同前缀配对并求和 Frobenius
        total_sq = 0.0
        for base in A_map.keys():
            if base not in B_map:
                continue
            A_cur, B_cur, sc = A_map[base], B_map[base], scale_map.get(base, 1.0)
            # 取历史 A/B
            A_key = f"{base}.lora_A.default.weight"
            B_key = f"{base}.lora_B.default.weight"
            A_bar = bar_theta.get(A_key, torch.zeros_like(A_cur)).float()
            B_bar = bar_theta.get(B_key, torch.zeros_like(B_cur)).float()
            # ΔW_eff = sc * (B_cur@A_cur - B_bar@A_bar)
            curr = torch.matmul(B_cur, A_cur)
            past = torch.matmul(B_bar, A_bar)
            dW = sc * (curr - past)
            total_sq += float((dW * dW).sum().item())
        return float(total_sq ** 0.5)



    # ------------------------------------------------------------------
    # Utilities to safely interact with the (optional) Accelerator handle
    # ------------------------------------------------------------------
    def _current_accelerator(self) -> Optional[Accelerator]:
        return getattr(self, "accelerator", None)

    def _unwrap_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        accel = self._current_accelerator()
        target = model if model is not None else self.model
        if accel is not None:
            return accel.unwrap_model(target)
        return target


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


        unwrapped = self.accelerator.unwrap_model(model)
        world_size = max(1, getattr(self.accelerator.state, "num_processes", 1))

        def _canon(name: str) -> str:
            return name[7:] if name.startswith("module.") else name




        for name, p in unwrapped.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue

            name_c = _canon(name)
            g = p.grad.detach()
            need_upcast = (g.dtype in (torch.float16, torch.bfloat16))
            g2 = (g.float() * g.float()) if need_upcast else (g * g)

            if ddp_avg_of_squares and world_size > 1:
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(g2, op=dist.ReduceOp.SUM)
                    g2 = g2 / float(world_size)

            if name_c not in F_curr:
                F_curr[name_c] = torch.zeros_like(
                    p, device=p.device, dtype=(torch.float32 if need_upcast else p.dtype)
                )

            if F_curr[name_c].dtype != g2.dtype:
                g2 = g2.to(F_curr[name_c].dtype)
            F_curr[name_c] = alpha * F_curr[name_c] + (1.0 - alpha) * g2

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
            # # 原有的训练逻辑
            # model.train()
            # inputs = self._prepare_inputs(inputs)
            #
            # with self.accelerator.autocast():
            #     outputs = model(**inputs)
            #     loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            #
            # # 梯度累积
            # if self.args.gradient_accumulation_steps > 1:
            #     loss = loss / self.args.gradient_accumulation_steps
            #
            # # 反向传播
            # self.accelerator.backward(loss)
            #
            # # 调试信息
            # if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            #     print(f"[DEBUG] 反向传播完成, loss: {loss.item():.6f}")

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



            # # 关键：等待所有进程完成
            # self.accelerator.wait_for_everyone()
            #
            # if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            #     print(f"[DEBUG] 所有进程同步完成")

            # return loss.detach()

        # except Exception as e:
        #     logger.error(f"training_step 错误: {e}")
        #     # 返回一个虚拟的损失值避免卡死
        #     return torch.tensor(0.0, requires_grad=False)

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
            logger.info(f"[Task {task_id}] 开始调用父类train方法")
            print(
                "[whoami] get_train_dataloader =",
                self.get_train_dataloader.__qualname__,
                self.get_train_dataloader.__module__
            )
            output = super().train(**kwargs)
            #logger.info(f"[Task {task_id}] 父类train方法执行完成，进程 {self.accelerator.process_index}")
            #self._wait_for_everyone()
            #logger.info(f"[Task {task_id}] 所有进程同步完成，进程 {self.accelerator.process_index}")
            return output

        elif self.method == "adaptive" and task_id > 1:

            def _canon_name(name: str) -> str:
                return name[7:] if name.startswith("module.") else name

            state_has_history = self.continual_state.has_valid_history() if self.continual_state is not None else False

            # ========== 冷启动：该 client 没有历史 -> 走一次常规训练 + Fisher ==========
            if self.continual_state is None or base_params is None or (
                    self.continual_state is not None and not state_has_history
            ):
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
            self.eta_per_layer = defaultdict(list)

            # 启用激活探针
            self.enable_activation_probe = bool(getattr(self.args, "enable_activation_probe", True))
            # === [新增] 在进入自适应训练循环前安装激活探针 ===
            if self.accelerator.is_main_process and getattr(self.args, "enable_activation_probe", True):
                self._act_probe = _ActivationProbe(
                    topk=5,
                    ratio_trigger=float(getattr(self.args, "actprobe_ratio_trigger", 30.0)),
                    print_limit=int(getattr(self.args, "actprobe_print_limit", 10)),
                )
                named_modules = dict(model_plain.named_modules())
                lora_param_names = [n for (n, p) in model_plain.named_parameters() if ("lora_A" in n or "lora_B" in n)]
                self._act_probe.register(model_plain, named_modules, lora_param_names)


            # ---------- 初始化 per-param 缓存（规范化键名） ----------
            F_curr = {}
            bar_F_raw = {}
            bar_theta = {}


            if use_adaptive_logic:

                for n, p in model_plain.named_parameters():
                    if not p.requires_grad:
                        continue
                    cn = _canon_name(n)
                    F_curr[cn] = torch.zeros_like(p, device=device)

                    if self.continual_state.has_valid_history():
                        # 历史 F / θ 均以规范化键名存取
                        bar_F_raw[cn] = self.continual_state.get_f_past(cn)
                        bar_theta[cn] = self.continual_state.theta_last[cn]
                    else:
                        bar_F_raw[cn] = torch.zeros_like(p, device="cpu")  # 确保在CPU
                        bar_theta[cn] = torch.zeros_like(p, device="cpu")  # 确保在CPU

                # ---------- 算法态量 ----------
                r2, r2_start = {}, {}
                B_round, conf = {}, {}

                # 初始化半径和收益状态（关键：需要跨进程同步）
                for name, p in model_plain.named_parameters():
                    if not p.requires_grad:
                        continue
                    cn = _canon_name(name)
                    # r²_ℓ = ‖θ^(ℓ) - bar_theta^(ℓ)‖²_{bar_F^(ℓ)}
                    f_past = bar_F_raw[cn].to(device)
                    hist_mean = torch.clamp(f_past.mean(), min=1e-12)
                    f_past_eff = f_past / hist_mean
                    theta_curr = p.detach()  # 当前层参数θ^(ℓ)
                    theta_past = bar_theta[cn].to(device)
                    r2[cn] = ((theta_curr - theta_past).pow(2) * f_past_eff).sum()
                    # r²_ℓ,start = r²_ℓ（记录初始半径）
                    r2_start[cn] = r2[cn].clone()
                    # B^round_ℓ = 0（收益累计）
                    B_round[cn] = torch.tensor(0.0, device=device)
                    # conf_ℓ = 0（冲突次数统计）
                    conf[cn] = 0

            else:
                # 基线模式：仅初始化“手动更新必要参数”（无多余代码）
                lr_base = self.args.learning_rate  # 基线用固定学习率（与自适应lr_cap一致）
                # 基线无需bar_F/bar_B，仅记录初始参数用于计算delta
                for k, p in model.named_parameters():
                    if p.requires_grad:
                        bar_theta[k] = p.detach().clone()


            # ==================== 训练循环 ====================
            num_epochs = int(self.args.num_train_epochs)
            steps_per_epoch = len(train_dataloader)
            actual_steps = 0  # 真实已处理的 mini-batch 数（用于 p_round 分母）
            #eta_save = []
            round_s = []
            Delta_s = []
            diag_loss = []
            diag_weff = []
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
                for step, batch in enumerate(train_dataloader):

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

                    self.accelerator.backward(loss)

                    raw_loss = loss.detach().item() * self.args.gradient_accumulation_steps
                    # 输出日志：包含epoch、step、当前batch的loss
                    if self.accelerator.is_main_process:
                        raw_loss = loss.detach().item() * self.args.gradient_accumulation_steps
                        logger.info(
                            f"Task {task_id} | Epoch [{epoch + 1}/{num_epochs}] | "
                            f"Batch [{step + 1}/{steps_per_epoch}] | Batch Loss: {raw_loss:.6f}"
                        )
                        diag_loss.append(raw_loss)

                    # before_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

                    ###########################################
                    # 第二步：调用独立函数，在线更新Fisher
                    ###########################################
                    # 调用函数：传入模型、当前batch、F_curr、EMA系数、设备
                    F_curr, F_batch = update_online_fisher(
                        model=model,
                        batch=batch,
                        F_curr=F_curr,
                        alpha_ema=self.alpha,
                        device=device
                    )

                    # global_norm = None
                    # if any(p.grad is not None for p in model.parameters()):
                    #     s = 0.0
                    #     for p in model.parameters():
                    #         if p.grad is not None:
                    #             s += float(p.grad.detach().float().norm().item()) ** 2
                    #     global_norm = (s ** 0.5)
                    #
                    # # === [新增] 如有需要，打印可疑层的详细激活/反传统计 ===
                    # if getattr(self, "_act_probe", None) is not None:
                    #     clip_norm = float(getattr(self.args, "clip_grad_norm", 1.0))
                    #     self._act_probe.report_if_needed(model_plain, logger, global_norm, clip_norm)
                    #
                    #
                    #
                    # # 监控点1：梯度范数检查
                    # grad_norms = {}
                    # for name, p in model_plain.named_parameters():
                    #     if p.grad is not None:
                    #         grad_norm = torch.norm(p.grad).item()
                    #         grad_norms[name] = grad_norm
                    #         if grad_norm > 1000:  # 梯度爆炸阈值
                    #             logger.warning(f"梯度爆炸检测: {name}, 梯度范数: {grad_norm}")
                    #
                    # # 监控点2：Fisher值检查
                    # fisher_values = {}
                    # for name in F_curr:
                    #     fisher_mean = torch.mean(F_curr[name]).item()
                    #     fisher_values[name] = fisher_mean
                    #     if fisher_mean < 1e-12 or fisher_mean > 1e6:
                    #         logger.warning(f"Fisher值异常: {name}, 均值: {fisher_mean}")
                    #
                    # # 将监控数据保存
                    # if self.accelerator.is_main_process:
                    #     gradient_monitor['grad_norms'].append(grad_norms)
                    #     gradient_monitor['fisher_values'].append(fisher_values)
                    #     gradient_monitor['loss_values'].append(raw_loss)
                    #     try:
                    #         weff = self._compute_delta_weff_norm(model_plain, bar_theta)
                    #         diag_weff.append(weff)
                    #     except Exception:
                    #         pass


                    ###########################################
                    # 第三步：自适应学习率计算
                    ###########################################
                    # 自适应学习率
                    with torch.no_grad():
                        if use_adaptive_logic:
                            for name, p in model.named_parameters():
                                if not p.requires_grad or p.grad is None:
                                    continue

                                cn = _canon_name(name)

                                # Step 1
                                g = p.grad.detach()

                                # F_batch = g * g
                                # F_curr[name] = self.alpha * F_curr[name] + (1 - self.alpha) * F_batch
                                F_curr_safe = F_curr[cn]
                                # Step 2
                                # v = g / F_curr_safe
                                # 2. 过去任务Fisher（bar_F^(ℓ)）
                                f_past_cpu = bar_F_raw.get(cn, None)  # 从 CPU 获取

                                # 检查 CPU 上的 f_past 是否有效
                                if f_past_cpu is None or torch.all(f_past_cpu <= 0) or f_past_cpu.mean() <= 1e-12:
                                    f_past_eff = torch.ones_like(p)  # 在 GPU (p.device) 上创建
                                else:
                                    f_past = f_past_cpu.to(device)  # <-- 即时移至 GPU
                                    f_past_eff = f_past / f_past.mean()
                                    del f_past  # <-- 可选：立即释放 f_past 的 GPU 显存


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
                                        prev_scale = self._eta_scale.get(cn, 1.0)
                                        ratio = curr_mean / hist_mean
                                        ratio = max(self.fisher_floor, ratio)

                                        scale = (1 - self._eta_scale_rho) * prev_scale + self._eta_scale_rho * ratio
                                        scale = max(self._eta_smin, min(self._eta_smax, scale))
                                        self._eta_scale[cn] = scale

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
                                preconditioner = (F_soft.pow(gamma)) / scale
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
                                # delta_theta = p - bar_theta.get(cn, torch.zeros_like(p))
                                theta_past = bar_theta.get(cn, torch.zeros_like(p, device="cpu")).to(
                                    device)  # <-- 即时移至 GPU
                                delta_theta = p - theta_past
                                del theta_past  # <-- 立即释放 theta_past 的 GPU 显存
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
                                # if name == 'base_model.model.encoder.block.10.layer.0.SelfAttention.q.lora_B.default.weight':
                                #     pass
                                # Delta = (self.radius ** 2) - r2[name]
                                Delta_local = (self.radius ** 2) - r2[cn]
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

                                    # 缩放
                                    Delta_eff = (self.trust_radius_shrink ** 2) * Delta_eff
                                    beta_eff = self.beta * self.beta_mul
                                    if b_raw_alg.item() < 0.0:  # 冲突层：b_B^(ℓ) < 0（新旧任务方向冲突）
                                        # 闭式步长公式（严格按算法）：
                                        #term_u = b_raw_alg - self.sigma
                                        # term_u = b_raw_alg - torch.as_tensor(self.sigma, device=device, dtype=p.dtype)
                                        term_u = b_raw_alg - torch.as_tensor(self.sigma, device=device, dtype=p.dtype)
                                        discriminant = term_u ** 2 + beta_eff * a_safe * Delta_eff
                                        discriminant = torch.clamp(discriminant, min=0.0)  # 确保开方非负
                                        xx = (term_u + torch.sqrt(discriminant))
                                        yy = (beta_eff * a_safe)
                                        eta_closed = xx / yy
                                        eta_ori_step.append(eta_closed.detach().float().mean().cpu().clone())
                                        # 步长上限：不超过初始学习率 η₀
                                        eta_alg = eta_closed * self.eta_shrink
                                        eta_alg = torch.minimum(
                                            eta_alg,
                                            torch.tensor(self.args.learning_rate, device=device, dtype=p.dtype),
                                        )
                                        if eta_alg.item() > 0:
                                            conf[cn] += 1
                                    else:
                                        eta_trust = torch.sqrt(Delta_eff / a_safe)
                                        eta_alg = eta_trust * self.eta_shrink
                                        eta_save_step.append(eta_alg.detach().float().mean().cpu().clone())
                                        eta_alg = torch.clamp(
                                            eta_alg,
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

                                B_round[cn] = B_round[cn] + gain

                                r2[cn] = r2[cn] - 2.0 * eta_alg * b_raw_alg + (eta_alg * eta_alg) * a_raw_alg

                                # --------------------------
                                # 算法步骤11：更新参数 θ^(ℓ) = θ^(ℓ) - η·v_B^(ℓ)
                                # --------------------------

                                # #method 1: use SGD
                                # if eta.item() != 0.0:
                                #     p.add_(-eta * v_scaled)  # 修正：使用缩放后的v_scaled

                                #method 2: use adawm
                                # ---- 用 AdamW 替换手动 SGD（仍以 eta 为每层步长）----
                                if eta.item() != 0.0:
                                    state = self.adam_states.setdefault(cn, {
                                        "m": torch.zeros_like(p, device=p.device, dtype=p.dtype),
                                        "v": torch.zeros_like(p, device=p.device, dtype=p.dtype),
                                        "t": 0,
                                    })
                                    state["t"] += 1
                                    t = state["t"]
                                    # 一阶/二阶动量在“预条件方向 v_scaled”上统计
                                    state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * v_scaled
                                    state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (v_scaled * v_scaled)
                                    # 偏置修正
                                    m_hat = state["m"] / (1 - (self.beta1 ** t))
                                    v_hat = state["v"] / (1 - (self.beta2 ** t))
                                    # Adam 方向
                                    adam_dir = m_hat / (torch.sqrt(v_hat) + self.eps)
                                    # decoupled weight decay（与 Adam 方向解耦）
                                    wd = float(self.adamw_weight_decay) if hasattr(self, "adamw_weight_decay") else 0.0
                                    if wd > 0.0:
                                        p.add_(-eta * wd * p)   # W ← W - η·wd·W
                                    p.add_(-eta * adam_dir)    # W ← W - η·AdamDir


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

                    if self.accelerator.is_main_process:
                        s = 0
                        for i in eta_d:
                            if i == self.args.learning_rate:
                                s += 1
                        if len(eta_d) == s:
                            print('这个batch所有层都是统一学习率')

                    # if step % 1 == 0:
                    #     health_report = monitor_gradient_health(step, model, F_curr, bar_F_raw, r2, B_round)
                    #
                    #     if health_report['issues']:
                    #         logger.warning(f"步骤 {step} 健康检查发现问题:")
                    #         for issue in health_report['issues']:
                    #             logger.warning(f"  - {issue}")

                        # 记录关键指标
                        # logger.info(f"步骤 {step} 关键指标:")
                        # for metric, value in health_report['metrics'].items():
                        #     if 'grad_norm' in metric or 'b_round' in metric:
                        #         logger.info(f"  {metric}: {value:.6f}")

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



                selection_map = {n: f for n, f in zip(names, selected_flags)}

                unwrapped_model = self.accelerator.unwrap_model(model)

                name_to_param_raw = dict(unwrapped_model.named_parameters())


                # 1) 自适应模式：仅传输选中层的delta
                # state_dict = model.state_dict()
                adaptive_delta = {}
                for name in base_params:

                    p_raw = name_to_param_raw.get(name, None)
                    if p_raw is None:
                        continue  # 或者 log warning
                    local_param = p_raw.detach().cpu()

                    # 仅选中的层传输真实delta，未选中的层传输零（不实际传输，仅占位）
                    #adaptive_delta[name] = (base_params[name] - local_param) if sel else torch.zeros_like(local_param)
                    if selection_map.get(name, True):
                        adaptive_delta[name] = base_params[name] - local_param
                    else:
                        adaptive_delta[name] = torch.zeros_like(local_param)

                delta = adaptive_delta

                # 3) 本轮Fisher计算
                F_client = {}
                for k in base_params:
                    if k in name_to_param_raw:  # 确保参数存在
                        # 使用 name_to_param_raw 来获取形状
                        F_client[k] = F_curr.get(k, torch.zeros_like(name_to_param_raw[k])).detach().cpu()
            else:
                # 基线模式：传输所有可训练参数的delta
                state_dict = model.state_dict()
                baseline_delta = {}
                for name, p in model.named_parameters():
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
            theta_last = {}
            for k in base_params.keys():
                p_raw = name_to_param_raw.get(k, None)
                if p_raw is not None:
                    theta_last[k] = p_raw.detach().cpu()

            # 返回值新增通信节省时间
            self.accelerator.wait_for_everyone()

            # ====== 训练结束：在主进程画图并给出增长性判断 ======
            if self.accelerator.is_main_process:
                import os, json, numpy as _np, matplotlib.pyplot as _plt, time as _t
                os.makedirs(self.args.output_dir, exist_ok=True)
                # 持久化 step 级诊断
                jpath = os.path.join(self.args.output_dir, f"step_metrics_task{task_id}.jsonl")
                with open(jpath, "w", encoding="utf-8") as fw:
                    for i,(lval, wv) in enumerate(zip(diag_loss, diag_weff)):
                        fw.write(json.dumps({"step": i+1, "loss": float(lval), "delta_weff": float(wv)})+"\n")
                # 画损失曲线
                try:
                    _plt.figure()
                    _plt.plot(range(1, len(diag_loss)+1), diag_loss)
                    _plt.xlabel("Step"); _plt.ylabel("Train Loss"); _plt.title(f"Task {task_id} - Loss Curve")
                    _plt.tight_layout()
                    _plt.savefig(os.path.join(self.args.output_dir, "loss_curve.png"), dpi=180)
                    _plt.close()
                except Exception:
                    pass
                # 画 ||ΔW_eff|| 曲线
                try:
                    _plt.figure()
                    _plt.plot(range(1, len(diag_weff)+1), diag_weff)
                    _plt.xlabel("Step"); _plt.ylabel(r"||$\Delta W_{\rm eff}$||$_F$")
                    _plt.title(f"Task {task_id} - Effective LoRA ΔW Norm")
                    _plt.tight_layout()
                    _plt.savefig(os.path.join(self.args.output_dir, "delta_weff_curve.png"), dpi=180)
                    _plt.close()
                except Exception:
                    pass
                # 趋势判断（是否随步数增长）
                trend = "unknown"
                try:
                    if len(diag_weff) >= 3:
                        x = _np.arange(len(diag_weff))
                        k = _np.polyfit(x, _np.array(diag_weff, dtype=float), 1)[0]  # 线性斜率
                        # Spearman 相关（粗略 monotonic）
                        from scipy.stats import spearmanr as _spr
                        rho, _ = _spr(x, _np.array(diag_weff, dtype=float))
                        trend = f"slope={k:.4e}, spearman_rho={rho:.3f}, increasing={bool(k>0 and rho>0.2)}"
                except Exception:
                    pass
                with open(os.path.join(self.args.output_dir, "diagnostics_summary.txt"), "w", encoding="utf-8") as fw:
                    fw.write(f"ΔW_eff trend: {trend}\n")
                    fw.write(f"loss_points={len(diag_loss)}, weff_points={len(diag_weff)}\n")

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
