from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
from fed_continual_state import ContinualState
from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX
from collections import deque
from tqdm.auto import tqdm
import torch.nn.utils as utils

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


def compute_fisher(model, dataloader, alpha: float = 0.9, engine=None):
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
        radius: float = 0.5,
        sigma: float = 0.0,
        tau: float = 0.0,
        beta: float = 1.0,
        alpha: float = 0.5,
        comm_budget: int = 0,
        layer_costs: Dict[str, int] = None,** kwargs,
    ):
        self.continual_state = kwargs.pop("state", None)  # 这里直接赋值给 continual_state
        super().__init__(*args, **kwargs)  # 此时 kwargs 中已无 state，父类不会报错
        self.radius = radius
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




    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        # 根据输入数据的类型进行适当的处理，确保输入数据符合模型的要求。
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        # 根据当前训练环境（是否使用 AMP、是在 CPU 还是 CUDA 上）和 PyTorch 版本，返回一个合适的 autocast 上下文管理器，
        # 以便在 with 中使用混合精度加速模型推理或训练。
        with self.compute_loss_context_manager():
            # 算前向loss
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        ########################### Regularization ##########################
        # orthogonal_loss = 0.
        # for name, param in self.model.named_parameters():
        #     if "lora_A" in name:
        #         for name_, param_ in self.model.named_parameters():
        #             if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
        #                 orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
        #                 break # target modules have been matched
        #
        # # l2-normalization for loranew_A/B
        # l2_loss = 0.
        # for name, param in self.model.named_parameters():
        #     if "loranew_" in name:
        #         l2_loss += torch.norm(param, p=2)
        #
        # lamda_1 = self.args.lamda_1
        # lamda_2 = self.args.lamda_2
        #
        # #logger.info(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {lamda_1}; λ2: {lamda_2}")
        # logger.info(
        #     f"orthogonal_loss: {orthogonal_loss}; l2_loss: {l2_loss}; accuracy_loss: {loss.item()}; λ1: {lamda_1}; λ2: {lamda_2}")
        # loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
        ######################################################################

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def train(
        self,
        task_id: int = 1,
        base_params: Dict[str, torch.Tensor] = None,
        cid: int = -1,
        **kwargs,
    ):  # type: ignore[override]

        if not hasattr(self, "_eta_scale"):
            self._eta_scale = {}  # per-parameter EMA 的 s
        self._eta_scale_rho = getattr(self, "eta_scale_rho", 0.1)  # EMA 系数，0.05~0.2
        self._eta_smin = getattr(self, "eta_smin", 0.1)  # 缩放下界
        self._eta_smax = getattr(self, "eta_smax", 10.0)  # 缩放上界

        if self.method == "lora_origin":
            return super().train(**kwargs)


        if self.method == "adaptive" and task_id == 1:
            return super().train(**kwargs)

        elif self.method == "adaptive" and task_id > 1:
            state_has_history = self.continual_state.has_valid_history() if self.continual_state is not None else False
            # 如果该节点之前从未被训练过（无历史）：走一次普通训练 + Fisher
            if self.continual_state is None or base_params is None or (
                    self.continual_state is not None and not state_has_history
            ):
                super().train(**kwargs)
                state_dict = self.model.state_dict()
                delta = {k: base_params[k] - state_dict[k].detach().cpu() for k in base_params}
                F_client = compute_fisher(self.model, self.get_train_dataloader())
                F_client = {k: F_client.get(k, torch.zeros_like(base_params[k])) for k in base_params}
                theta_last = {k: state_dict[k].detach().cpu() for k in F_client.keys()}
                return delta, F_client, theta_last

            # --------------Train---------------
            model = self.model
            dataloader = self.get_train_dataloader()
            device = next(model.parameters()).device
            model.train()

            # 初始化F_curr为小正值，提升稳定性
            F_curr = {
                n: 1e-3 * torch.ones_like(p, device=device)  # 初始化为1e-6，而非0
                for n, p in model.named_parameters()
                if p.requires_grad
            }

            # Load old task
            bar_F_raw  = {k: v.to(device) for k, v in self.continual_state.bar_F.items()}
            bar_B_raw  = {k: v.to(device) for k, v in self.continual_state.bar_B.items()}
            bar_theta = {}
            for k, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if (k in bar_F_raw) and (k in bar_B_raw):
                    bar_theta[k] = bar_B_raw[k] / (bar_F_raw[k] + 1e-8)
                else:
                    bar_theta[k] = torch.zeros_like(p, device=device)

            # 工程
            bar_F_eff, bar_B_eff = {}, {}
            scale_s, calibrated = {}, False
            S_MIN, S_MAX, EPS_S = 0.1, 1000.0, 1e-12  # 放缩上/下界

            r2, r2_start = {}, {}
            B_round, conf = {}, {}
            num_epochs = int(self.args.num_train_epochs)
            steps_per_epoch = len(dataloader)
            actual_steps = 0  # 真实已处理的 mini-batch 数（用于 p_round 分母）

            # for name, p in model.named_parameters():
            #     if not p.requires_grad:
            #         continue
            #     f = bar_F.get(name, torch.zeros_like(p, device=device))
            #     r2[name] = ((p - bar_theta.get(name, torch.zeros_like(p))).pow(2) * f).sum()
            #     r2_start[name] = r2[name].clone()
            #     B_round[name] = torch.tensor(0.0, device=device)
            #     conf[name] = 0



            # /*-------------Begin train------------
            for epoch in range(num_epochs):
                for step, batch in enumerate(dataloader):
                    actual_steps += 1
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    # 清梯度
                    model.zero_grad(set_to_none=True)

                    # 前向
                    with self.compute_loss_context_manager():
                        outputs = model(**batch)
                        loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs["loss"]

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    # 反向传播
                    loss.backward()

                    # 打印loss结果
                    raw_loss = loss.detach().item() * self.args.gradient_accumulation_steps
                    # 输出日志：包含epoch、step、当前batch的loss
                    logger.info(
                        f"Task {task_id} | Epoch [{epoch + 1}/{num_epochs}] | Batch [{step + 1}/{steps_per_epoch}] | "
                        f"Batch Loss: {raw_loss:.6f}"
                    )

                    # 自适应学习率
                    with torch.no_grad():
                        F_norm_per_param = {}
                        for name, p in model.named_parameters():
                            if not p.requires_grad or p.grad is None:
                                continue
                            g = p.grad
                            F_batch = g * g
                            F_curr[name] = self.alpha * F_curr[name] + (1 - self.alpha) * F_batch

                            # 对F归一化
                            F_min = torch.min(F_curr[name])
                            F_max = torch.max(F_curr[name])
                            epsilon = 1e-3  # 避免除零，同时接近0以保留[0,1]的相对比例特性
                            if (F_max - F_min) <= 1e-12:
                                F_norm = torch.full_like(F_curr[name], epsilon)
                            else:
                                # 1. 缩放到[0, 1]：保留原始F_curr的相对比例
                                F_scaled = (F_curr[name] - F_min) / (F_max - F_min + 1e-12)
                                # 2. 偏移到[ε, 1]：确保最小值为ε，最大值为1，既安全又接近0-1
                                F_norm = F_scaled * (1 - epsilon) + epsilon
                            F_norm_per_param[name] = F_norm

                        if not calibrated:
                            for name, p in model.named_parameters():
                                if not p.requires_grad:
                                    continue
                                if name in bar_F_raw:
                                    m_curr = float(F_norm_per_param[name].mean().detach().cpu())
                                    m_hist = float(bar_F_raw[name].mean().detach().cpu())
                                    if m_hist > 0.0:
                                        s = m_curr / (m_hist + EPS_S)
                                        s = max(S_MIN, min(s, S_MAX))
                                    else:
                                        s = 1.0
                                    scale_s[name] = s
                                    bar_F_eff[name] = bar_F_raw[name] * s
                                    bar_B_eff[name] = bar_B_raw[name] * s
                                    # === 新增：把半径/σ/τ同步到同一尺度（单位对齐）===
                                    if "R2_eff" not in locals():
                                        R2_eff = {}
                                        sigma_eff_map = {}
                                        tau_eff_map = {}

                                    R2_eff[name] = (self.radius ** 2) * s
                                    sigma_eff_map[name] = torch.as_tensor(self.sigma, device=device) * s
                                    tau_eff_map[name] = torch.as_tensor(self.tau, device=device) * s
                                else:
                                    # 没有历史，就视为 0 惩罚
                                    bar_F_eff[name] = torch.zeros_like(p, device=device)

                                # 用“生效的历史 Fisher”初始化半径与统计量
                                f_eff = bar_F_eff[name]
                                theta_ref = bar_theta.get(name, torch.zeros_like(p))
                                r2[name] = ((p - theta_ref).pow(2) * f_eff).sum()
                                r2_start[name] = r2[name].clone()
                                B_round[name] = torch.tensor(0.0, device=device)
                                conf[name] = 0
                            calibrated = True  # 仅做一次


                        damping_factor = 1e-5
                        for name, p in model.named_parameters():
                            if not p.requires_grad or p.grad is None:
                                continue

                            g = p.grad
                            F_norm = F_norm_per_param[name]
                            v = g / (F_norm + damping_factor)

                            f_eff = bar_F_eff.get(name, torch.zeros_like(p))
                            diff = p - bar_theta[name]
                            a_raw = (v * f_eff * v).sum()  # v^T \bar F_eff v
                            b_raw = (v * f_eff * diff).sum()  # v^T \bar F_eff (θ-θ̄)
                            Delta = (self.radius ** 2) - r2[name]  # 或者你用的 R2_eff[name] - r2[name]

                            a_curr = (v * F_norm * v).sum()

                            # 2) 每层维护一个 a 的 EMA 作为基线
                            if not hasattr(self, "_a_ema"):
                                self._a_ema = {}
                            rho = getattr(self, "a_ema_rho", 0.1)  # EMA 系数（0.05~0.2）
                            a_prev = self._a_ema.get(name, a_curr.detach())
                            a_ema = (1 - rho) * a_prev + rho * a_curr.detach()
                            self._a_ema[name] = a_ema

                            # 3) 只在“步长计算”里给 a 一个动态下界
                            mu = getattr(self, "a_curr_floor_mu", 0.05)  # 0.02~0.2
                            kappa = getattr(self, "a_ema_floor_kappa", 0.1)  # 0.05~0.3
                            a_eta = torch.maximum(a_raw, torch.maximum(mu * a_curr,
                                                                       kappa * a_ema.to(a_raw.dtype).to(a_raw.device)))

                            # # ======= ★ 关键：把“步长用的 Δ”压到目标尺度 ★ =======
                            # # 目标：eta_closed 不超过 eta_goal（通常取你想要的上限量级，比如 1e-3 或 0.5*1e-3）
                            # eta_goal = torch.as_tensor(getattr(self, "eta_goal", self.args.learning_rate),
                            #                            device=device)
                            # beta_t = torch.as_tensor(self.beta, device=device)
                            # # 近似反推：Δ_goal ≈ eta_goal^2 * β * a_eta
                            # Delta_goal = (eta_goal * eta_goal) * beta_t * a_eta
                            # Delta_step = torch.clamp_min(torch.minimum(Delta, Delta_goal), 0.0)  # 仅用于闭式步长
                            #
                            # EPS = 1e-8
                            # LR_T = torch.as_tensor(self.args.learning_rate, device=device)
                            #
                            # if b_raw.item() < 0.0:
                            #     term = (b_raw - self.sigma)  # 若你做了 σ 的尺度对齐，替换为 sigma_eff_map[name]
                            #     # 用 Δ_step 进入闭式公式（只改这一处 Δ）
                            #     disc = term * term + beta_t * a_eta * (
                            #                 Delta_step - self.tau)  # τ 同理，可用 tau_eff_map[name]
                            #     disc = torch.clamp(disc, min=0.0)
                            #     eta_closed = (term + torch.sqrt(disc)) / (beta_t * a_eta + EPS)
                            #
                            #     # 信赖域（严格几何）仍用原始 Δ 与 a_raw，保持你的遗忘定义
                            #     eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))
                            #
                            #     # 最终步长：闭式 ∩ 信赖域 ∩ 全局 lr
                            #     eta = torch.minimum(torch.minimum(eta_closed, eta_tr), LR_T)
                            #     conf[name] += 1
                            #     logger.info(f"eta is {eta} when conflict")
                            # else:
                            #     # 非冲突：你原来的策略（也可加上信赖域）
                            #     eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))
                            #     eta = torch.minimum(LR_T, eta_tr)

                            beta_t = torch.as_tensor(self.beta, device=device)
                            lr_cap = torch.as_tensor(self.args.learning_rate, device=device)

                            # 冲突项（与闭式一致）：term_u = (b - σ)
                            term_u = b_raw - self.sigma

                            # —— 用“冲突强度”决定目标步长是 lr 的几分之几 —— #
                            # 归一尺度：√(β a_eta · Δ) ；冲突越强，目标越接近 lr，反之越小
                            scale = torch.sqrt(
                                torch.clamp(beta_t * a_eta * torch.clamp(Delta, min=0.0) + 1e-12, min=1e-12))
                            b_norm = torch.abs(term_u) / scale  # 无量纲冲突强度

                            conf_k = getattr(self, "conf_k", 1.0)  # 0.5~2.0：对冲突强度的敏感度
                            eta_min_frac = getattr(self, "eta_min_frac", 0.01)  # 0.05~0.3：最小占比，给很弱冲突一个小步长
                            # 映射到 (eta_min_frac, 1)：平滑且有界
                            w_conf = eta_min_frac + (1.0 - eta_min_frac) * (b_norm * conf_k / (1.0 + b_norm * conf_k))

                            eta_goal_dyn = lr_cap * w_conf  # 动态目标步长（≤ lr）

                            # —— 精确反推 Δ_goal（包含 b 项；比 “η^2 β a” 更精确）—— #
                            # 由闭式反解：η* = ((b-σ)+√((b-σ)^2 + β a (Δ-τ)))/(β a)
                            # 推得：Δ_goal = β a η*^2 - 2 (b-σ) η* + τ
                            Delta_goal_exact = beta_t * a_eta * (
                                        eta_goal_dyn ** 2) - 2.0 * term_u * eta_goal_dyn + self.tau

                            # 仅用于“求步长”的 Δ（不改你的遗忘几何）
                            Delta_step = torch.clamp_min(torch.minimum(Delta, Delta_goal_exact), 0.0)

                            EPS = 1e-8
                            if b_raw.item() < 0.0:
                                disc = term_u * term_u + beta_t * a_eta * (Delta_step - self.tau)
                                disc = torch.clamp(disc, min=0.0)
                                eta_closed = (term_u + torch.sqrt(disc)) / (beta_t * a_eta + EPS)

                                # 信赖域上界仍用原 Δ 与 a_raw（保持几何）
                                eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))

                                eta = torch.minimum(torch.minimum(eta_closed, eta_tr), lr_cap)
                                conf[name] += 1
                                logger.info(f"[{name}] eta={float(eta):.6g} w_conf={float(w_conf):.3f} "
                                            f"goal={float(eta_goal_dyn):.3g} b_norm={float(b_norm):.3g}")
                            else:
                                eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))
                                eta = torch.minimum(lr_cap, eta_tr)













                            # # 4) 冲突闭式步长 + 温和信赖域上界（半径上界仍用 a_raw，几何不变）
                            # LR_T = torch.as_tensor(self.args.learning_rate, device=device)
                            # EPS = 1e-8
                            #
                            # if b_raw.item() < 0.0:
                            #     term = (b_raw - self.sigma)  # 若你已做 σ 对齐，可替换为 sigma_eff_map[name]
                            #     disc = term * term + self.beta * a_eta * (Delta - self.tau)  # τ 同理；不想动就保留原样
                            #     disc = torch.clamp(disc, min=0.0)
                            #     eta_closed = (term + torch.sqrt(disc)) / (self.beta * a_eta + EPS)
                            #
                            #     # 信赖域上限（仍以历史几何衡量半径，保持你的遗忘定义）
                            #     eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))
                            #
                            #     eta = torch.minimum(torch.minimum(eta_closed, eta_tr), LR_T)
                            #     conf[name] += 1
                            # else:
                            #     # 非冲突：用基础 lr，也可与 eta_tr 取 min 防越半径
                            #     eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))
                            #     eta = torch.minimum(LR_T, eta_tr)



                            # /* -----------step-2----------------
                            # a = (v * bar_F[name] * v).sum() if name in bar_F else torch.tensor(0.0, device=device)
                            # diff = p - bar_theta[name]
                            # b_val = (v * bar_F[name] * diff).sum() if name in bar_F else torch.tensor(0.0, device=device)
                            # Delta = (self.radius ** 2) - r2[name]

                            # if b_val.item() < 0.0:
                            #     term = (b_val - self.sigma)
                            #     disc = term * term + self.beta * a * (Delta - self.tau)
                            #     disc = torch.clamp(disc, min=0.0)
                            #     eta = (term + torch.sqrt(disc)) / (self.beta * a + 1e-8)
                            #     eta = torch.clamp(eta, max=self.args.learning_rate)
                            #     conf[name] += 1
                            # else:
                            #     eta = torch.as_tensor(self.args.learning_rate, device=device)

                            # 新任务收益（二阶）
                            Q = torch.sum(g * v)
                            gain = (eta - 0.5 * eta * eta) * Q
                            gain = torch.clamp(gain, min=0.0)

                            # C = a  # 自然梯度等价：C = v^T f_eff v == a
                            # gain_2 = torch.clamp(eta * Q - 0.5 * eta * eta * C, min=0.0)
                            #
                            # if gain == gain_2:
                            #     pass

                            B_round[name] = B_round[name] + gain

                            # 半径更新：r^2 = r^2 - 2η b + η^2 a
                            # r2[name] = r2[name] - 2.0 * eta * b_val + (eta * eta) * a

                            r2[name] = r2[name] - 2.0 * eta * b_raw + (eta * eta) * a_raw

                            # 参数更新
                            p.add_(-eta * v)


            # ====== Step-3 ======
            F_round = {n: 0.5 * torch.clamp(r2[n] - r2_start[n], min=0.0) for n in r2}
            p_round = {n: conf[n] / max(actual_steps, 1) for n in conf}  # ←← 用真实步数

            # --- 轮内稳健归一 + 自适应 λ ---

            # eps = 1e-8
            # names = list(B_round.keys())
            # b_vals = torch.tensor([B_round[n].item() for n in names])
            # c_vals = torch.tensor([(p_round[n] * F_round[n]).item() for n in names])
            #
            # # 分位数归一
            # med_b = torch.quantile(b_vals, 0.5)
            # med_c = torch.quantile(c_vals, 0.5) if (c_vals > 0).any() else torch.tensor(1.0)
            #
            # def norm_b(x):
            #     return x / (med_b + eps)
            #
            # def norm_c(x):
            #     return x / (med_c + eps)
            #
            # # 自适应 λ
            # lambda_base = getattr(self, "lambda_conf", 1.0)
            # lambda_eff = float(lambda_base) * float(med_b / (med_c + eps))
            #
            # # B 的 p95 截断，抑制异常层
            # b_cap = torch.quantile(b_vals, 0.95).item()
            # for n in names:
            #     B_round[n] = torch.clamp(B_round[n], max=b_cap)
            #
            # # 构建背包价值与成本
            # values, costs = [], []
            # for n in names:
            #     b = norm_b(B_round[n])
            #     c = norm_c(p_round[n] * F_round[n])
            #     v = float(b - lambda_eff * c)
            #     values.append(v)
            #     costs.append(int(self.layer_costs.get(n, 1)))
            #
            # selected = _knapsack(values, costs, self.comm_budget)


            # select layer update
            values, costs, names = [], [], []
            for name in B_round:
                val = (B_round[name] - self.lambda_conf * p_round[name] * F_round[name]).item()
                values.append(val)
                costs.append(int(self.layer_costs.get(name, 1)))
                names.append(name)
            selected = _knapsack(values, costs, self.comm_budget)

            # 1) 选层打包（你现有的）
            delta = {}
            state_dict = model.state_dict()
            for name, sel in zip(names, selected):
                local_param = state_dict[name].detach().cpu()
                delta[name] = (base_params[name] - local_param) if sel else torch.zeros_like(local_param)

            # 2) 本轮 Fisher（保持你现有的）
            F_client = {k: v.detach().cpu() for k, v in F_curr.items()}

            # 3) 本轮结束时该客户端的 θ*（只取可训练/LoRA keys）
            theta_last = {k: state_dict[k].detach().cpu() for k in F_client.keys()}

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
