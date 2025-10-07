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
        comm_bandwidth: float = 1.0,
        comm_fixed_cost: float = 0.0,
        comm_budget: int = 0,
        layer_costs: Dict[str, int] = None,** kwargs,
    ):
        self.continual_state = kwargs.pop("state", None)  # 这里直接赋值给 continual_state
        super().__init__(*args, **kwargs)  # 此时 kwargs 中已无 state，父类不会报错
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
            use_adaptive_logic = True  # 切换：False=基线模式，True=自适应模式
            batch_times = []  # 存储每个batch的训练耗时（毫秒）
            warmup_skip = 1  # 跳过第一个batch（GPU热身时间，避免干扰统计）
            logger.info(f"===== 开始训练 | 模式：{'自适应' if use_adaptive_logic else '基线'} | Task {task_id} =====")

            model = self.model
            dataloader = self.get_train_dataloader()
            device = next(model.parameters()).device
            model.train()

            # 工程
            F_curr = None
            bar_F_raw = None
            bar_B_raw = None
            bar_theta = {}
            # bar_F_eff, bar_B_eff = {}, {}
            #scale_s, calibrated = {}, False
            #S_MIN, S_MAX, EPS_S = 0.1, 1000.0, 1e-12  # 原代码参数
            r2, r2_start = {}, {}
            B_round, conf = {}, {}

            if use_adaptive_logic:
                # 初始化F_curr为小正值，提升稳定性
                # F_curr = {
                #     n: 1e-3 * torch.ones_like(p, device=device)  # 初始化为1e-6，而非0
                #     for n, p in model.named_parameters()
                #     if p.requires_grad
                # }
                F_curr = {
                    n: torch.zeros_like(p, device=device)  # 令F_curr^l = 0
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

                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    # r²_ℓ = ‖θ^(ℓ) - bar_theta^(ℓ)‖²_{bar_F^(ℓ)}
                    f_past = bar_F_raw.get(name, torch.zeros_like(p))  # 过去任务Fisher
                    theta_curr = p  # 当前层参数θ^(ℓ)
                    theta_past = bar_theta.get(name, torch.zeros_like(p))  # 过去最优参数bar_theta^(ℓ)
                    r2[name] = ((theta_curr - theta_past).pow(2) * f_past).sum()
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


            num_epochs = int(self.args.num_train_epochs)
            steps_per_epoch = len(dataloader)
            actual_steps = 0  # 真实已处理的 mini-batch 数（用于 p_round 分母）





            # /*-------------Begin train------------
            for epoch in range(num_epochs):
                for step, batch in enumerate(dataloader):
                    batch_start = time.perf_counter()
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
                        if use_adaptive_logic:
                            # F_norm_per_param = {}
                            for name, p in model.named_parameters():
                                if not p.requires_grad or p.grad is None:
                                    continue
                                g = p.grad
                                F_batch = g * g
                                F_curr[name] = self.alpha * F_curr[name] + (1 - self.alpha) * F_batch
                                # --------------------------
                                # 算法步骤7：计算 v_B^(ℓ)、a_B^(ℓ)、b_B^(ℓ)、Δ_ℓ
                                # --------------------------
                                # 1. v_B^(ℓ) = g_B^(ℓ) ⊘ F_curr^(ℓ)（加1e-8避免除零）
                                F_curr_safe = F_curr[name] + 1e-8  # 算法隐含：避免F_curr为0导致除零
                                v = g / F_curr_safe
                                # 2. 过去任务Fisher（bar_F^(ℓ)）
                                f_past = bar_F_raw.get(name, torch.zeros_like(p))
                                # 3. 当前参数与过去最优参数的差：Δθ = θ^(ℓ) - bar_theta^(ℓ)
                                delta_theta = p - bar_theta.get(name, torch.zeros_like(p))
                                # 4. a_B^(ℓ) = (v_B^(ℓ))^T * bar_F^(ℓ) * v_B^(ℓ)
                                a_raw = (v * f_past * v).sum()
                                # 5. b_B^(ℓ) = (v_B^(ℓ))^T * bar_F^(ℓ) * Δθ
                                b_raw = (v * f_past * delta_theta).sum()
                                # 6. 半径余量 Δ_ℓ = R² - r²_ℓ
                                Delta = (self.radius ** 2) - r2[name]

                                # --------------------------
                                # 算法步骤8：冲突判断与步长计算
                                # --------------------------
                                if b_raw.item() < 0.0:  # 冲突层：b_B^(ℓ) < 0（新旧任务方向冲突）
                                    # 闭式步长公式（严格按算法）：
                                    term_u = b_raw - self.sigma
                                    discriminant = term_u ** 2 + self.beta * a_raw * (Delta - self.tau)
                                    discriminant = torch.clamp(discriminant, min=0.0)  # 确保开方非负
                                    eta_closed = (term_u + torch.sqrt(discriminant)) / (self.beta * a_raw + 1e-8)
                                    # 步长上限：不超过初始学习率 η₀
                                    eta = torch.minimum(eta_closed, torch.tensor(self.args.learning_rate, device=device))
                                    # 记录冲突次数 conf_ℓ += 1
                                    conf[name] += 1
                                    # logger.info(f"[{name}] 冲突层 | 步长: {float(eta):.6g} | 冲突次数: {conf[name]}")
                                else:  # 非冲突层：步长 = 初始学习率 η₀
                                    eta = torch.tensor(self.args.learning_rate, device=device)

                                # --------------------------
                                # 算法步骤9：累计收益 B^round_ℓ
                                # --------------------------
                                # 二阶收益 Q_B^(ℓ) = (g_B^(ℓ))^T * (F_curr^(ℓ))^(-1) * g_B^(ℓ)（等价于 g·v）
                                Q = torch.sum(g * v)
                                # 收益计算：max{0, (η - 0.5η²) * Q}
                                gain = (eta - 0.5 * eta ** 2) * Q
                                gain = torch.clamp(gain, min=0.0)  # 收益非负
                                B_round[name] = B_round[name] + gain

                                # --------------------------
                                # 算法步骤10：更新马氏半径 r²_ℓ
                                # --------------------------
                                r2[name] = r2[name] - 2.0 * eta * b_raw + (eta ** 2) * a_raw

                                # --------------------------
                                # 算法步骤11：更新参数 θ^(ℓ) = θ^(ℓ) - η·v_B^(ℓ)
                                # --------------------------
                                p.add_(-eta * v)



                                # # 对F归一化
                                # F_min = torch.min(F_curr[name])
                                # F_max = torch.max(F_curr[name])
                                # epsilon = 1e-3  # 避免除零，同时接近0以保留[0,1]的相对比例特性
                                # if (F_max - F_min) <= 1e-12:
                                #     F_norm = torch.full_like(F_curr[name], epsilon)
                                # else:
                                #     # 1. 缩放到[0, 1]：保留原始F_curr的相对比例
                                #     F_scaled = (F_curr[name] - F_min) / (F_max - F_min + 1e-12)
                                #     # 2. 偏移到[ε, 1]：确保最小值为ε，最大值为1，既安全又接近0-1
                                #     F_norm = F_scaled * (1 - epsilon) + epsilon
                                # F_norm_per_param[name] = F_norm

                            # if not calibrated:
                            #     for name, p in model.named_parameters():
                            #         if not p.requires_grad:
                            #             continue
                            #         if name in bar_F_raw:
                            #             m_curr = float(F_norm_per_param[name].mean().detach().cpu())
                            #             m_hist = float(bar_F_raw[name].mean().detach().cpu())
                            #             if m_hist > 0.0:
                            #                 s = m_curr / (m_hist + EPS_S)
                            #                 s = max(S_MIN, min(s, S_MAX))
                            #             else:
                            #                 s = 1.0
                            #             scale_s[name] = s
                            #             bar_F_eff[name] = bar_F_raw[name] * s
                            #             bar_B_eff[name] = bar_B_raw[name] * s
                            #             # === 新增：把半径/σ/τ同步到同一尺度（单位对齐）===
                            #             if "R2_eff" not in locals():
                            #                 R2_eff = {}
                            #                 sigma_eff_map = {}
                            #                 tau_eff_map = {}
                            #
                            #             R2_eff[name] = (self.radius ** 2) * s
                            #             sigma_eff_map[name] = torch.as_tensor(self.sigma, device=device) * s
                            #             tau_eff_map[name] = torch.as_tensor(self.tau, device=device) * s
                            #         else:
                            #             # 没有历史，就视为 0 惩罚
                            #             bar_F_eff[name] = torch.zeros_like(p, device=device)
                            #
                            #         # 用“生效的历史 Fisher”初始化半径与统计量
                            #         f_eff = bar_F_eff[name]
                            #         theta_ref = bar_theta.get(name, torch.zeros_like(p))
                            #         r2[name] = ((p - theta_ref).pow(2) * f_eff).sum()
                            #         r2_start[name] = r2[name].clone()
                            #         B_round[name] = torch.tensor(0.0, device=device)
                            #         conf[name] = 0
                            #     calibrated = True  # 仅做一次


                            # damping_factor = 1e-5
                            # for name, p in model.named_parameters():
                            #     if not p.requires_grad or p.grad is None:
                            #         continue
                            #
                            #     g = p.grad
                            #     F_norm = F_norm_per_param[name]
                            #     v = g / (F_norm + damping_factor)
                            #
                            #     f_eff = bar_F_eff.get(name, torch.zeros_like(p))
                            #     diff = p - bar_theta[name]
                            #     a_raw = (v * f_eff * v).sum()  # v^T \bar F_eff v
                            #     b_raw = (v * f_eff * diff).sum()  # v^T \bar F_eff (θ-θ̄)
                            #     Delta = (self.radius ** 2) - r2[name]  # 或者你用的 R2_eff[name] - r2[name]
                            #
                            #     a_curr = (v * F_norm * v).sum()
                            #
                            #     # 2) 每层维护一个 a 的 EMA 作为基线
                            #     if not hasattr(self, "_a_ema"):
                            #         self._a_ema = {}
                            #     rho = getattr(self, "a_ema_rho", 0.1)  # EMA 系数（0.05~0.2）
                            #     a_prev = self._a_ema.get(name, a_curr.detach())
                            #     a_ema = (1 - rho) * a_prev + rho * a_curr.detach()
                            #     self._a_ema[name] = a_ema
                            #
                            #     # 3) 只在“步长计算”里给 a 一个动态下界
                            #     mu = getattr(self, "a_curr_floor_mu", 0.05)  # 0.02~0.2
                            #     kappa = getattr(self, "a_ema_floor_kappa", 0.1)  # 0.05~0.3
                            #     a_eta = torch.maximum(a_raw, torch.maximum(mu * a_curr,
                            #                                                kappa * a_ema.to(a_raw.dtype).to(a_raw.device)))
                            #
                            #     beta_t = torch.as_tensor(self.beta, device=device)
                            #     lr_cap = torch.as_tensor(self.args.learning_rate, device=device)
                            #
                            #     # 冲突项（与闭式一致）：term_u = (b - σ)
                            #     term_u = b_raw - self.sigma
                            #
                            #     # —— 用“冲突强度”决定目标步长是 lr 的几分之几 —— #
                            #     # 归一尺度：√(β a_eta · Δ) ；冲突越强，目标越接近 lr，反之越小
                            #     scale = torch.sqrt(
                            #         torch.clamp(beta_t * a_eta * torch.clamp(Delta, min=0.0) + 1e-12, min=1e-12))
                            #     b_norm = torch.abs(term_u) / scale  # 无量纲冲突强度
                            #
                            #     conf_k = getattr(self, "conf_k", 1.0)  # 0.5~2.0：对冲突强度的敏感度
                            #     eta_min_frac = getattr(self, "eta_min_frac", 0.01)  # 0.05~0.3：最小占比，给很弱冲突一个小步长
                            #     # 映射到 (eta_min_frac, 1)：平滑且有界
                            #     w_conf = eta_min_frac + (1.0 - eta_min_frac) * (b_norm * conf_k / (1.0 + b_norm * conf_k))
                            #
                            #     eta_goal_dyn = lr_cap * w_conf  # 动态目标步长（≤ lr）
                            #
                            #     # —— 精确反推 Δ_goal（包含 b 项；比 “η^2 β a” 更精确）—— #
                            #     # 由闭式反解：η* = ((b-σ)+√((b-σ)^2 + β a (Δ-τ)))/(β a)
                            #     # 推得：Δ_goal = β a η*^2 - 2 (b-σ) η* + τ
                            #     Delta_goal_exact = beta_t * a_eta * (
                            #                 eta_goal_dyn ** 2) - 2.0 * term_u * eta_goal_dyn + self.tau
                            #
                            #     # 仅用于“求步长”的 Δ（不改你的遗忘几何）
                            #     Delta_step = torch.clamp_min(torch.minimum(Delta, Delta_goal_exact), 0.0)
                            #
                            #     EPS = 1e-8
                            #     if b_raw.item() < 0.0:
                            #         disc = term_u * term_u + beta_t * a_eta * (Delta_step - self.tau)
                            #         disc = torch.clamp(disc, min=0.0)
                            #         eta_closed = (term_u + torch.sqrt(disc)) / (beta_t * a_eta + EPS)
                            #
                            #         # 信赖域上界仍用原 Δ 与 a_raw（保持几何）
                            #         eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))
                            #
                            #         eta = torch.minimum(torch.minimum(eta_closed, eta_tr), lr_cap)
                            #         conf[name] += 1
                            #         logger.info(f"[{name}] eta={float(eta):.6g} w_conf={float(w_conf):.3f} "
                            #                     f"goal={float(eta_goal_dyn):.3g} b_norm={float(b_norm):.3g}")
                            #     else:
                            #         eta_tr = torch.sqrt(torch.clamp(Delta, min=0.0) / (a_raw + EPS))
                            #         eta = torch.minimum(lr_cap, eta_tr)
                            #
                            #
                            #     # 新任务收益（二阶）
                            #     Q = torch.sum(g * v)
                            #     gain = (eta - 0.5 * eta * eta) * Q
                            #     gain = torch.clamp(gain, min=0.0)
                            #
                            #     # C = a  # 自然梯度等价：C = v^T f_eff v == a
                            #     # gain_2 = torch.clamp(eta * Q - 0.5 * eta * eta * C, min=0.0)
                            #     #
                            #     # if gain == gain_2:
                            #     #     pass
                            #
                            #     B_round[name] = B_round[name] + gain
                            #
                            #     # 半径更新：r^2 = r^2 - 2η b + η^2 a
                            #     # r2[name] = r2[name] - 2.0 * eta * b_val + (eta * eta) * a
                            #
                            #     r2[name] = r2[name] - 2.0 * eta * b_raw + (eta * eta) * a_raw
                            #
                            #     # 参数更新
                            #     p.add_(-eta * v)
                        else:
                            # --------------------------
                            # 基线模式：仅保留“手动SGD更新”（无任何自适应逻辑）
                            # --------------------------
                            for name, p in model.named_parameters():
                                if not p.requires_grad or p.grad is None:
                                    continue
                                # 基线：无Fisher归一化，v=原始梯度；无动态步长，eta=固定学习率
                                v_baseline = p.grad
                                eta_baseline = lr_base
                                # 与自适应模式相同的手动更新方式（保证公平性）
                                p.add_(-eta_baseline * v_baseline)

                    # ==============================================
                    # 仅添加：记录当前batch耗时（跳过热身batch）
                    # ==============================================
                    batch_end = time.perf_counter()
                    batch_time_ms = (batch_end - batch_start) * 1000  # 转毫秒
                    # 跳过热身batch（第一个batch可能含GPU初始化耗时）
                    if (epoch == 0 and step >= warmup_skip) or epoch > 0:
                        batch_times.append(batch_time_ms)
                        # 每10个batch打印一次耗时（可选，便于实时观察）
                        if step % 10 == 0:
                            avg_curr = np.mean(batch_times) if batch_times else 0.0
                            logger.info(f"Batch [{step + 1}] 耗时：{batch_time_ms:.2f} ms | 累计平均：{avg_curr:.2f} ms")

            state_dict = model.state_dict()
            if len(batch_times) > 0:
                avg_batch_time = np.mean(batch_times)
                std_batch_time = np.std(batch_times)

                # ==============================================
                # 关键修改：文件名加入客户端ID（cid），确保唯一性
                # ==============================================
                # 文件名格式：[模式]_task[任务ID]_cid[客户端ID]_batch_times.npy
                save_filename = (
                    f"{'adaptive' if use_adaptive_logic else 'baseline'}"
                    f"_task{task_id}_cid{cid}_batch_times.npy"
                )
                save_path = os.path.join(self.args.output_dir, save_filename)
                np.save(save_path, batch_times)

                # 日志中增加客户端ID信息，便于追踪
                logger.info(
                    f"\n===== 训练结束 | 模式：{'自适应' if use_adaptive_logic else '基线'} | 客户端ID：{cid} =====")
                logger.info(f"总有效batch数：{len(batch_times)}")
                logger.info(f"平均batch耗时：{avg_batch_time:.2f} ms（±{std_batch_time:.2f} ms）")
                logger.info(f"时间数据保存至：{save_path}")
            else:
                logger.warning(f"客户端ID：{cid} 无有效batch时间记录（可能所有batch都被视为热身）")
                avg_batch_time = 0.0

            delta = {}
            F_client = {}
            # 新增：通信时间计算相关参数（带宽单位：MB/s，默认100MB/s；固定开销单位：秒）
            comm_bandwidth = self.comm_bandwidth if hasattr(self, 'comm_bandwidth') else 100.0
            comm_fixed_cost = self.comm_fixed_cost if hasattr(self, 'comm_fixed_cost') else 0.1
            baseline_comm_time = 0.0
            adaptive_comm_time = 0.0
            save_time = 0.0

            # 辅助函数：计算通信数据量（字节）和时间（秒）
            def calculate_comm_metrics(param_dict: Dict[str, torch.Tensor]) -> Tuple[float, int]:
                total_bytes = 0
                for param in param_dict.values():
                    # 计算参数总字节数（元素数 × 每个元素字节数，float32为4字节）
                    total_bytes += param.numel() * param.element_size()
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
                    costs.append(int(self.layer_costs.get(name, 1)))
                    names.append(name)
                selected = _knapsack(values, costs, self.comm_budget)

                # 1) 自适应模式：仅传输选中层的delta
                state_dict = model.state_dict()
                adaptive_delta = {}
                for name, sel in zip(names, selected):
                    local_param = state_dict[name].detach().cpu()
                    # 仅选中的层传输真实delta，未选中的层传输零（不实际传输，仅占位）
                    adaptive_delta[name] = (base_params[name] - local_param) if sel else torch.zeros_like(local_param)
                delta = adaptive_delta

                # 计算自适应模式通信时间
                adaptive_comm_time, adaptive_bytes = calculate_comm_metrics(
                    {k: v for k, v in delta.items() if not v.equal(torch.zeros_like(v))}  # 过滤零值delta
                )

                # 2) 计算基线模式通信时间（传输所有可训练参数的delta）
                baseline_delta = {}
                for name, p in model.named_parameters():
                    if name in base_params and p.requires_grad:
                        local_param = state_dict[name].detach().cpu()
                        baseline_delta[name] = base_params[name] - local_param
                baseline_comm_time, baseline_bytes = calculate_comm_metrics(baseline_delta)

                # 计算节省的通信时间
                save_time = baseline_comm_time - adaptive_comm_time

                # 日志输出通信统计
                logger.info(
                    f"通信统计 - 自适应模式: {adaptive_bytes / (1024 * 1024):.2f} MB, 时间: {adaptive_comm_time:.4f}秒\n"
                    f"通信统计 - 基线模式: {baseline_bytes / (1024 * 1024):.2f} MB, 时间: {baseline_comm_time:.4f}秒\n"
                    f"通信节省时间: {save_time:.4f}秒 (节省比例: {save_time / baseline_comm_time * 100:.2f}%)"
                )

                # 3) 本轮Fisher计算
                F_client = {k: v.detach().cpu() for k, v in F_curr.items()}
            else:
                # 基线模式：传输所有可训练参数的delta
                state_dict = model.state_dict()
                baseline_delta = {}
                for name, p in model.named_parameters():
                    if name in base_params and p.requires_grad:
                        local_param = state_dict[name].detach().cpu()
                        baseline_delta[name] = base_params[name] - local_param
                delta = baseline_delta

                # 计算基线模式通信时间（自适应模式未启用，节省时间为0）
                baseline_comm_time, baseline_bytes = calculate_comm_metrics(baseline_delta)
                adaptive_comm_time = 0.0

                logger.info(
                    f"通信统计 - 基线模式: {baseline_bytes / (1024 * 1024):.2f} MB, 时间: {baseline_comm_time:.4f}秒"
                )

                F_client = {k: torch.zeros_like(v) for k, v in base_params.items()}

            # 4) 本轮结束时该客户端的θ*（只取可训练/LoRA参数）
            theta_last = {k: state_dict[k].detach().cpu() for k in F_client.keys()}

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
