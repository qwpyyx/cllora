import logging
import os
import time
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

# logger = logging.getLogger(__name__)

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
    if total_steps > 0:
        for name in fisher:
            fisher[name] = fisher[name] / total_steps
    else:
        logger.warning("未记录到有效梯度步骤，Fisher保持初始零值")

    # 转移到CPU并与base_params对齐（兼容原有返回格式）
    return {k: v.detach().cpu() for k, v in fisher.items()}


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
        alpha: float = 0.1,
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
        self._force_sgd = False
        self.fisher_floor = getattr(self.args, "fisher_floor", 1e-3)

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

        if self.state.global_step % 1 == 0:  # 每10步打印一次损失
            self.log({"train_loss": loss.item()})  # 关键修改：用字典封装指标


        return loss.detach()

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
            # batch_times = []  # 存储每个batch的训练耗时（毫秒）
            # warmup_skip = 1  # 跳过第一个batch（GPU热身时间，避免干扰统计）
            logger.info(f"===== 开始训练 | 模式：{'自适应' if use_adaptive_logic else '基线'} | Task {task_id} =====")

            model = self.model
            dataloader = self.get_train_dataloader()
            device = next(model.parameters()).device
            model.train()

            # 工程
            F_curr = None
            bar_F_raw = None
            bar_theta = {}
            r2, r2_start = {}, {}
            B_round, conf = {}, {}

            if use_adaptive_logic:

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
            eta_save = []
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

                                # Step 1
                                g = p.grad
                                F_batch = g * g
                                F_curr[name] = self.alpha * F_curr[name] + (1 - self.alpha) * F_batch
                                #F_curr_safe = F_curr[name] + self.fisher_floor # 算法隐含：避免F_curr为0导致除零
                                F_curr_safe = F_curr[name]
                                # Step 2
                                # v = g / F_curr_safe
                                # 2. 过去任务Fisher（bar_F^(ℓ)）
                                f_past = bar_F_raw.get(name, torch.zeros_like(p))

                                # Fisher 信息的绝对量级不影响 “参数重要性的区分”，仅相对比例有意义
                                if torch.any(f_past > 0):
                                    f_past_mean = torch.mean(f_past)
                                    if f_past_mean > 0:
                                        f_past = f_past / f_past_mean  # 归一化，让f_past均值为1


                                scale = 1.0
                                if torch.is_tensor(f_past) and torch.numel(f_past) > 0 and torch.any(f_past > 0):
                                    curr_mean = torch.mean(F_curr_safe).item()
                                    hist_mean = torch.mean(f_past).item()
                                    if hist_mean > 0:
                                        prev_scale = self._eta_scale.get(name, 1.0)
                                        ratio = curr_mean / (hist_mean + self.fisher_floor)
                                        scale = (1 - self._eta_scale_rho) * prev_scale + self._eta_scale_rho * ratio
                                        scale = max(self._eta_smin, min(self._eta_smax, scale))
                                        self._eta_scale[name] = scale
                                preconditioner = F_curr_safe / scale
                                # preconditioner = torch.clamp(preconditioner, min=self.fisher_floor)

                                v = g / preconditioner

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

                                scale_tensor = torch.as_tensor(scale, device=device, dtype=p.dtype)
                                # v_alg 对应“算法中的 v_B^(ℓ)”，后续 a/b/Q 等均基于该无缩放版本计算，
                                # 只在最终更新时再与 scale_tensor 结合，保持量纲一致。
                                # v_alg = v_scaled / scale_tensor
                                v_alg = v_scaled

                                # 3. 当前参数与过去最优参数的差：Δθ = θ^(ℓ) - bar_theta^(ℓ)
                                delta_theta = p - bar_theta.get(name, torch.zeros_like(p))
                                # 4. a_B^(ℓ) = (v_B^(ℓ))^T * bar_F^(ℓ) * v_B^(ℓ)
                                a_raw_alg = (v_alg * f_past * v_alg).sum()
                                # 5. b_B^(ℓ) = (v_B^(ℓ))^T * bar_F^(ℓ) * Δθ
                                b_raw_alg = (v_alg * f_past * delta_theta).sum()
                                # 6. 半径余量 Δ_ℓ = R² - r²_ℓ
                                Delta = (self.radius ** 2) - r2[name]

                                if Delta.item() <= self.tau:
                                    eta_alg = torch.zeros(1, device=device, dtype=p.dtype).squeeze()
                                    eta = eta_alg
                                else:
                                    a_safe = torch.clamp(a_raw_alg, min=1e-12)
                                    Delta_eff = torch.clamp(Delta - self.tau, min=0.0)

                                    # --------------------------
                                    # 算法步骤8：冲突判断与步长计算
                                    # --------------------------
                                    if b_raw_alg.item() < 0.0:  # 冲突层：b_B^(ℓ) < 0（新旧任务方向冲突）
                                        # 闭式步长公式（严格按算法）：
                                        term_u = b_raw_alg - self.sigma
                                        discriminant = term_u ** 2 + self.beta * a_safe * Delta_eff
                                        discriminant = torch.clamp(discriminant, min=0.0)  # 确保开方非负
                                        eta_closed = (term_u + torch.sqrt(discriminant)) / (self.beta * a_safe + 1e-6)
                                        # 步长上限：不超过初始学习率 η₀
                                        eta_alg = torch.minimum(
                                            eta_closed,
                                            torch.tensor(self.args.learning_rate, device=device, dtype=p.dtype),
                                        )
                                        if eta_alg.item() > 0:
                                            conf[name] += 1
                                    else:
                                        eta_trust = torch.sqrt(Delta_eff / (a_safe + 1e-6))
                                        eta_alg = torch.clamp(
                                            eta_trust,
                                            max=torch.as_tensor(self.args.learning_rate, device=device, dtype=p.dtype),
                                        )

                                    # 确保 eta 为标量并非 NaN
                                    if torch.isnan(eta_alg):
                                        eta_alg = torch.zeros(1, device=device, dtype=p.dtype).squeeze()

                                    eta = eta_alg / scale_tensor

                                # --------------------------
                                # 算法步骤9：累计收益 B^round_ℓ
                                # --------------------------
                                # 二阶收益 Q_B^(ℓ) = (g_B^(ℓ))^T * (F_curr^(ℓ))^(-1) * g_B^(ℓ)（等价于 g·v）
                                Q_alg = torch.sum(g * v_alg)
                                # 收益计算：max{0, (η - 0.5η²) * Q}
                                gain = (eta_alg - 0.5 * eta_alg ** 2) * Q_alg
                                gain = torch.clamp(gain, min=0.0)  # 收益非负
                                B_round[name] = B_round[name] + gain

                                # --------------------------
                                # 算法步骤10：更新马氏半径 r²_ℓ
                                # --------------------------
                                r2[name] = r2[name] - 2.0 * eta_alg * b_raw_alg + (eta_alg ** 2) * a_raw_alg

                                # --------------------------
                                # 算法步骤11：更新参数 θ^(ℓ) = θ^(ℓ) - η·v_B^(ℓ)
                                # --------------------------
                                if eta.item() != 0.0:
                                    p.add_(-eta * v_scaled)  # 修正：使用缩放后的v_scaled

                                eta_save.append(eta)


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
