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
        radius: float = 1.0,
        sigma: float = 0.0,
        tau: float = 0.0,
        beta: float = 1.0,
        alpha: float = 0.9,
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
        **kwargs,
    ):  # type: ignore[override]
        """Train the model.

        When ``task_id`` is 1, this simply falls back to ``Seq2SeqTrainer.train``
        and acts as normal fine-tuning. For later tasks (``task_id`` > 1) the
        continual-learning optimisation with communication-budget aware
        parameter selection is executed.
        """

        state_has_history = self.continual_state.has_valid_history() if self.continual_state is not None else False

        # 任务1逻辑：仅调用父类训练，不计算delta和F_client（由federated_uie_lora.py处理）
        if task_id == 1:
            return super().train(**kwargs)

        if task_id > 1 and self.args.deepspeed:
            self._init_deepspeed()  # 初始化DeepSpeed引擎

        # 后续任务逻辑（保持不变）
        if self.continual_state is None or base_params is None or (
                self.continual_state is not None and not state_has_history):
            super().train(**kwargs)

            # 补充计算delta和F_client（仅用于后续任务中首次被选中的节点）
            state_dict = self.model.state_dict()
            delta = {
                k: base_params[k] - state_dict[k].detach().cpu()
                for k in base_params
            }

            from federated_uie_lora import compute_fisher
            F_client = compute_fisher(self.model, self.get_train_dataloader())
            F_client = {k: F_client.get(k, torch.zeros_like(base_params[k])) for k in base_params}

            return delta, F_client

        model = self.model
        dataloader = self.get_train_dataloader()
        # self.create_optimizer_and_scheduler(
        #     num_training_steps=len(dataloader) * int(self.args.num_train_epochs)
        # )
        device = next(model.parameters()).device
        model.train()

        # TODO 是否需要考虑归一化的问题
        F_curr = {
            n: torch.zeros_like(p, device=device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        bar_F = {k: v.to(device) for k, v in self.continual_state.bar_F.items()}  # 转移到 device（如 cuda:0）
        bar_B = {k: v.to(device) for k, v in self.continual_state.bar_B.items()}
        bar_theta = {}
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if k in bar_F and k in bar_B:
                bar_theta[k] = bar_B[k] / (bar_F[k] + 1e-8)
            else:
                bar_theta[k] = torch.zeros_like(p, device=device)

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
            B_round[name] = torch.tensor(0.0, device=device)
            conf[name] = 0

        # ====== 进度条与平滑指标设置（新增）======
        num_epochs = int(self.args.num_train_epochs)
        steps_per_epoch = len(dataloader)
        tqdm_total = num_epochs * steps_per_epoch  # 仅用于显示
        show_bar = getattr(self, "is_world_process_zero", lambda: True)()

        pbar = tqdm(
            total=tqdm_total,
            disable=not show_bar,
            dynamic_ncols=True,
            leave=True,
            desc=f"Task {task_id} | training",
        )

        ema_loss = None
        ema_beta = 0.98
        window = deque(maxlen=50)
        log_every = max(1, int(getattr(self.args, "logging_steps", 10)))
        actual_steps = 0  # 真实已处理的 mini-batch 数（用于 p_round 分母）

        for epoch in range(int(self.args.num_train_epochs)):
            for step, batch in enumerate(dataloader):
                actual_steps += 1
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # 清梯度
                model.zero_grad(set_to_none=True)

                with self.compute_loss_context_manager():
                    outputs = model(**batch)
                    loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs["loss"]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # 反向传播
                loss.backward()

                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        g = p.grad
                        F_batch = g * g
                        F_curr[name] = self.alpha * F_curr[name] + (1 - self.alpha) * F_batch

                        # Natural-gradient style preconditioning
                        v = g / (F_curr[name] + 1e-8)

                        a = (v * bar_F[name] * v).sum() if name in bar_F else torch.tensor(0.0, device=device)
                        diff = p - bar_theta[name]
                        b_val = (v * bar_F[name] * diff).sum() if name in bar_F else torch.tensor(0.0, device=device)
                        Delta = (self.radius ** 2) - r2[name]

                        if b_val.item() < 0.0:
                            term = (b_val - self.sigma)
                            disc = term * term + self.beta * a * (Delta - self.tau)
                            disc = torch.clamp(disc, min=0.0)
                            eta = (term + torch.sqrt(disc)) / (self.beta * a + 1e-8)
                            eta = torch.clamp(eta, max=self.args.learning_rate)
                            conf[name] += 1  # ←← 新增：冲突计数
                        else:
                            eta = torch.as_tensor(self.args.learning_rate, device=device)

                        # 新任务收益（二阶）
                        Q = torch.sum((g * g) / (F_curr[name] + 1e-8))
                        gain = (eta - 0.5 * eta * eta) * Q
                        gain = torch.clamp(gain, min=0.0)
                        B_round[name] = B_round[name] + gain

                        # 半径更新：r^2 = r^2 - 2η b + η^2 a
                        r2[name] = r2[name] - 2.0 * eta * b_val + (eta * eta) * a

                        # 参数更新
                        p.add_(-eta * v)

                if show_bar:
                    # 显示“未缩放”的loss更直观：若用了梯度累积，把显示值乘回去
                    display_loss = loss.detach()
                    if self.args.gradient_accumulation_steps > 1:
                        display_loss = display_loss * self.args.gradient_accumulation_steps
                    loss_val = float(display_loss)

                    # EMA + 滑窗
                    ema_loss = loss_val if ema_loss is None else (ema_beta * ema_loss + (1 - ema_beta) * loss_val)
                    window.append(loss_val)
                    mean_last = sum(window) / len(window)

                    # 控制刷新频率
                    if (actual_steps % log_every) == 0 or actual_steps == 1:
                        pbar.set_postfix_str(
                            f"epoch={epoch + 1}/{num_epochs}  loss={loss_val:.4f}  ema={ema_loss:.4f}  last{len(window)}={mean_last:.4f}"
                        )
                    pbar.update(1)

            # # （可选）回传到 HF 的日志/回调体系，便于 TensorBoard
            # if (actual_steps % log_every) == 0 or actual_steps == 1:
            #     try:
            #         fractional_epoch = epoch + (step + 1) / max(1, steps_per_epoch)
            #         self.log({"loss": float(loss.detach() * (self.args.gradient_accumulation_steps if self.args.gradient_accumulation_steps > 1 else 1)),
            #                   "ema_loss": float(ema_loss) if ema_loss is not None else None,
            #                   "epoch": fractional_epoch})
            #         if hasattr(self, "callback_handler"):
            #             self.control = self.callback_handler.on_log(self.args, self.state, getattr(self, "control", None))
            #         if hasattr(self, "state"):
            #             # 给 HF 的 global_step 也累计一下（可选）
            #             self.state.global_step = (self.state.global_step or 0) + 1
            #     except Exception:
            #         pass

        # 训练循环结束后
        if show_bar:
            pbar.close()

        # ====== 轮末统计 ======
        F_round = {n: 0.5 * torch.clamp(r2[n] - r2_start[n], min=0.0) for n in r2}
        p_round = {n: conf[n] / max(actual_steps, 1) for n in conf}  # ←← 用真实步数
        values, costs, names = [], [], []
        for name in B_round:
            val = (B_round[name] - self.lambda_conf * p_round[name] * F_round[name]).item()
            values.append(val)
            costs.append(int(self.layer_costs.get(name, 1)))
            names.append(name)
        selected = _knapsack(values, costs, self.comm_budget)

        # 增量
        delta = {}
        state_dict = model.state_dict()
        for name, sel in zip(names, selected):
            local_param = state_dict[name].detach().cpu()
            delta[name] = (base_params[name] - local_param) if sel else torch.zeros_like(local_param)
        return delta, {k: v.detach().cpu() for k, v in F_curr.items()}

    #TODO 测试有问题，没结果，是否是训练参数也没更新
    # prediction_step
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
#
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
