#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning the library models for sequence to sequence.
"""

# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import datasets
import nltk
import numpy as np
from datasets import load_dataset
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

# ===== Compatibility: transformers 4.51+ renamed evaluation_strategy → eval_strategy =====
_TRANSFORMERS_VERSION = tuple(int(x) for x in transformers.__version__.split(".")[:2])
_TRANSFORMERS_NEW_EVAL = _TRANSFORMERS_VERSION >= (4, 51)
if _TRANSFORMERS_NEW_EVAL:
    for _i, _arg in enumerate(sys.argv):
        if _arg == "--evaluation_strategy":
            sys.argv[_i] = "--eval_strategy"

# privacy
from uie_collator import DataCollatorForUIE
from uie_dataset_lora import gen_cache_path
from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask

# ignore all warning
# warnings.filterwarnings("ignore")
os.environ['WANDB_DISABLED'] = "True"
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


def _configure_decoder_tokenizer_and_config(tokenizer, config):
    """Use model-native BOS/EOS and add a safe PAD token for decoder-only models.

    This avoids the old LLaMA-2-only hard-code bos=1/eos=2/pad=1.  For
    Llama-3.x, prefer an existing reserved padding token that is distinct
    from all native EOS ids, so generation can use a reliable attention mask.
    """
    eos_ids = getattr(config, "eos_token_id", None)
    if isinstance(eos_ids, (list, tuple, set)):
        eos_id_set = set(int(x) for x in eos_ids)
    elif eos_ids is not None:
        eos_id_set = {int(eos_ids)}
    else:
        eos_id_set = set()

    def _valid_existing_token(tok: str) -> bool:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            return False
        if tid is None:
            return False
        if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id:
            return False
        return int(tid) >= 0 and int(tid) not in eos_id_set

    if tokenizer.pad_token is None or (tokenizer.pad_token_id is not None and int(tokenizer.pad_token_id) in eos_id_set):
        chosen_pad = None
        for cand in ("<|finetune_right_pad_id|>", "<|reserved_special_token_0|>", "<|pad|>"):
            if _valid_existing_token(cand):
                chosen_pad = cand
                break
        if chosen_pad is not None:
            tokenizer.pad_token = chosen_pad
        elif tokenizer.unk_token is not None and (tokenizer.unk_token_id is None or int(tokenizer.unk_token_id) not in eos_id_set):
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "left"

    if getattr(config, "bos_token_id", None) is None and tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    if getattr(config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id


def _sync_generation_special_tokens(model, tokenizer, config):
    if not hasattr(model, "generation_config"):
        return
    if getattr(model.generation_config, "bos_token_id", None) is None:
        model.generation_config.bos_token_id = getattr(config, "bos_token_id", None) or tokenizer.bos_token_id
    if getattr(model.generation_config, "eos_token_id", None) is None:
        model.generation_config.eos_token_id = getattr(config, "eos_token_id", None) or tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_method: Optional[str] = field(
        default=None, metadata={"help": "T5/roberta/llama"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )

    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )

    use_baseline_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use a single LoRA configuration for all tasks (baseline LoRA)."}
    )

    ues_flash_attention: bool = field(
        default=True,
        metadata={
            "help": "If True, ues flash attention for llama model."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    task: int = field(default=1, metadata={
        "help": "Current task index starting from 1; first task runs without continual constraints."})


@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})

    # Compatibility: transformers 4.51+ removed evaluation_strategy, renamed to eval_strategy.
    # This property bridges the two so existing code (args.evaluation_strategy) continues to work.
    @property
    def evaluation_strategy(self):
        return self.eval_strategy

    @evaluation_strategy.setter
    def evaluation_strategy(self, value):
        self.eval_strategy = value

    lamda_1: float = field(default=0.5)
    lamda_2: float = field(default=0)
    method: str = field(
        default="lora_origin",
        metadata={
            "help": (
                "Training/compression method. Continual methods: lora_origin, adaptive, ewc, replay, gem, pilora, lorm. "
                "Migrated FLM-TopK baselines: flasc/topk, compeft/topk_pq, fedcomp, flm_topk/block_opt."
            )
        },
    )

    # ===== Migrated compression baselines from the old FLM-TopK framework =====
    baseline_packet_num: int = field(
        default=0,
        metadata={
            "help": (
                "Number of 1500-byte packets used by migrated compression baselines. "
                "0 means reuse federated_args.comm_budget as the per-client packet budget."
            )
        },
    )
    baseline_blocks: int = field(
        default=1024,
        metadata={"help": "Initial interval/block number for FLM-TopK/block_opt. ComPEFT uses global TopK+PQ and does not use blocks."},
    )
    baseline_bit: int = field(
        default=18,
        metadata={"help": "Maximum value bit length used by ComPEFT/FLM-TopK quantization."},
    )
    baseline_min_bit: int = field(
        default=4,
        metadata={"help": "Minimum value bit length used by FLM-TopK packet optimization."},
    )
    baseline_topk_method: str = field(
        default="gradient",
        metadata={"help": "TopK score used by old framework: gradient, graproduct, graproduct_2, or adalora."},
    )
    baseline_flm_opt_max_iter: int = field(
        default=40,
        metadata={"help": "Maximum optimization iterations for FLM-TopK/block_opt block-size search. Old slow behavior used about 1000."},
    )
    baseline_flm_max_blocks: int = field(
        default=256,
        metadata={"help": "Cap optimized block count for FLM-TopK/block_opt. <=0 disables the cap."},
    )
    baseline_flm_disable_optim: bool = field(
        default=False,
        metadata={"help": "If True, skip FLM-TopK block-size optimization and use baseline_blocks directly."},
    )
    fedcomp_use_residual: bool = field(
        default=True,
        metadata={"help": "If True, mimic old FedComp residual replay: local init = global - previous residual."},
    )

    uplink_mbps: str = field(default="10,100", metadata={"help": "逗号分隔的上行带宽(Mbps)，用于计算通信节省时间下界"})
    packet_bytes: int = field(default=1500, metadata={"help": "每个传输包的有效负载字节数"})
    radius: float = field(default=1.0, metadata={"help": "Constraint radius for adaptive optimizer."})
    # optim: str = field(default="sgd", metadata={"help": "The method for CL: [lora_origin, adaptive]."})
    vartheta: float = field(
            default = 0.3,
        metadata = {"help": "RieSelect Eq. (8) estimation-error margin vartheta."}
                        )
    varsigma: float = field(
            default = 0.3,
        metadata = {"help": "RieSelect Eq. (8) estimation-error margin varsigma."}
                        )
    beta_eps: float = field(
            default = 1e-12,
        metadata = {"help": "Numerical epsilon used in Eq. (9) beta-hat denominator."}
                        )
    random_layer_selection: bool = field(
        default=False,
        metadata={
            "help": "If True, select uploaded layers randomly within budget instead of using knapsack optimization."}
    )

    # [新增] 消融实验开关：禁用自适应学习率 (Step 2)，但在 Adaptive 方法下仍计算收益供 Step 3 使用
    ablation_no_adaptive_lr: bool = field(
        default=False,
        metadata={
            "help": "If True, use fixed learning rate (AdamW behavior) but keep calculating B_round for Knapsack."}
    )

    # For ewc
    ewc_lambda: float = field(
        default=5000.0,  # 默认值根据经验设定，通常 LoRA 需要较大的正则化系数
        metadata={"help": "EWC regularization coefficient."}
    )

    # For replay
    replay_buffer_size: int = field(
        default=10,
        metadata={"help": "Number of samples to keep from previous tasks for experience replay."}
    )

    # For A-gem
    gem_gamma: float = field(
        default=0.5,
        metadata={"help": "Margin for GEM projection (usually 0.5). Only used when method='gem'."}
    )

    # For pilora
    pilora_lambda_ortho: float = field(default=0.1)
    pilora_reg_targets: str = field(default="A,B")   # "A" or "B" or "A,B"
    pilora_normalize: bool = field(default=False)

    upload_atomic_mode: str = field(
        default="tensor",
        metadata={
            "help": (
                "Upload atomicity mode for lora_origin sparse upload. "
                "Choices: tensor, ab_pair, qv_block. "
                "tensor = original tensor-level upload; "
                "ab_pair = upload LoRA A/B of the same module together; "
                "qv_block = upload q_proj/v_proj LoRA tensors in the same transformer layer together."
            )
        }
    )


    upload_score_mode: str = field(
        default="factor_norm",
        metadata={
            "help": (
                "Score used for lora_origin sparse upload selection. "
                "Choices: factor_norm, effective_norm, sn_p1p2. "
                "factor_norm = current A/B-pair score sqrt(||U_A||^2+||U_B||^2); "
                "effective_norm = rank A/B pairs by ||B U_A + U_B A + U_B U_A||_F; "
                "sn_p1p2 = server-side signal-noise P1 allocation plus gap-aware P2 assignment. "
                "effective_norm is supported for ab_pair; sn_p1p2 is supported for ab_pair and qv_block."
            )
        }
    )

    sn_gap_eta: float = field(
        default=1.0,
        metadata={"help": "Gap-aware redundancy coefficient eta for upload_score_mode=sn_p1p2."}
    )

    sn_force_full_budget: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, P1 keeps allocating units until the total budget is exhausted even when "
                "marginal gains become non-positive. Default False follows the positive-marginal theory."
            )
        }
    )

    sn_min_signal_eps: float = field(
        default=1e-12,
        metadata={"help": "Numerical floor used when clipping estimated signal/noise in sn_p1p2."}
    )

    sn_save_diagnostics: bool = field(
        default=True,
        metadata={"help": "Save per-round signal/noise quota and assignment diagnostics for sn_p1p2."}
    )

    sn_p1_norm_mode: str = field(
        default="raw",
        metadata={
            "help": (
                "Normalization mode for P1 signal-noise allocation. "
                "Choices: raw/none, rank, depth_rank, depth_balanced. "
                "depth_rank rank-normalizes a_m and b_m inside depth groups; "
                "depth_balanced additionally reserves P1 quota for lower/middle/upper depth groups."
            )
        }
    )

    sn_depth_group_ratios: str = field(
        default="1,1,2",
        metadata={
            "help": (
                "Comma-separated lower,middle,upper P1 quota ratios used when "
                "sn_p1_norm_mode=depth_balanced. Default 1,1,2."
            )
        }
    )

    # ===== Ours + fine-grained encoding variants =====
    # The original Ours uploads selected qv-blocks as raw LoRA tensors. These knobs
    # let SN-P1P2 first choose a larger candidate set, then apply value-level
    # encoding inside the selected candidates under the original packet budget.
    sn_encoder_mode: str = field(
        default="none",
        metadata={"help": "Encoding inside SN-P1P2 selected candidates: none/raw, compeft/topk_pq, or flasc/topk."},
    )
    sn_candidate_budget_multiplier: float = field(
        default=1.0,
        metadata={"help": "Multiplier applied to comm_budget only for SN-P1P2 candidate scheduling before encoding."},
    )
    sn_candidate_budget: int = field(
        default=0,
        metadata={"help": "Absolute candidate scheduling budget for SN-P1P2. If >0, overrides multiplier."},
    )
    sn_encoder_packet_num: int = field(
        default=0,
        metadata={"help": "Packet budget for SN encoder. 0 means reuse comm_budget."},
    )
    sn_encoder_bit: int = field(
        default=18,
        metadata={"help": "Value bit length used by SN encoder when mode=compeft/topk_pq."},
    )
    sn_encoder_min_bit: int = field(
        default=4,
        metadata={"help": "Minimum bit length reserved for compatibility with BaselineCompressor."},
    )
    sn_encoder_blocks: int = field(
        default=192,
        metadata={"help": "Reserved for future block encoders; ComPEFT/FLASC encoder variants ignore it."},
    )

    upload_diversity_mode: str = field(
        default="none",
        metadata={
            "help": (
                "System-level diversity mode for lora_origin sparse upload. "
                "Choices: none, group_mask, coverage_penalty. "
                "none = independent client-side Top-K; "
                "group_mask = assign different clients to different LoRA unit groups; "
                "coverage_penalty = penalize units already selected by previous clients in the same round."
            )
        }
    )

    diversity_num_groups: int = field(
        default=4,
        metadata={
            "help": (
                "Number of LoRA unit groups used when upload_diversity_mode=group_mask."
            )
        }
    )

    coverage_penalty_beta: float = field(
        default=1.0,
        metadata={
            "help": (
                "Penalty strength used when upload_diversity_mode=coverage_penalty. "
                "The adjusted score is score / (1 + beta * current_round_coverage_count)."
            )
        }
    )

    diagnose_pair_saliency: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, record A/B-pair saliency diagnostics: "
                "factor-norm vs effective-update ranking mismatch and "
                "Top-K instability under equivalent LoRA reparameterization. "
                "This does not affect training, selection, residuals, or aggregation."
            )
        }
    )

    pair_saliency_reparam_scales: str = field(
        default="0.25,0.5,2,4",
        metadata={
            "help": (
                "Comma-separated positive scaling factors used by pair saliency diagnostics. "
                "For each A/B pair, A' = cA and B' = B/c leave BA unchanged but change factor norms."
            )
        }
    )

    pair_saliency_save_top_units: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, save top factor-score and effective-score A/B-pair unit names in diagnostics."
            )
        }
    )

    pair_saliency_top_n: int = field(
        default=20,
        metadata={"help": "Number of top units to save when pair_saliency_save_top_units=True."}
    )

    diagnose_residual_errors: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, record FedLoRA residual diagnostic metrics, including "
                "split error, drift error, and compensation error ratios."
            )
        }
    )

    diagnose_drift_max_age: int = field(
        default=5,
        metadata={
            "help": (
                "Maximum residual age for multi-round drift diagnostics. "
                "Only used when diagnose_residual_errors=True."
            )
        }
    )

    lora_residual_accumulation: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use naive factor-space residual accumulation/error feedback "
                "for sparse LoRA upload. Only used for method='lora_origin'."
            )
        }
    )

    lora_residual_max_age: int = field(
        default=-1,
        metadata={
            "help": (
                "Maximum age of residual tensors. -1 means no age cap, i.e., naive infinite residual accumulation. "
                "Only used when lora_residual_accumulation=True."
            )
        }
    )

    diagnose_residual_replay_gain: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, diagnose whether historical LoRA residuals reduce the current loss "
                "when replayed on the current global LoRA state."
            )
        }
    )

    replay_gain_max_clients_per_round: int = field(
        default=2,
        metadata={
            "help": "Maximum number of selected clients per round used for residual replay-gain diagnosis."
        }
    )

    replay_gain_max_batches: int = field(
        default=1,
        metadata={
            "help": (
                "Number of mini-batches used to estimate replay gain for each residual age bucket."
            )
        }
    )

    replay_gain_min_age: int = field(
        default=1,
        metadata={
            "help": "Minimum residual age considered in replay-gain diagnosis."
        }
    )

    replay_gain_max_age: int = field(
        default=8,
        metadata={
            "help": "Maximum residual age considered in replay-gain diagnosis. -1 means no upper bound."
        }
    )

    replay_gain_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Scaling factor applied to the residual when computing replay gain. "
                "Use 1.0 for local residual utility diagnosis; use a smaller value such as 0.05 "
                "to mimic one-client contribution in aggregation."
            )
        }
    )






@dataclass
class FederatedArguments:
    """Arguments for federated learning scenario."""
    mode: str = field(
        default="federated",
        metadata={"help": "Training mode: centralized or federated"},
    )
    num_clients: int = field(
        default=50,
        metadata={"help": "Total number of clients in federated learning."},
    )
    clients_per_round: int = field(
        default=5,
        metadata={"help": "Number of clients sampled in each round."},
    )
    global_rounds: int = field(
        default=1,
        metadata={"help": "Total number of global federated rounds."},
    )
    local_epochs: int = field(
        default=3,
        metadata={"help": "Local training epochs for each selected client."},
    )
    dirichlet_alpha: float = field(
        default=50,
        metadata={"help": "Dirichlet alpha controlling data heterogeneity."},
    )

    partition_strategy: str = field(
        default="quantity",
        metadata={
            "help": (
                "Client data partition strategy. "
                "quantity = Dirichlet controls only client sample counts; "
                "label = label/category-skew Dirichlet partition using partition_label_key."
            )
        },
    )

    partition_label_key: str = field(
        default="Dataset",
        metadata={
            "help": (
                "Dataset field used as label/category for label-skew Dirichlet partition. "
                "For converted Dolly UIE data, use Dataset, where each Dolly category is written as one Dataset."
            )
        },
    )
    federated_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Seed for client sampling in federated rounds (independent of training_args.seed)."}
    )

    # ---------- Continual FL hyperparameters ----------
    comm_budget: Optional[int] = field(
        default=1200,
        metadata={
            "help": "Maximum upload cost per round. None means all LoRA layers are sent."
        },
    )

    use_arithmetic_fisher: str = field(
        default="False",
        metadata={"help": "Training mode: centralized or federated"},
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments, FederatedArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, federated_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, federated_args = parser.parse_args_into_dataclasses()

    # DDP + LoRA + gradient checkpointing is safer with non-reentrant checkpointing.
    # This mirrors the explicit call inside federated_uie_lora.py and also helps
    # Transformers Trainer versions that read TrainingArguments directly.
    if getattr(training_args, "gradient_checkpointing", False):
        try:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        except Exception:
            pass
        if getattr(training_args, "ddp_find_unused_parameters", None) is None:
            try:
                training_args.ddp_find_unused_parameters = False
            except Exception:
                pass

    # T5 / LLama / Qwen federated path
    if federated_args.mode == "federated":
        from federated_uie_lora import run_federated_training
        run_federated_training(model_args, data_args, training_args, federated_args)
        return


    # --------------------------- O-LoRA ---------------------------
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu},"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)

    # Get the UIE dataset
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples
    )
    raw_datasets.cleanup_cache_files()

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'adapter' in model_args.model_name_or_path:  # load lora-config
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        if 'llama' in model_args.model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            _configure_decoder_tokenizer_and_config(tokenizer, config)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        _configure_decoder_tokenizer_and_config(tokenizer, config)
    else:  # load original config
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # 模型类别设置
    if 'llama' in model_args.model_name_or_path.lower():  # add llama
        model_class = LlamaForCausalLM_with_lossmask
        tokenizer.padding_side = 'left'
    else:
        model_class = AutoModelForSeq2SeqLM

    # 已经有了训练好的 LoRA 适配器参数，此时只需要把这些参数加载进模型中
    if 'adapter' in model_args.model_name_or_path:  # add lora-adapter to the original model
        model = model_class.from_pretrained(config.base_model_name_or_path)
        # 加载 LoRA 适配器，里面有个函数load_adapter
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    # 在现有的模型上 初始化一个新的 LoRA 适配器
    elif 'llama' in model_args.model_name_or_path.lower():
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )

        # 这里修改其他PEFT方法
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

        # 如 prefix tuning
        # from peft import PrefixTuningConfig, get_peft_model
        #
        # peft_config = PrefixTuningConfig(
        #     task_type=TaskType.CAUSAL_LM,  # 任务类型，比如语言模型
        #     num_virtual_tokens=20,  # 添加的虚拟前缀的长度
        #     prefix_projection=False,  # 是否对前缀进行线性投影
        # )
        # model = get_peft_model(model, peft_config)
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32,
            lora_dropout=0.1
        )
        # 应该是修改这部分
        model = get_peft_model(model, peft_config)

    # 确保模型的词嵌入矩阵与 tokenizer 的词汇表大小一致
    model.resize_token_embeddings(len(tokenizer))

    if 'llama' in model_args.model_name_or_path.lower():
        _sync_generation_special_tokens(model, tokenizer, config)
        print("[run_uie_lora] tokenizer.bos/eos/pad =", tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        print("[run_uie_lora] config.bos/eos/pad =", getattr(config, "bos_token_id", None), getattr(config, "eos_token_id", None), getattr(config, "pad_token_id", None))
        print("[run_uie_lora] generation.bos/eos/pad =", getattr(model.generation_config, "bos_token_id", None), getattr(model.generation_config, "eos_token_id", None), getattr(model.generation_config, "pad_token_id", None))

    # fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
    # fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
    # optional: lora_A/B is trainable but should not move too far from lorapre_A/B
    # (constrained in "training_step"[uie_trainer_lora.py])
    for name, param in model.named_parameters():
        # if name.find("loranew_") != -1:
        #     param.requires_grad = True
        if name.find("lora_") != -1:
            param.requires_grad = True
        # this module should always be frozen because we change the vocabulary
        elif name.find("shared") != -1:
            param.requires_grad = False

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
            unique_tasks = set(predict_dataset['Dataset'])
            num_tasks = len(unique_tasks)
            samples_per_task = data_args.max_predict_samples // num_tasks

            # 确保每个任务有足够的样本
            task_datasets = []
            for task in unique_tasks:
                task_data = predict_dataset.filter(lambda example: example['Dataset'] == task)
                task_data = task_data.shuffle(seed=training_args.seed).select(
                    range(min(samples_per_task, len(task_data))))
                task_datasets.append(task_data)

            # 将不同任务的数据集拼接成最终的预测数据集
            from datasets import concatenate_datasets
            predict_dataset = concatenate_datasets(task_datasets)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file
    )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        # 对生成式模型的输出进行后处理
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        # 按类别进行分类，考虑的是所有TC类的准确率
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainer = UIETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # T
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        peft_model_id = training_args.output_dir + "/adapter"
        # 保存训练下来的模型参数和tokenizer
        # trainer.model.save_pretrained(peft_model_id)
        # tokenizer.save_pretrained(peft_model_id)

        if trainer.is_world_process_zero():
            trainer.model.save_pretrained(peft_model_id)
            tokenizer.save_pretrained(peft_model_id)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

    # Evaluation
    results = {}
    # in case the batch is shorter than max length, the output should be padded
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        logger.info("*** Loading CheckPoint ***")

        # if data_args.max_predict_samples is not None:
        #     predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        # train_seq2seq.py
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

    # distributed_state.wait_for_everyone()
    return results


if __name__ == "__main__":
    main()
