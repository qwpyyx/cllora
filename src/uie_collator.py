import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Union
import torch
from transformers.data.data_collator import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ("gpt", "opt", "llama", "mpt", "qwen", "baichuan")
SUPPORTED_SEQ2SEQ_MODELS = ("t5", "bart", "mbart", "pegasus", "t5.1.1")


def _check_model_name(name: str, candidates) -> bool:
    name = (name or "").lower()
    return any(tag in name for tag in candidates)


@dataclass
class DataCollatorForUIE:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    model_name_or_path: Optional[str] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None
    # 新增：用于缓存 decoder_start_token_id，避免持有 model 引用
    decoder_start_token_id: Optional[int] = None

    def __post_init__(self):
        """
        初始化后立即提取必要信息并断开对 model 的引用，防止显存泄露。
        """
        if self.model is not None:
            # 1. 提取模型名称 (用于判断是 seq2seq 还是 decoder-only)
            if self.model_name_or_path is None:
                if hasattr(self.model, "config"):
                    self.model_name_or_path = getattr(self.model.config, "_name_or_path", "")
                else:
                    self.model_name_or_path = str(self.model.__class__.__name__)

            # 2. 提取 decoder_start_token_id (仅 Seq2Seq 需要，如 T5)
            if self.decoder_start_token_id is None and hasattr(self.model, "config"):
                self.decoder_start_token_id = getattr(self.model.config, "decoder_start_token_id", None)
                # T5 fallback: 如果没有 decoder_start_token_id，通常使用 pad_token_id
                if self.decoder_start_token_id is None:
                    self.decoder_start_token_id = getattr(self.model.config, "pad_token_id", None)

            # 3. 关键：断开对 GPU 模型的引用！
            self.model = None

    def __call__(self, batch: List[Dict], return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        # 使用提取好的 model_name_or_path，不再访问 self.model.config
        model_name = self.model_name_or_path or getattr(self.tokenizer, "name_or_path", "") or ""

        if _check_model_name(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif _check_model_name(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            # 默认走 seq2seq（T5/BART）
            # logger.warning(f"[UIE Collator] Unknown model family for '{model_name}', defaulting to seq2seq.")
            model_inputs = self.seq2seq_call(batch, return_tensors)
        return model_inputs

    def get_instruction(self, instance):
        # "instructions \n options \n {0} \n Answer: "
        instruction = instance['Instance']["instruction"]
        content = instance['Instance']['sentence']

        # add task/ds prefix
        prefix = ''
        if self.add_task_name:
            prefix += "Task:" + instance['Task'] + '\n'
        if self.add_dataset_name:
            ds_name = self.common_dataset_name if self.common_dataset_name else instance['Dataset']
            prefix = prefix + "Dataset:"
            prefix = prefix + ds_name + '\n' if prefix else instance['Dataset'] + '\n'
        if prefix:
            instruction = prefix + instruction

        try:
            instruction = instruction.format(content)
        except Exception:
            pass  # 保持原逻辑容错
        return instruction

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        手动实现 shift right 逻辑，替代 model.prepare_decoder_input_ids_from_labels
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if self.max_source_length and len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            elif self.max_source_length:
                sources.append(
                    self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            else:
                sources.append(source)

        if self.text_only:
            model_inputs = {"inputs": sources, "labels": labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = labels["attention_mask"].bool()
            model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

            # 核心修改：使用本地方法生成 decoder_input_ids，不再依赖 self.model
            if self.decoder_start_token_id is not None:
                decoder_input_ids = self.shift_tokens_right(
                    model_inputs["labels"],
                    self.tokenizer.pad_token_id,
                    self.decoder_start_token_id
                )
                model_inputs["decoder_input_ids"] = decoder_input_ids
            else:
                # 只有当无法获取 decoder_start_token_id 时（非常罕见），尝试回退或报警
                # 对于 T5/BART，decoder_start_token_id 是必须的
                pass

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        sources = []
        label_lens = []
        labels = []
        max_len = -1

        # 安全获取 max_source/target_length
        max_src_len = self.max_source_length or 512
        max_tgt_len = self.max_target_length or 50

        if batch[0]['subset'] == "train":
            limit_input_len = max_src_len + max_tgt_len
        else:
            limit_input_len = max_src_len

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            # add bos and eos
            # 注意：部分 tokenizer 可能没有 bos_token，需要处理
            bos = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
            eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ""

            task_input = bos + instruction
            label = label + eos

            tokenized_input = self.tokenizer(task_input)["input_ids"]
            tokenized_label = self.tokenizer(label)["input_ids"]

            # (input) for inference, (input + label) for training
            if instance['subset'] in ['dev', 'test']:
                label_lens.append(0)
                if len(tokenized_input) <= limit_input_len:
                    max_len = max(len(tokenized_input), max_len)
                    sources.append(task_input)
                else:
                    max_len = limit_input_len
                    input_wo_label = self.tokenizer.decode(
                        tokenized_input[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_wo_label)
            else:
                if len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    max_len = max(len(tokenized_input) + len(tokenized_label), max_len)
                    label_lens.append(len(tokenized_label))
                    sources.append(task_input + label)
                else:
                    max_len = max_src_len  # 使用安全获取的 max_src_len
                    input_w_label = self.tokenizer.decode(
                        (tokenized_input + tokenized_label)[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_w_label)
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))

        if self.text_only:
            model_inputs = {"inputs": sources, 'labels': labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )

            label_mask = model_inputs["attention_mask"].bool()
            model_inputs["labels"] = model_inputs['input_ids'].masked_fill(~label_mask, self.label_pad_token_id)

            # loss mask
            max_len = min(max_len, limit_input_len)
            loss_mask = torch.ones((label_mask.shape))
            for k, label_len in enumerate(label_lens):
                if max_len - label_len - 1 > 0:
                    loss_mask[k, : max_len - label_len - 1] = 0
            model_inputs['loss_mask'] = loss_mask.masked_fill(~label_mask, 0)

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        loss_label = []
        if hasattr(model_inputs, 'loss_mask'):
            for loss, id in zip(model_inputs.loss_mask, model_inputs.input_ids):
                loss_label.append(self.tokenizer.decode((loss * id).view(-1).int()))

            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label, mask_label in zip(sources, labels, loss_label):
                    f.write(text + '\n')
                    f.write(label + '\n')
                    f.write(mask_label + '\n\n')
        else:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                # 注意：labels['input_ids'] 或者是 labels 本身，取决于调用处
                labels_to_write = labels['input_ids'] if isinstance(labels, dict) else labels
                for text, label in zip(sources, labels_to_write):
                    # 处理 label 为 tensor 或 list 的情况
                    if hasattr(label, 'tolist'):
                        label = label.tolist()
                    # 过滤 -100
                    valid_label = [t for t in label if t != -100]
                    f.write(text + '\n')
                    f.write(self.tokenizer.decode(valid_label, clean_up_tokenization_spaces=False) + '\n')