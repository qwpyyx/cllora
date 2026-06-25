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
    decoder_start_token_id: Optional[int] = None

    def __post_init__(self):
        if self.model is not None:
            if self.model_name_or_path is None:
                if hasattr(self.model, "config"):
                    self.model_name_or_path = getattr(self.model.config, "_name_or_path", "")
                else:
                    self.model_name_or_path = str(self.model.__class__.__name__)

            if self.decoder_start_token_id is None and hasattr(self.model, "config"):
                self.decoder_start_token_id = getattr(self.model.config, "decoder_start_token_id", None)
                if self.decoder_start_token_id is None:
                    self.decoder_start_token_id = getattr(self.model.config, "pad_token_id", None)
            self.model = None

    def __call__(self, batch: List[Dict], return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_name = self.model_name_or_path or getattr(self.tokenizer, "name_or_path", "") or ""

        if _check_model_name(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif _check_model_name(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            model_inputs = self.seq2seq_call(batch, return_tensors)
        return model_inputs

    def get_instruction(self, instance):
        instruction = instance['Instance']["instruction"]
        content = instance['Instance']['sentence']

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
            pass
        return instruction

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def seq2seq_call(self, batch, return_tensors):
        # T5 逻辑保持不变
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

            if self.decoder_start_token_id is not None:
                decoder_input_ids = self.shift_tokens_right(
                    model_inputs["labels"],
                    self.tokenizer.pad_token_id,
                    self.decoder_start_token_id
                )
                model_inputs["decoder_input_ids"] = decoder_input_ids
            self._save_samples(model_inputs, sources, labels)
        return model_inputs

    def decoder_call(self, batch, return_tensors):
        # 参照 cl_collator.py 和 run_llama.py 的标准设置
        self.tokenizer.padding_side = 'left'
        pad_token_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id

        input_ids_list = []
        labels_list = []
        input_ids_wo_label_list = []  # 纯 Prompt，用于生成
        final_sources = []

        max_src_len = self.max_source_length or 512
        max_tgt_len = self.max_target_length or 50
        max_seq_len = max_src_len + max_tgt_len

        for instance in batch:
            label = instance['Instance']['label']
            instruction = self.get_instruction(instance)
            instruction = instruction.replace("{0}", "").strip()
            # 1. 准备文本
            # 参考别人的做法，通常 prompt 包含 Output:
            prompt_text = instruction
            target_text = label + self.tokenizer.eos_token

            # 2. 处理 input_ids_wo_label (纯 Prompt)
            # add_special_tokens=True 会自动加 BOS
            tokenized_prompt = self.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
            if len(tokenized_prompt) > max_src_len:
                tokenized_prompt = tokenized_prompt[:max_src_len]

            # 3. 处理 input_ids (Prompt + Label)
            full_text = prompt_text + target_text
            tokenized_full = self.tokenizer(full_text, add_special_tokens=True)["input_ids"]
            if len(tokenized_full) > max_seq_len:
                tokenized_full = tokenized_full[:max_seq_len]

            # 4. 制作 Labels (核心：使用 -100 屏蔽 Prompt 部分)
            # 这样标准模型就会自动忽略这部分的 Loss，不需要 loss_mask
            prompt_len = len(tokenized_prompt)
            if prompt_len > len(tokenized_full):
                prompt_len = len(tokenized_full)

            # Prompt 部分设为 -100，Label 部分保留原 ID
            full_labels = [-100] * prompt_len + tokenized_full[prompt_len:]

            # 5. 填充列表
            is_train = instance.get('subset', '') == 'train'
            if is_train:
                input_ids_list.append(tokenized_full)
                labels_list.append(full_labels)
            else:
                input_ids_list.append(tokenized_prompt)
                labels_list.append([-100] * len(tokenized_prompt))

            # 这一列始终存放纯 Prompt ID，用于 Trainer 做生成时的输入
            input_ids_wo_label_list.append(tokenized_prompt)
            final_sources.append(full_text)

        # --- Batch Padding (Left Padding) ---
        max_batch_len = max(len(x) for x in input_ids_list)
        max_batch_len_wo = max(len(x) for x in input_ids_wo_label_list)

        final_input_ids = []
        final_labels = []
        final_att_mask = []
        final_input_ids_wo_label = []
        final_att_mask_wo_label = []

        for i in range(len(input_ids_list)):
            inp = input_ids_list[i]
            lbl = labels_list[i]
            inp_wo = input_ids_wo_label_list[i]

            # Pad Main Input
            pad_len = max_batch_len - len(inp)
            padded_inp = [pad_token_id] * pad_len + inp
            padded_lbl = [-100] * pad_len + lbl  # Pad 部分 Label 也是 -100
            padded_mask = [0] * pad_len + [1] * len(inp)

            # Pad Input Without Label
            pad_len_wo = max_batch_len_wo - len(inp_wo)
            padded_inp_wo = [pad_token_id] * pad_len_wo + inp_wo
            padded_mask_wo = [0] * pad_len_wo + [1] * len(inp_wo)

            final_input_ids.append(padded_inp)
            final_labels.append(padded_lbl)
            final_att_mask.append(padded_mask)
            final_input_ids_wo_label.append(padded_inp_wo)
            final_att_mask_wo_label.append(padded_mask_wo)

        model_inputs = {
            'input_ids': torch.tensor(final_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(final_att_mask, dtype=torch.long),
            'labels': torch.tensor(final_labels, dtype=torch.long),
            'input_ids_wo_label': torch.tensor(final_input_ids_wo_label, dtype=torch.long),
            # Required for decoder-only generation with left padding.  In Llama-3.x
            # pad/eos can otherwise be ambiguous and generate() cannot infer masks.
            'attention_mask_wo_label': torch.tensor(final_att_mask_wo_label, dtype=torch.long)
        }

        # [删除] model_inputs['loss_mask'] = ... (不再需要)

        self._save_samples(model_inputs, final_sources, labels_list)
        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        # 安全的获取逻辑，支持 dict 和 BatchEncoding
        def get_val(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        loss_mask = get_val(model_inputs, 'loss_mask')
        input_ids = get_val(model_inputs, 'input_ids')

        # 尝试写入文件
        try:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                # 优先使用 loss_mask 逻辑
                if loss_mask is not None:
                    for text, label, loss, id_t in zip(sources, labels, loss_mask, input_ids):
                        # 解码被 Mask 选中的部分
                        masked_tokens = (loss * id_t).view(-1).int()
                        # 去掉 0 (Padding/Ignored) 再解码，防止解码出一堆乱码
                        valid_tokens = masked_tokens[masked_tokens != 0]
                        mask_label_str = self.tokenizer.decode(valid_tokens)

                        # 如果 label 是 ID 列表，这里也解码一下方便阅读
                        label_str = label
                        if isinstance(label, list) or isinstance(label, torch.Tensor):
                            # 过滤 -100
                            valid_lbl = [t for t in label if t != -100]
                            label_str = self.tokenizer.decode(valid_lbl, skip_special_tokens=False)

                        f.write(f"Input: {text}\n")
                        f.write(f"Label: {label_str}\n")
                        f.write(f"Masked: {mask_label_str}\n\n")
                else:
                    # 回退逻辑 (T5)
                    # 这里的 labels 可能是 BatchEncoding 或 tensor
                    labels_val = labels
                    if hasattr(labels, 'get'):  # 如果是 dict/BatchEncoding
                        labels_val = labels.get('input_ids', labels)

                    for text, label in zip(sources, labels_val):
                        if hasattr(label, 'tolist'):
                            label = label.tolist()
                        valid_label = [t for t in label if t != -100]
                        decoded_label = self.tokenizer.decode(valid_label, clean_up_tokenization_spaces=False)
                        f.write(f"Input: {text}\n")
                        f.write(f"Label: {decoded_label}\n\n")
        except Exception as e:
            # 只是记录日志，不要因为这就崩了训练
            logger.warning(f"Failed to save samples: {e}")
            pass