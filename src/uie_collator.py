import logging
from torch.nn.parallel import DistributedDataParallel
import torch
from transformers.data.data_collator import *
import inspect

logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama', 'gpt2-xl','gpt2-large']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False


@dataclass
class DataCollatorForUIE:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
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

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        original_model = self.model
        if hasattr(original_model, "module"):
            original_model = original_model.module

        model_name = original_model.config._name_or_path
        # print(model_name)
        if check_model(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            raise ValueError('Unsupport model {}!'.format(model_name))

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

        # TODO, support few shot
        # add few shot samples
        samples = ''
        if len(instance['Samples']) > 0:
            raise Exception('Few shot is coming soon...')
        if samples:
            content = samples + content
        # TODO, fix bug
        try:
            instruction = instruction.format(content)
        finally:
            return instruction


    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        # TODO, support online demo
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

            # prepare decoder_input_ids
            if self.model is not None:
                original_model = self.model
                if hasattr(original_model, "module"):
                    original_model = original_model.module
                decoder_input_ids = original_model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids

            self._save_samples(model_inputs, sources, labels)
        logger.info(
            f"seq2seq_call: input_ids shape {model_inputs['input_ids'].shape}, labels shape {model_inputs['labels'].shape}")
        return model_inputs

    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        sources = []
        label_lens = []
        labels = []
        max_len = -1
        if batch[0]['subset'] == "train":
            limit_input_len = self.max_source_length + self.max_target_length
        else:
            limit_input_len = self.max_source_length

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            # add bos and eos
            task_input = self.tokenizer.bos_token + instruction
            label = label + self.tokenizer.eos_token

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
                    max_len = self.max_source_length
                    input_w_label = self.tokenizer.decode(
                        (tokenized_input + tokenized_label)[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_w_label)
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))

        # TODO, support online demo
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

            label_pad = self.label_pad_token_id
            label_mask_attn = model_inputs["attention_mask"].bool()
            # 先按原来方式生成“全序列标签”，只把 padding 位置置 -100
            model_inputs["labels"] = model_inputs['input_ids'].masked_fill(~label_mask_attn, label_pad)

            # ==== 关键：根据模型是否支持 loss_mask 决定处理方式 ====
            original_model = self.model.module if hasattr(self.model, "module") else self.model
            accepts_loss_mask = False
            try:
                params = inspect.signature(original_model.forward).parameters
                accepts_loss_mask = "loss_mask" in params
            except Exception:
                accepts_loss_mask = False

            max_len = min(max_len, limit_input_len)

            if accepts_loss_mask:
                # 保持你原有的 loss_mask 路径（如 LlamaForCausalLM_with_lossmask）
                loss_mask = torch.ones_like(model_inputs["attention_mask"], dtype=torch.float)
                for k, label_len in enumerate(label_lens):
                    loss_mask[k, : max_len - label_len - 1] = 0
                model_inputs["loss_mask"] = loss_mask.masked_fill(~label_mask_attn, 0)
            else:
                # GPT-2 等不支持 loss_mask 的模型：把非答案区 label 直接置为 -100
                labels = model_inputs["labels"].clone()
                for k, label_len in enumerate(label_lens):
                    # 你原逻辑中有效“答案区”是末尾的 label_len 部分
                    # 因此把前面的 prompt/input 区域设为 -100
                    cut = max_len - label_len - 1
                    cut = max(cut, 0)
                    labels[k, :cut] = label_pad
                model_inputs["labels"] = labels
                # 确保不再传 loss_mask
                if "loss_mask" in model_inputs:
                    del model_inputs["loss_mask"]

            self._save_samples(model_inputs, sources, labels)
            logger.info(
                f"decoder_call: input_ids shape {model_inputs['input_ids'].shape}, "
                f"labels shape {model_inputs['labels'].shape}, "
                f"{'with' if accepts_loss_mask else 'no'} loss_mask"
            )
            return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        acc = getattr(self, "accelerator", None)
        if acc is not None and not acc.is_main_process:
            return

        loss_label = []
        if 'loss_mask' in model_inputs:  # 用字典键判断更安全，避免hasattr的潜在问题
            for loss, input_id in zip(model_inputs['loss_mask'], model_inputs['input_ids']):
                loss_label.append(self.tokenizer.decode((loss * input_id).view(-1).int(), skip_special_tokens=True))

            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label, mask_label in zip(sources, labels, loss_label):
                    f.write(f"Source: {text}\n")
                    f.write(f"Label: {label}\n")
                    f.write(f"Loss Mask: {mask_label}\n\n")
        else:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label_ids in zip(sources, labels['input_ids']):
                    label_text = self.tokenizer.decode(label_ids, clean_up_tokenization_spaces=False)
                    f.write(f"Source: {text}\n")
                    f.write(f"Label: {label_text}\n\n")

