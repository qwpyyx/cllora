# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""CL_Benchmark Dataset."""

import json
import os
import random
import datasets
from hashlib import md5
#
logger = datasets.logging.get_logger(__name__)
TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
ANSWER_PREFIX = "Answer:"
SINGLE_QUOTES_SUBSTITUTE = "#$%#"
AUX_PROB = 0.3

#将一系列与数据集和任务配置相关的参数组合成一个字符串，然后使用 MD5 哈希算法对其进行哈希计算，从而生成一个唯一的缓存路径
# def gen_cache_path(cache_dir, data_args):
#     hash_str = data_args.data_dir + data_args.task_config_dir + \
#                data_args.instruction_file + data_args.instruction_strategy + \
#                str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
#     hash_obj = md5(hash_str.encode("utf-8"))
#     hash_id = hash_obj.hexdigest()
#     cache_path = os.path.join(cache_dir, str(hash_id))
#
#     return cache_path
from hashlib import md5
import os

def gen_cache_path(cache_dir, data_args):
    """
    为不同数据配置生成稳定的缓存子目录。
    - 始终纳入：data_dir、task_config_dir、max_num_instances_per_task、max_num_instances_per_eval_task
    - 有就纳入：instruction_file、instruction_strategy、task_order、add_task_name、add_dataset_name、common_dataset_name、num_examples
      （这些可选字段用于区分不同实验配置，若为 None/空字符串则忽略）
    """
    def _opt(v):
        return None if v is None else str(v)

    parts = [
        str(getattr(data_args, "data_dir", "")),
        str(getattr(data_args, "task_config_dir", "")),
        str(getattr(data_args, "max_num_instances_per_task", "")),
        str(getattr(data_args, "max_num_instances_per_eval_task", "")),
    ]

    # 仅当存在且非空才加入（避免 superni = None 时报错）
    for k in [
        "instruction_file",
        "instruction_strategy",
        "task_order",
        "add_task_name",
        "add_dataset_name",
        "common_dataset_name",
        "num_examples",
    ]:
        v = _opt(getattr(data_args, k, None))
        if v not in (None, ""):
            parts.append(v)

    hash_str = "|".join(parts)
    hash_id = md5(hash_str.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, hash_id)


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class UIEConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            instruction_file=None,
            instruction_strategy=None,
            task_config_dir=None,
            num_examples=None,
            max_num_instances_per_task=None,
            max_num_instances_per_eval_task=None,
            over_sampling=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.over_sampling = over_sampling
        self.instructions = self._parse_instruction(instruction_file)
        self.task_configs = self._parse_task_config(task_config_dir)
        self.instruction_strategy = instruction_strategy
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task


    def _parse_instruction(self, instruction_file):
        """
        Instruction example:
        {
          "RE": [
            {"instruction_type": "zero-shot", "instruction": "Given a phrase that describes the relationship between
            two words, extract the words and the lexical relationship between them.
            The output format should be :[(word1, relation, word2)]. \n"},
          ],
          "NER": [
            {"instruction_type": "zero-shot", "instruction": "Please list all entity words in the text that
            fit the category.Output format is [(word1, type1), (word2, type2))]. \n"},
          ],
          "EE": [
            {"instruction_type": "zero-shot", "instruction": "Extract the event information in the text
            and return them in the event list. \n"}
          ]
        }
        """
        if not instruction_file:
            return None
        instructions = {"zero-shot": {}, "few-shot": {}}

        with open(instruction_file, 'r+') as f:
            origin_instructions = json.load(f)

        for task in origin_instructions:
            for task_instruction in origin_instructions[task]:
                instruct_type = task_instruction["instruction_type"]
                if instruct_type == "zero-shot":
                    instructions['zero-shot'][task] = instructions['zero-shot'].get(task, [])
                    instructions['zero-shot'][task].append(task_instruction["instruction"])
                elif instruct_type == "few-shot":
                    instructions['few-shot'][task] = instructions['few-shot'].get(task, [])
                    instructions['few-shot'][task].append(task_instruction["instruction"])
                else:
                    raise ValueError("Invalid instruction type {}, please check your instruction file {}"
                                     .format(instruct_type, instruction_file))
        return instructions


    def _parse_task_config(self, task_config_dir):
        """
        Task config file example:
            {
              "SC": [
                {"sampling strategy": "random", "dataset name": "amazon_review_full"}
              ],
              "TC": [
                {"sampling strategy": "full", "dataset name": "ag_news"}
              ]
            }
        """
        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs


# TODO, few-shot, 需要 load 的时候就将值存好，放在 "Examples" 里面
class UIEInstructions(datasets.GeneratorBasedBuilder):
    """InstructUIE Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = UIEConfig
    BUILDER_CONFIGS = [
        UIEConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "Task": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "Samples": [{
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "instruction": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }
                }
            ),
            supervised_keys=None
        )

    def _load_dataset_only(self, dataset_path):
        with open(dataset_path, encoding="utf-8") as f:
            return json.load(f)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_configs is None:
            logger.error("Please provide right input: data_dir or task_config_dir!")

        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir
        task_configs = self.config.task_configs

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['train'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['dev'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": None,  # default load total test samples to test
                    "subset": "test"
                }),
        ]


    def _load_dataset(self, dataset_path, labels_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
        with open(labels_path, encoding="utf-8") as labels_f:
            labels = json.load(labels_f)

        return instances, labels


    def _get_instruction(self, task):
        assert self.config.instruction_strategy in INSTRUCTION_STRATEGIES
        if self.config.num_examples is not None and self.config.num_examples > 0:
            task_instructions = self.config.instructions['few-shot'][task]
        else:
            task_instructions = self.config.instructions['zero-shot'][task]
        if self.config.instruction_strategy == "single":
            return task_instructions[0]
        else:
            return random.choice(task_instructions)


    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances!=None and self.config.over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))

        return instances


    def load_SC_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        # sentiment classification
        # you should first modify the original dataset to the standard format (json):
        # {"label": xxx, "sentence": "Title" + xxx + "\nText: " + xxx + "\n"}
        
        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "SC", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('SC')
            instruction += "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:" # value of "sentence" will be filled in {0}
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example


    def load_TC_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        # text classification

        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "TC", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('TC')
            instruction += "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example


    def load_NLI_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "NLI", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('NLI')
            instruction += "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example


    def load_QQP_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "QQP", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('QQP')
            instruction += "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example
    

    def load_BoolQA_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "BoolQA", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('BoolQA')
            instruction += "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example
            

    def load_COPA_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "COPA", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = "{0}" + "\nAnswer:"
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example


    def load_MultiRC_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "MultiRC", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('MultiRC')
            instruction += "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example


    def load_WiC_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "WiC", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('WiC')
            instruction += "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"
            label = instance['label']

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example

    def load_LongSeq_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances,
                             subset):
        data = self._load_dataset_only(dataset_path)
        input_mode = 'zeroshot'

        definition = ""
        if len(data.get("Definition", [])) > 0:
            if input_mode in ('fewshot', 'zeroshot'):
                if isinstance(data["Definition"], list):
                    definition = data["Definition"][0].strip()
                else:
                    definition = data["Definition"].strip()
                definition += "\n"

        sample_template = {"Task": "CL", "Dataset": dataset_name, "Samples": [], "subset": subset}

        for idx, instance in enumerate(data['Instances']):
            example = sample_template.copy()
            instruction = ""
            instruction += "{0}\n"
            instruction += "Output: "

            pos_examples = []
            if input_mode == 'fewshot':
                for j, pos_example in enumerate(data.get("Positive Examples", [])[:1]):
                    pos_example_str = f"Positive Example {j + 1} -\n"
                    pos_example_str += f"Input: {pos_example['input'].strip()}\n"
                    pos_example_str += f"Output: {pos_example['output'].strip()}\n"
                    pos_examples.append(pos_example_str)

            instruction = definition + "".join(pos_examples) + instruction

            if isinstance(instance["output"], list):
                label = instance["output"][random.randint(0, len(instance["output"]) - 1)]
            else:
                label = instance["output"]

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['input'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }
            yield example

    def load_SuperNI_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances,
                             subset):
        data = self._load_dataset_only(dataset_path)
        input_mode = 'zeroshot'

        definition = ""
        if input_mode in ('fewshot', 'zeroshot'):
            if isinstance(data["Definition"], list):
                definition = "Definition: " + data["Definition"][0].strip()
            else:
                definition = "Definition: " + data["Definition"].strip()
            definition += "\n\n"

        sample_template = {"Task": "CL", "Dataset": dataset_name, "Samples": [], "subset": subset}

        for idx, instance in enumerate(data['Instances']):
            example = sample_template.copy()

            instruction = ""
            if input_mode in ('fewshot', 'zeroshot'):
                instruction += "Now complete the following example -\n"
            instruction += "Input: {0}\n"
            instruction += "Output: "

            pos_examples = []
            if input_mode == 'fewshot':
                for j, pos_example in enumerate(data.get("Positive Examples", [])[:1]):
                    pos_example_str = f"Positive Example {j + 1} -\n"
                    pos_example_str += f"Input: {pos_example['input'].strip()}\n"
                    pos_example_str += f"Output: {pos_example['output'].strip()}\n"
                    pos_examples.append(pos_example_str)

            instruction = definition + "".join(pos_examples) + instruction

            if isinstance(instance["output"], list):
                label = instance["output"][random.randint(0, len(instance["output"]) - 1)]
            else:
                label = instance["output"]

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['input'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }
            yield example

    # def _generate_examples(self, path=None, task_config=None, max_num_instances_per_task=None, subset=None):
    #     """Yields examples."""
    #     logger.info(f"Generating tasks from = {path}")
    #
    #     for task in task_config:
    #         if task == "SC":
    #             load_func = self.load_SC_dataset
    #         elif task == 'TC':
    #             load_func = self.load_TC_dataset
    #         elif task == 'NLI':
    #             load_func = self.load_NLI_dataset
    #         elif task == 'QQP':
    #             load_func = self.load_QQP_dataset
    #         elif task == 'BoolQA':
    #             load_func = self.load_BoolQA_dataset
    #         elif task == 'COPA':
    #             load_func = self.load_COPA_dataset
    #         elif task == 'MultiRC':
    #             load_func = self.load_MultiRC_dataset
    #         elif task == 'WiC':
    #             load_func = self.load_WiC_dataset
    #         else:
    #             raise ValueError("Unsupport {} task, plz check {} task config!".format(task, subset))
    #
    #         # load dataset
    #         for dataset in task_config[task]:
    #             ds_name = dataset["dataset name"]
    #             sampling_strategy = dataset.get("sampling strategy", "random")
    #             ds_path = os.path.join(path, task, ds_name, subset + '.json')
    #             labels_path = os.path.join(path, task, ds_name, 'labels.json')
    #             assert os.path.exists(ds_path)
    #             assert os.path.exists(labels_path)
    #
    #             idx = -1
    #             instances = []
    #             for sample in load_func(ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
    #                                     subset):
    #                 idx += 1
    #                 instances.append(sample)
    #                 yield f"{task}##{ds_path}##{idx}", sample
    def _generate_examples(self, path=None, task_config=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")

        # 允许的老任务（你的原有目录结构：<Task>/<Dataset>/<split>.json + labels.json）
        LEGACY_TASKS = {"SC", "TC", "NLI", "QQP", "BoolQA", "COPA", "MultiRC", "WiC"}
        print("开始生成样本。。。。。")
        print(f"task config is:{task_config}")
        for task in task_config:
            # 1) 传统任务：仍然按你原来的加载函数和路径组织来做
            if task in LEGACY_TASKS:
                if task == "SC":
                    load_func = self.load_SC_dataset
                elif task == "TC":
                    load_func = self.load_TC_dataset
                elif task == "NLI":
                    load_func = self.load_NLI_dataset
                elif task == "QQP":
                    load_func = self.load_QQP_dataset
                elif task == "BoolQA":
                    load_func = self.load_BoolQA_dataset
                elif task == "COPA":
                    load_func = self.load_COPA_dataset
                elif task == "MultiRC":
                    load_func = self.load_MultiRC_dataset
                elif task == "WiC":
                    load_func = self.load_WiC_dataset
                else:
                    raise ValueError(f"Unsupport {task} task, please check {subset} task config!")

                for dataset in task_config[task]:
                    ds_name = dataset["dataset name"]
                    sampling_strategy = dataset.get("sampling strategy", "random")
                    ds_path = os.path.join(path, task, ds_name, subset + '.json')
                    labels_path = os.path.join(path, task, ds_name, 'labels.json')
                    assert os.path.exists(ds_path), f"Missing dataset file: {ds_path}"
                    assert os.path.exists(labels_path), f"Missing labels file: {labels_path}"

                    idx = -1
                    for sample in load_func(
                        ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task, subset
                    ):
                        idx += 1
                        yield f"{task}##{ds_path}##{idx}", sample

            # 2) 对方整合目录：Long_Sequence（无 labels.json，定义见对方 loader）
            elif task == "Long_Sequence":
                # 需要你在类中提供 self.load_LongSeq_dataset（参考对方实现）
                load_func = self.load_LongSeq_dataset
                for dataset in task_config[task]:
                    ds_name = dataset["dataset name"]
                    sampling_strategy = dataset.get("sampling strategy", "random")
                    ds_path = os.path.join(path, "Long_Sequence", ds_name, subset + ".json")
                    assert os.path.exists(ds_path), f"Missing dataset file: {ds_path}"

                    idx = -1
                    for sample in load_func(
                        ds_path, None, ds_name, sampling_strategy, max_num_instances_per_task, subset
                    ):
                        idx += 1
                        yield f"{task}##{ds_path}##{idx}", sample

            # 3) SuperNI（无 labels.json，定义见对方 loader）
            elif task == "SuperNI":
                # 需要你在类中提供 self.load_SuperNI_dataset（参考对方实现）
                load_func = self.load_SuperNI_dataset
                for dataset in task_config[task]:
                    ds_name = dataset["dataset name"]
                    sampling_strategy = dataset.get("sampling strategy", "random")
                    ds_path = os.path.join(path, "SuperNI", ds_name, subset + ".json")
                    assert os.path.exists(ds_path), f"Missing dataset file: {ds_path}"

                    idx = -1
                    for sample in load_func(
                        ds_path, None, ds_name, sampling_strategy, max_num_instances_per_task, subset
                    ):
                        idx += 1
                        yield f"{task}##{ds_path}##{idx}", sample

            else:
                raise ValueError(f"Unsupport task '{task}', please check your task config for split '{subset}'.")
