#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a *manually downloaded* GSM8K dataset into the SuperNI-style JSON format
expected by uie_dataset_lora.py.

Compared with v2, this script does NOT download from Hugging Face Hub.
It reads GSM8K from a local directory, e.g. a manually downloaded dataset repo
that contains folders like:

    <gsm8k_local_dir>/main/
    <gsm8k_local_dir>/socratic/

and inside each subset folder, files such as:
    train-00000-of-00001.parquet
    test-00000-of-00001.parquet
or similarly named json/jsonl/csv/parquet files.

Design choices for this codebase:
1. Keep the official GSM8K train/test split from the local files.
2. Create a loader-compatible dev split by copying the official test split.
   This dev split is only a placeholder for the current dataset loader and
   should not be used for model selection if you do not run validation.
3. Support two label modes:
   - final: final numeric answer only (recommended for first experiments)
   - cot:   original rationale + final answer text

Output structure:
  <output_root>/SuperNI/gsm8k/train.json
  <output_root>/SuperNI/gsm8k/dev.json
  <output_root>/SuperNI/gsm8k/test.json

Task config structure:
  <task_config_dir>/train_tasks.json
  <task_config_dir>/dev_tasks.json
  <task_config_dir>/test_tasks.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_SUPPORTED_EXTS = (".parquet", ".json", ".jsonl", ".csv")


def extract_final_answer(ans: str) -> str:
    """Extract the final numeric answer from a GSM8K answer string."""
    ans = str(ans).strip()
    if "####" in ans:
        ans = ans.split("####")[-1].strip()
    nums = _NUM_RE.findall(ans)
    if not nums:
        return ans
    return nums[-1].replace(",", "").strip()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_task_configs(task_config_dir: str) -> None:
    ensure_dir(task_config_dir)
    config = {"SuperNI": [{"dataset name": "gsm8k", "sampling strategy": "random"}]}
    write_json(config, os.path.join(task_config_dir, "train_tasks.json"))
    write_json(config, os.path.join(task_config_dir, "dev_tasks.json"))
    write_json(config, os.path.join(task_config_dir, "test_tasks.json"))


def build_superni_doc(instances: List[Dict], split_name: str, answer_mode: str) -> Dict:
    definition = (
        "Solve the following grade-school math word problem. "
        "Return only the final answer as a number."
        if answer_mode == "final"
        else "Solve the following grade-school math word problem and provide the reasoning followed by the final answer."
    )

    out_instances = []
    for i, ex in enumerate(instances):
        question = str(ex["question"]).strip()
        answer = str(ex["answer"]).strip()
        label = extract_final_answer(answer) if answer_mode == "final" else answer
        out_instances.append(
            {
                "id": f"gsm8k-{split_name}-{i}",
                "input": question,
                "output": [label],
            }
        )

    return {
        "Dataset": "gsm8k",
        "Split": split_name,
        "Definition": [definition],
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": out_instances,
    }


def _normalize_split_name(name: str) -> Optional[str]:
    n = name.lower()
    if "train" in n:
        return "train"
    if "test" in n:
        return "test"
    if "validation" in n or "valid" in n or "dev" in n or "eval" in n:
        # HF GSM8K repo历史上有时会把官方held-out split写成eval/validation，
        # 这里统一映射，后面优先用test，没有test时再用它兜底。
        return "validation"
    return None


def _collect_data_files(base_dir: str) -> Dict[str, List[str]]:
    """
    Recursively collect local dataset files and group them by inferred split.
    Supported file types: parquet/json/jsonl/csv.
    """
    split2files: Dict[str, List[str]] = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.lower().endswith(_SUPPORTED_EXTS):
                continue
            split = _normalize_split_name(fn)
            if split is None:
                continue
            split2files[split].append(os.path.join(root, fn))

    for k in split2files:
        split2files[k] = sorted(split2files[k])
    return dict(split2files)


def _ext_to_loader(files: List[str]) -> str:
    exts = {os.path.splitext(f)[1].lower() for f in files}
    if len(exts) != 1:
        raise ValueError(f"Mixed file extensions in one split are not supported: {exts}")
    ext = next(iter(exts))
    if ext == ".parquet":
        return "parquet"
    if ext in (".json", ".jsonl"):
        return "json"
    if ext == ".csv":
        return "csv"
    raise ValueError(f"Unsupported file extension: {ext}")


def _load_local_split(files: List[str]) -> List[Dict]:
    loader = _ext_to_loader(files)
    ds_dict = load_dataset(loader, data_files={"data": files})
    ds: Dataset = ds_dict["data"]

    required = {"question", "answer"}
    cols = set(ds.column_names)
    if not required.issubset(cols):
        raise ValueError(
            f"Loaded local GSM8K split is missing required columns {required}. "
            f"Found columns: {sorted(cols)}"
        )

    return [dict(x) for x in ds]


def resolve_subset_dir(gsm8k_local_dir: str, subset: str) -> str:
    # 优先使用 <root>/<subset>
    cand = os.path.join(gsm8k_local_dir, subset)
    if os.path.isdir(cand):
        return cand
    # 允许用户直接把 subset 目录本身当成 root 传入
    if os.path.isdir(gsm8k_local_dir):
        return gsm8k_local_dir
    raise ValueError(f"Cannot find subset directory. Tried: {cand} and {gsm8k_local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k_local_dir", type=str, required=True,
                        help="Local path to manually downloaded GSM8K dataset root or subset directory.")
    parser.add_argument("--subset", type=str, default="main", choices=["main", "socratic"],
                        help="Subset to use. Usually 'main'.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory that will contain SuperNI/gsm8k/*.json")
    parser.add_argument("--task_config_dir", type=str, required=True,
                        help="Directory to write train_tasks.json/dev_tasks.json/test_tasks.json")
    parser.add_argument("--answer_mode", type=str, default="final", choices=["final", "cot"],
                        help="Use final numeric answer only, or keep original chain-of-thought text")
    args = parser.parse_args()

    subset_dir = resolve_subset_dir(args.gsm8k_local_dir, args.subset)
    split2files = _collect_data_files(subset_dir)

    if "train" not in split2files:
        raise ValueError(
            f"No train split files found under: {subset_dir}. "
            f"Expected filenames containing 'train' with extensions in {_SUPPORTED_EXTS}."
        )

    # 优先 official test；如果本地命名成 validation/eval，则兜底使用它作为 held-out split
    heldout_key = "test" if "test" in split2files else "validation" if "validation" in split2files else None
    if heldout_key is None:
        raise ValueError(
            f"No test/validation split files found under: {subset_dir}. "
            f"Expected filenames containing 'test' or 'validation'/'eval'."
        )

    train_list = _load_local_split(split2files["train"])
    test_list = _load_local_split(split2files[heldout_key])

    target_dir = os.path.join(args.output_root, "SuperNI", "gsm8k")
    ensure_dir(target_dir)

    train_doc = build_superni_doc(train_list, "train", args.answer_mode)
    test_doc = build_superni_doc(test_list, "test", args.answer_mode)
    # loader-compatible placeholder dev split; not intended for tuning if you do not run validation
    dev_doc = build_superni_doc(test_list, "dev", args.answer_mode)

    write_json(train_doc, os.path.join(target_dir, "train.json"))
    write_json(dev_doc, os.path.join(target_dir, "dev.json"))
    write_json(test_doc, os.path.join(target_dir, "test.json"))

    write_task_configs(args.task_config_dir)

    print("Done.")
    print(f"Subset dir:      {subset_dir}")
    print(f"Train instances: {len(train_list)}")
    print(f"Test instances:  {len(test_list)}")
    print(f"Data dir:        {target_dir}")
    print(f"Task config dir: {args.task_config_dir}")
    print(f"Held-out source: {heldout_key}")
    print("Note: dev.json is a loader-compatible placeholder copied from the official held-out split.")


if __name__ == "__main__":
    main()
