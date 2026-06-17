#!/usr/bin/env python
# coding=utf-8
"""
Download and convert Databricks Dolly-15k into the UIE/SuperNI-style format
expected by uie_dataset_lora.py.

Output layout:
  <output_root>/
    data/
      SuperNI/
        dolly_<category>/
          train.json
          dev.json
          test.json
    task_config/
      train_tasks.json
      dev_tasks.json
      test_tasks.json

Then use:
  --data_dir <output_root>/data
  --task_config_dir <output_root>/task_config

Notes:
- Dolly has one original HF split. This script creates train/dev/test splits
  stratified by Dolly category.
- Each Dolly category is written as one "Dataset" under task "SuperNI".
  Therefore, in federated training, use:
    --partition_strategy label
    --partition_label_key Dataset
  so Dirichlet(alpha) creates category-skew semantic non-IID clients.
"""

import argparse
import json
import os
import random
import re
import hashlib
from collections import defaultdict
from pathlib import Path


def _safe_name(x: str) -> str:
    x = (x or "unknown").strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown"


def _load_dolly(hf_name: str, cache_dir: str = None, local_jsonl: str = None):
    if local_jsonl:
        rows = []
        with open(local_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The 'datasets' package is required to download Dolly. "
            "Install it with: pip install datasets"
        ) from e

    ds = load_dataset(hf_name, split="train", cache_dir=cache_dir)
    return [dict(x) for x in ds]


def _format_input(row):
    instruction = str(row.get("instruction", "") or "").strip()
    context = str(row.get("context", "") or "").strip()

    if context:
        return f"Instruction:\n{instruction}\n\nContext:\n{context}"
    return f"Instruction:\n{instruction}"


def _format_output(row):
    return str(row.get("response", "") or "").strip()


def _split_rows(rows, train_ratio, dev_ratio, seed):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)

    n = len(rows)
    if n == 0:
        return [], [], []

    n_train = int(round(n * train_ratio))
    n_dev = int(round(n * dev_ratio))

    # ensure every non-tiny category has at least one example in each split
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_dev = max(1, min(n_dev, n - n_train - 1))
    elif n == 2:
        n_train, n_dev = 1, 0
    else:
        n_train, n_dev = 1, 0

    train = rows[:n_train]
    dev = rows[n_train:n_train + n_dev]
    test = rows[n_train + n_dev:]
    return train, dev, test


def _write_superni_file(path: Path, rows, category: str, category_name: str):
    definition = (
        "Given an instruction and optional context, write a helpful, correct, "
        "and concise response."
    )
    data = {
        "Definition": [definition],
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": []
    }

    for idx, row in enumerate(rows):
        inp = _format_input(row)
        out = _format_output(row)
        if not inp or not out:
            continue
        data["Instances"].append({
            "id": f"{category}_{idx}",
            "input": inp,
            "output": [out]
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--hf_name", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_jsonl", type=str, default=None,
                        help="Optional local Dolly-like jsonl file. If provided, no HF download is attempted.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples_per_category", type=int, default=-1)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.dev_ratio < 0 or args.train_ratio + args.dev_ratio >= 1:
        raise ValueError("Require train_ratio > 0, dev_ratio >= 0, and train_ratio + dev_ratio < 1.")

    rows = _load_dolly(args.hf_name, cache_dir=args.cache_dir, local_jsonl=args.local_jsonl)

    by_cat = defaultdict(list)
    for row in rows:
        cat = str(row.get("category", "unknown") or "unknown").strip()
        by_cat[cat].append(row)

    rng = random.Random(args.seed)
    output_root = Path(args.output_root)
    data_root = output_root / "data"
    task_config_root = output_root / "task_config"

    split_config = {"train": {"SuperNI": []}, "dev": {"SuperNI": []}, "test": {"SuperNI": []}}
    summary = []

    for cat in sorted(by_cat.keys()):
        rows_cat = list(by_cat[cat])
        rng.shuffle(rows_cat)
        if args.max_samples_per_category and args.max_samples_per_category > 0:
            rows_cat = rows_cat[:args.max_samples_per_category]

        safe_cat = _safe_name(cat)
        ds_name = f"dolly_{safe_cat}"

        train_rows, dev_rows, test_rows = _split_rows(
            rows_cat,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            seed=args.seed + int(hashlib.md5(cat.encode("utf-8")).hexdigest()[:8], 16) % 100000,
        )

        ds_dir = data_root / "SuperNI" / ds_name
        _write_superni_file(ds_dir / "train.json", train_rows, safe_cat, cat)
        _write_superni_file(ds_dir / "dev.json", dev_rows, safe_cat, cat)
        _write_superni_file(ds_dir / "test.json", test_rows, safe_cat, cat)

        for split in ["train", "dev", "test"]:
            split_config[split]["SuperNI"].append({
                "sampling strategy": "full",
                "dataset name": ds_name
            })

        summary.append({
            "category": cat,
            "dataset_name": ds_name,
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
            "total": len(rows_cat),
        })

    task_config_root.mkdir(parents=True, exist_ok=True)
    for split, cfg in split_config.items():
        file_name = "dev_tasks.json" if split == "dev" else f"{split}_tasks.json"
        with open(task_config_root / file_name, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    with open(output_root / "dolly_conversion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Dolly UIE data prepared.")
    print(f"  data_dir:        {data_root}")
    print(f"  task_config_dir: {task_config_root}")
    print(f"  categories:      {len(summary)}")
    print("Use these arguments:")
    print(f"  --data_dir {data_root}")
    print(f"  --task_config_dir {task_config_root}")
    print("For semantic non-IID FL:")
    print("  --partition_strategy label")
    print("  --partition_label_key Dataset")


if __name__ == "__main__":
    main()
