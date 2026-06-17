import re
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Dict

# Matches GSM8K-style final answers after #### or any standalone number.
_HASH_ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _canonicalize_number(text: str) -> str:
    text = str(text).strip()
    text = text.replace(",", "")
    if not text:
        return ""
    try:
        dec = Decimal(text)
        # Avoid scientific notation and trim trailing zeros.
        s = format(dec, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s
    except (InvalidOperation, ValueError):
        return text.strip()



def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric answer from GSM8K-style outputs.

    Supports either:
    - raw dataset labels containing '#### 42'
    - model generations that may include reasoning and end with a number
    - plain final-answer-only labels such as '42'
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # 1) Prefer the official GSM8K delimiter if present.
    m = _HASH_ANS_RE.search(s)
    if m:
        return _canonicalize_number(m.group(1))

    # 2) Remove common currency/unit noise but keep signs/decimals.
    s = s.replace("$", " ")

    # 3) Fall back to the last numeric span in the string.
    nums = _NUM_RE.findall(s)
    if nums:
        return _canonicalize_number(nums[-1])

    # 4) If no number exists, return the normalized raw string.
    return s.strip().lower()



def compute_gsm8k_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    assert len(predictions) == len(references), (
        f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    )
    pred_ans = [extract_gsm8k_final_answer(p) for p in predictions]
    gold_ans = [extract_gsm8k_final_answer(g) for g in references]
    exact = sum(int(p == g) for p, g in zip(pred_ans, gold_ans))
    metrics = {
        "gsm8k_em": round(100.0 * exact / len(references), 4),
        "exact_match": round(100.0 * exact / len(references), 4),
    }
    return metrics



def compute_grouped_gsm8k_metrics(predictions: List[str], references: List[str], groups: Iterable[str]) -> Dict[str, float]:
    assert len(predictions) == len(references) == len(list(groups)) if not isinstance(groups, list) else len(predictions) == len(references) == len(groups)
    if not isinstance(groups, list):
        groups = list(groups)
    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        examples_by_group.setdefault(group, []).append((pred, gold))

    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_gsm8k_metrics(list(task_predictions), list(task_references))
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results
