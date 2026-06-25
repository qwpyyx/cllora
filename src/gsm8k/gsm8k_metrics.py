import re
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Dict

# Matches GSM8K-style final answers after #### or any standalone number.
_HASH_ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

# Explicit answer cues used by many decoder-only LLMs.
# We keep this conservative and still prefer the official #### delimiter first.
_ANSWER_CUE_RE = re.compile(
    r"(?i)(?:final\s+answer|answer|ans|答案)\s*(?:is|=|:|：)?\s*([-+]?\d[\d,]*(?:\.\d+)?)"
)

# If a model emits `answer\nExplanation: ...`, the original last-number rule
# will often pick a number from the explanation.  These markers identify
# where the answer-first region ends.
# Extended for Qwen3 which generates patterns like "Okay, let's see...",
# "Let me think...", "Now, first...".
_EXPLANATION_MARKER_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:explanation|reasoning|rationale|solution|解析|解释"
    r"|okay\b|ok\b|let(?:'s|\s+me)\b|first[,，]|now[,，]"
    r")\s*[:：]?"
)


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


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _maybe_extract_answer_first_region(text: str) -> str:
    """Extract from an answer-first prefix before Explanation/Reasoning.

    Llama-style generations sometimes look like:
        "540\n\nExplanation: ... 9 ..."
    The correct final answer is the answer-first prefix, not the last number in
    the explanation.  This helper is intentionally conservative: it only fires
    when an explicit explanation/reasoning marker is present and the prefix
    contains a numeric answer.
    """
    marker = _EXPLANATION_MARKER_RE.search(text)
    if not marker:
        return ""

    prefix = text[:marker.start()].strip()
    if not prefix:
        return ""

    # If the first non-empty line begins with a number, treat that as the final
    # answer.  This handles "540\nExplanation: ..." and "540.\nReasoning: ...".
    first_line = _first_nonempty_line(prefix)
    m = re.match(r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)\b", first_line)
    if m:
        return _canonicalize_number(m.group(1))

    # If the prefix contains an explicit answer cue, use the last such cue in
    # the prefix.  This handles "Final answer: 540\nExplanation: ...".
    cue_matches = list(_ANSWER_CUE_RE.finditer(prefix))
    if cue_matches:
        return _canonicalize_number(cue_matches[-1].group(1))

    # Conservative fallback within the prefix: if there is exactly one numeric
    # span before the explanation, it is very likely the answer.
    nums = _NUM_RE.findall(prefix)
    if len(nums) == 1:
        return _canonicalize_number(nums[0])

    return ""


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric answer from GSM8K-style outputs.

    Supports:
    - raw dataset labels containing '#### 42'
    - direct final-answer outputs such as '42'
    - answer-first generations such as '42\nExplanation: ...'
    - reasoning outputs that end with the answer

    Priority order:
    1) official GSM8K delimiter `####`
    2) answer-first prefix before Explanation/Reasoning markers
    3) explicit final-answer cues
    4) original last-number fallback
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

    # 3) Prefer answer-first regions before Explanation/Reasoning.
    prefix_ans = _maybe_extract_answer_first_region(s)
    if prefix_ans:
        return prefix_ans

    # 4) Prefer explicit final-answer cues if present.
    cue_matches = list(_ANSWER_CUE_RE.finditer(s))
    if cue_matches:
        return _canonicalize_number(cue_matches[-1].group(1))

    # 5) Fall back to the original behavior: last numeric span in the string.
    nums = _NUM_RE.findall(s)
    if nums:
        return _canonicalize_number(nums[-1])

    # 6) If no number exists, return the normalized raw string.
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
    if not isinstance(groups, list):
        groups = list(groups)
    assert len(predictions) == len(references) == len(groups), (
        f"# predictions={len(predictions)}, # references={len(references)}, # groups={len(groups)}"
    )

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
