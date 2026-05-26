"""
shared/synth_utils.py — Prompt templates, label validation, JSONL writer
for synthetic data generation.
"""

import json
import os
from pathlib import Path

# ── Label vocabularies (fixed per TASK SPEC) ──────────────────────────
ST1_LABELS = {"fallacy", "non-fallacy"}
ST2_LABELS = {
    "authority", "black-white", "hasty_generalization", "natural",
    "population", "slippery_slope", "tradition", "worse_problems",
}
ST3_LABELS = {
    "epistemic-external", "epistemic-internal",
    "practical-external", "practical-internal",
}

FALLACY_DESCRIPTIONS = {
    "authority": "Appeal to Authority — citing an authority figure as evidence without proper justification.",
    "black-white": "Black-or-White / False Dilemma — presenting only two options when more exist.",
    "hasty_generalization": "Hasty Generalization — drawing a broad conclusion from too few examples.",
    "natural": "Appeal to Nature — arguing something is good/right because it is natural.",
    "population": "Appeal to Population (Ad Populum) — arguing something is true because many people believe it.",
    "slippery_slope": "Slippery Slope — claiming one event will inevitably lead to extreme consequences.",
    "tradition": "Appeal to Tradition — arguing something is correct because it has always been done that way.",
    "worse_problems": "Appeal to Worse Problems (Relative Privation) — dismissing an issue because worse issues exist.",
}

SCHEME_DESCRIPTIONS = {
    "practical-internal": "Practical goal with internal basis — assessing a course of action based on properties of the subject (causes, consequences, values).",
    "practical-external": "Practical goal with external basis — assessing a course of action based on source authority/opinion.",
    "epistemic-internal": "Epistemic goal with internal basis — establishing a judgment based on properties of the subject.",
    "epistemic-external": "Epistemic goal with external basis — establishing a judgment based on source authority/opinion.",
}


def get_balanced_prompt(fallacy_type: str, count: int = 1) -> list[dict]:
    """Generate prompt for balanced synthetic data generation."""
    desc = FALLACY_DESCRIPTIONS.get(fallacy_type, fallacy_type)
    return [
        {"role": "system", "content": (
            "You are a dataset generator for NLP research on fallacy detection. "
            "Generate realistic Reddit-style comments that contain specific logical fallacies. "
            "Each example should be a self-contained argument that a real person might post online. "
            "Output valid JSON only."
        )},
        {"role": "user", "content": (
            f"Generate {count} example(s) of the fallacy type: {desc}\n\n"
            "Each example should be a realistic Reddit comment (2-5 sentences) that commits "
            "this specific fallacy. The fallacy should be clear but natural — not cartoonishly obvious.\n\n"
            f"For each example, output a JSON object on its own line with these fields:\n"
            f'{{"text_base": "the comment text", "fallacy_exists": 1, '
            f'"fallacy_type": "{fallacy_type}", '
            f'"argument_base": {{"claim": "main claim", "supports": ["premise1", "premise2"]}}}}\n\n'
            "Output ONLY the JSON objects, one per line, no other text."
        )},
    ]


def get_nonfallacy_prompt(resembles: str, count: int = 1) -> list[dict]:
    """Generate non-fallacious examples that resemble a specific fallacy type."""
    desc = FALLACY_DESCRIPTIONS.get(resembles, resembles)
    return [
        {"role": "system", "content": (
            "You are a dataset generator for NLP research on fallacy detection. "
            "Generate realistic Reddit-style comments that use reasoning similar to "
            "a specific fallacy type BUT are actually valid arguments (not fallacious). "
            "Output valid JSON only."
        )},
        {"role": "user", "content": (
            f"Generate {count} example(s) of VALID (non-fallacious) arguments that "
            f"superficially resemble: {desc}\n\n"
            "Each should be a realistic Reddit comment (2-5 sentences) that uses similar "
            "reasoning patterns but does NOT commit the fallacy. The argument should be "
            "logically sound despite resembling the fallacy.\n\n"
            f"For each example, output a JSON object on its own line:\n"
            f'{{"text_base": "the comment text", "fallacy_exists": 0, '
            f'"resembles_fallacy": "{resembles}", '
            f'"argument_base": {{"claim": "main claim", "supports": ["premise1", "premise2"]}}}}\n\n'
            "Output ONLY the JSON objects, one per line, no other text."
        )},
    ]


def get_counterfactual_prompt(original_text: str, original_type: str) -> list[dict]:
    """Generate a counterfactual pair: minimally edit to remove the fallacy."""
    return [
        {"role": "system", "content": (
            "You are a dataset generator for NLP research. Your task is to minimally "
            "edit a fallacious argument to make it non-fallacious, preserving the topic "
            "and as much wording as possible. Output valid JSON only."
        )},
        {"role": "user", "content": (
            f"The following argument contains a {original_type} fallacy:\n\n"
            f'"{original_text}"\n\n'
            "Minimally edit this text to remove the fallacy while keeping the same topic. "
            "The result should be a valid, non-fallacious argument.\n\n"
            "Output a single JSON object:\n"
            '{"text_base": "edited text", "fallacy_exists": 0, '
            f'"resembles_fallacy": "{original_type}", '
            '"argument_base": {"claim": "main claim", "supports": ["premise1"]}}\n\n'
            "Output ONLY the JSON object, no other text."
        )},
    ]


def get_hard_negative_prompt(fallacy_type: str, count: int = 1) -> list[dict]:
    """Generate hard negatives: non-fallacious but superficially look like fallacy_type."""
    desc = FALLACY_DESCRIPTIONS.get(fallacy_type, fallacy_type)
    return [
        {"role": "system", "content": (
            "You are an expert in argumentation and NLP dataset generation. "
            "Generate hard negative examples for fallacy detection: arguments that "
            "look like they contain a specific fallacy but are actually valid. "
            "These should be challenging for classifiers. Output valid JSON only."
        )},
        {"role": "user", "content": (
            f"Generate {count} HARD NEGATIVE example(s) for: {desc}\n\n"
            "These must be non-fallacious arguments that a fallacy classifier might "
            "incorrectly flag as containing this fallacy type. They should use similar "
            "vocabulary, structure, and reasoning patterns but be logically valid.\n\n"
            f"Output JSON objects, one per line:\n"
            f'{{"text_base": "the text", "fallacy_exists": 0, '
            f'"resembles_fallacy": "{fallacy_type}", '
            f'"argument_base": {{"claim": "claim", "supports": ["p1"]}}}}\n\n'
            "Output ONLY JSON, one object per line."
        )},
    ]


STRATEGY_PROMPT_FNS = {
    "balanced": get_balanced_prompt,
    "hard_negatives": get_hard_negative_prompt,
}


def validate_example(example: dict, subtask: str) -> bool:
    """Validate a generated example against the label vocabulary."""
    if not example.get("text_base") or len(example["text_base"].strip()) < 10:
        return False

    if subtask in ("st1", "st2"):
        fe = example.get("fallacy_exists")
        if fe not in (0, 1):
            return False
        if fe == 1:
            ft = example.get("fallacy_type", "")
            if ft not in ST2_LABELS:
                return False
        return True
    elif subtask == "st3":
        clf = example.get("classification", {})
        goal = clf.get("argument_goal", "")
        basis = clf.get("argument_basis", "")
        label = f"{goal}-{basis}"
        return label in ST3_LABELS
    return False


def parse_generated_lines(text: str) -> list[dict]:
    """Parse LLM output into list of JSON objects. Handles various formats."""
    results = []
    # Try line-by-line first
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # If no results, try parsing the whole thing as a JSON array
    if not results:
        try:
            arr = json.loads(text.strip())
            if isinstance(arr, list):
                results = [x for x in arr if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    return results


def append_to_jsonl(path: str, obj: dict):
    """Append a single JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: str) -> int:
    """Count existing lines in a JSONL file."""
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())
