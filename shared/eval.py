"""
shared/eval.py — Evaluation utilities for Touché 2026 Fallacy Detection.
Computes precision, recall, macro-F1 per subtask.
This is the ONLY scorer used across all experiments.
"""

from sklearn.metrics import precision_recall_fscore_support, classification_report
import json

# ── Label vocabularies ────────────────────────────────────────────────
ST1_LABELS = ["non-fallacy", "fallacy"]
ST2_LABELS = [
    "authority", "black-white", "hasty_generalization", "natural",
    "population", "slippery_slope", "tradition", "worse_problems",
]
ST3_LABELS = [
    "epistemic-external", "epistemic-internal",
    "practical-external", "practical-internal",
]

JOINT_LABELS = ["non-fallacy"] + ST2_LABELS  # 9-way: non-fallacy + 8 fallacy types

LABEL_MAPS = {
    "st1": ST1_LABELS,
    "st2": ST2_LABELS,
    "st3": ST3_LABELS,
    "joint_st1st2": JOINT_LABELS,
}

# Mapping from data's raw labels to official submission labels
RAW_TO_OFFICIAL = {
    "blackwhite": "black-white",
}


def normalize_label(label: str, subtask: str) -> str:
    """Map raw data labels to official submission labels."""
    label = label.strip().lower()
    label = RAW_TO_OFFICIAL.get(label, label)
    if subtask == "st1":
        if label in ("1", "fallacy"):
            return "fallacy"
        return "non-fallacy"
    return label


def evaluate(y_true: list[str], y_pred: list[str], subtask: str) -> dict:
    """
    Compute evaluation metrics for a subtask.
    Returns dict with precision, recall, f1_macro, f1_per_class.
    """
    labels = LABEL_MAPS[subtask]
    # Normalize
    y_true = [normalize_label(y, subtask) for y in y_true]
    y_pred = [normalize_label(y, subtask) for y in y_pred]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, f1_per_class, sup_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "precision": round(float(precision), 5),
        "recall": round(float(recall), 5),
        "f1_macro": round(float(f1), 5),
        "f1_per_class": {
            lbl: round(float(f1_per_class[i]), 5)
            for i, lbl in enumerate(labels)
        },
        "support_per_class": {
            lbl: int(sup_per_class[i])
            for i, lbl in enumerate(labels)
        },
        "classification_report": report,
        "subtask": subtask,
    }


def write_metrics(metrics: dict, path: str):
    """Write metrics dict to a JSON file. Falls back to raw dump on failure."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    except Exception:
        # Last-resort: dump as string so we never lose results
        fallback = str(path) + ".fallback.json"
        try:
            safe = {k: str(v) for k, v in metrics.items()}
            with open(fallback, "w", encoding="utf-8") as f:
                json.dump(safe, f, indent=2)
        except Exception:
            pass
