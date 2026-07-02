"""
shared/eval.py — Evaluation utilities for Touché 2026 Fallacy Detection.
Computes precision, recall, macro-F1 per subtask.
This is the ONLY scorer used across all experiments.
"""

from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
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


def confusion_dict(y_true: list[str], y_pred: list[str], subtask: str) -> dict:
    """
    Labelled confusion matrix for a subtask. Rows are the gold label, columns
    are the predicted label, both in the canonical LABEL_MAPS[subtask] order
    (also returned under "labels" so the matrix is self-documenting).
    """
    labels = LABEL_MAPS[subtask]
    y_true = [normalize_label(y, subtask) for y in y_true]
    y_pred = [normalize_label(y, subtask) for y in y_pred]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "matrix": cm.tolist(),
        "axes": {"rows": "true", "cols": "pred"},
        "subtask": subtask,
    }


def save_dev_logits(exp_dir, ids, label_ids, logits, id2label, subtask):
    """
    Persist per-example dev logits so the per-class breakdown and confusion
    matrix are reconstructible from committed artefacts (the gap the paper's
    §4.4 flags for exp_0073). Writes ``dev_logits.npz`` next to the run's
    metrics. ``logits`` is an (N, num_labels) array; column j corresponds to
    ``id2label[j]``. Returns the .npz path.
    """
    import numpy as np
    from pathlib import Path

    exp_dir = Path(exp_dir)
    num_labels = len(id2label)
    label_order = [id2label[i] for i in range(num_labels)]
    out = exp_dir / "dev_logits.npz"
    np.savez_compressed(
        out,
        ids=np.array(list(ids), dtype=object),
        label_ids=np.array(list(label_ids), dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32),
        label_order=np.array(label_order, dtype=object),
        subtask=np.array(subtask),
    )
    return str(out)


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
