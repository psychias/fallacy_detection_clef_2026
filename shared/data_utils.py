"""
shared/data_utils.py — Data loading, splitting, tokenization, and track guards.
"""

import json
import hashlib
import os
import random
from pathlib import Path
from collections import Counter

# ── Label maps ────────────────────────────────────────────────────────
ST1_LABEL2ID = {"non-fallacy": 0, "fallacy": 1}
ST1_ID2LABEL = {v: k for k, v in ST1_LABEL2ID.items()}

ST2_LABELS = [
    "authority", "black-white", "hasty_generalization", "natural",
    "population", "slippery_slope", "tradition", "worse_problems",
]
ST2_LABEL2ID = {l: i for i, l in enumerate(ST2_LABELS)}
ST2_ID2LABEL = {v: k for k, v in ST2_LABEL2ID.items()}

ST3_LABELS = [
    "epistemic-external", "epistemic-internal",
    "practical-external", "practical-internal",
]
ST3_LABEL2ID = {l: i for i, l in enumerate(ST3_LABELS)}
ST3_ID2LABEL = {v: k for k, v in ST3_LABEL2ID.items()}

JOINT_LABELS = ["non-fallacy"] + ST2_LABELS  # 9-way: non-fallacy + 8 fallacy types
JOINT_LABEL2ID = {l: i for i, l in enumerate(JOINT_LABELS)}
JOINT_ID2LABEL = {v: k for k, v in JOINT_LABEL2ID.items()}

LABEL_MAPS = {
    "st1": (ST1_LABEL2ID, ST1_ID2LABEL),
    "st2": (ST2_LABEL2ID, ST2_ID2LABEL),
    "st3": (ST3_LABEL2ID, ST3_ID2LABEL),
    "joint_st1st2": (JOINT_LABEL2ID, JOINT_ID2LABEL),
}

# Raw data label normalization
RAW_TO_OFFICIAL = {"blackwhite": "black-white"}

WORKSPACE = Path(os.environ.get(
    "FALLACY_WORKSPACE",
    "/content/drive/MyDrive/fallacy_detection"
))


def get_workspace():
    return WORKSPACE


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_fallacy_type(raw: str) -> str:
    """Map raw data labels to official submission labels."""
    return RAW_TO_OFFICIAL.get(raw, raw)


def get_st3_label(classification: dict) -> str:
    """Combine argument_goal and argument_basis into ST3 label."""
    return f"{classification['argument_goal']}-{classification['argument_basis']}"


def extract_label(entry: dict, subtask: str) -> str | None:
    """Extract the label for a given subtask from a data entry."""
    if subtask == "st1":
        return "fallacy" if entry["fallacy_exists"] == 1 else "non-fallacy"
    elif subtask == "st2":
        if entry.get("fallacy_exists") != 1:
            return None  # ST2 only applies to fallacious
        return normalize_fallacy_type(entry["fallacy_type"])
    elif subtask == "st3":
        if entry.get("fallacy_exists") != 0:
            return None  # ST3 only applies to non-fallacious
        return get_st3_label(entry["classification"])
    elif subtask == "joint_st1st2":
        if entry.get("fallacy_exists") == 0:
            return "non-fallacy"
        return normalize_fallacy_type(entry.get("fallacy_type", ""))
    raise ValueError(f"Unknown subtask: {subtask}")


def build_input_text(entry: dict, track: str, subtask: str) -> str:
    """
    Build the input text string for a given entry, respecting track constraints.
    Base track: only text_raw, text_raw_parent, text_raw_title, text_base, argument_base.
    Enhanced track: additionally text_enhanced, argument_enhanced.
    """
    if track == "base":
        parts = []
        if entry.get("text_raw_title"):
            parts.append(f"Title: {entry['text_raw_title']}")
        if entry.get("text_raw_parent"):
            parts.append(f"Parent: {entry['text_raw_parent']}")
        parts.append(f"Text: {entry['text_base']}")
        arg = entry.get("argument_base", {})
        if arg:
            parts.append(f"Claim: {arg.get('claim', '')}")
            supports = arg.get("supports", [])
            if supports:
                parts.append("Supports: " + " | ".join(supports))
        return "\n".join(parts)
    elif track == "enhanced":
        parts = []
        if entry.get("text_raw_title"):
            parts.append(f"Title: {entry['text_raw_title']}")
        if entry.get("text_raw_parent"):
            parts.append(f"Parent: {entry['text_raw_parent']}")
        text_val = entry.get('text_enhanced') or entry.get('text', '')
        parts.append(f"Text: {text_val}")
        arg = entry.get("argument_enhanced") or entry.get("argument", {})
        if arg:
            parts.append(f"Claim: {arg.get('claim', '')}")
            supports = arg.get("supports", [])
            if supports:
                parts.append("Supports: " + " | ".join(supports))
        return "\n".join(parts)
    elif track == "concat":
        parts = []
        if entry.get("text_raw_title"):
            parts.append(f"Title: {entry['text_raw_title']}")
        if entry.get("text_raw_parent"):
            parts.append(f"Parent: {entry['text_raw_parent']}")
        parts.append(f"Text: {entry.get('text_base', '')}")
        text_enh = entry.get('text_enhanced') or ''
        if text_enh:
            parts.append(f"Enhanced: {text_enh}")
        arg_base = entry.get("argument_base", {})
        if arg_base:
            parts.append(f"Claim: {arg_base.get('claim', '')}")
            supports = arg_base.get("supports", [])
            if supports:
                parts.append("Supports: " + " | ".join(supports))
        arg_enh = entry.get("argument_enhanced") or entry.get("argument", {})
        if arg_enh and arg_enh.get('claim'):
            parts.append(f"Enhanced claim: {arg_enh.get('claim', '')}")
        return "\n".join(parts)
    else:
        raise ValueError(f"Unknown track: {track}. Must be 'base', 'enhanced', or 'concat'.")


def filter_for_subtask(data: list[dict], subtask: str) -> list[dict]:
    """Filter data entries relevant to a subtask."""
    if subtask == "st1":
        return data  # all entries
    elif subtask == "st2":
        return [e for e in data if e.get("fallacy_exists") == 1]
    elif subtask == "st3":
        return [e for e in data if e.get("fallacy_exists") == 0]
    elif subtask == "joint_st1st2":
        return data  # all entries
    raise ValueError(f"Unknown subtask: {subtask}")


def make_splits(data: list[dict], subtask: str, seed: int = 42,
                dev_ratio: float = 0.15) -> tuple[list[dict], list[dict]]:
    """
    Create deterministic stratified train/dev split.
    Returns (train_split, dev_split).
    """
    filtered = filter_for_subtask(data, subtask)
    labels = [extract_label(e, subtask) for e in filtered]

    # Group by label for stratified split
    label_groups = {}
    for entry, label in zip(filtered, labels):
        if label is None:
            continue
        label_groups.setdefault(label, []).append(entry)

    rng = random.Random(seed)
    train_split, dev_split = [], []

    for label in sorted(label_groups.keys()):
        group = label_groups[label]
        rng.shuffle(group)
        n_dev = max(1, int(len(group) * dev_ratio))
        dev_split.extend(group[:n_dev])
        train_split.extend(group[n_dev:])

    # Shuffle each split deterministically
    rng.shuffle(train_split)
    rng.shuffle(dev_split)
    return train_split, dev_split


def compute_split_hash(split: list[dict]) -> str:
    """Compute a deterministic hash of a split's IDs for verification."""
    ids = sorted(e["id"] for e in split)
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def load_synth_data(synth_version: str, subtask: str) -> list[dict]:
    """Load synthetic data from data_synth/<synth_version>/data.jsonl."""
    ws = get_workspace()
    path = ws / "data_synth" / synth_version / "data.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Synth data not found: {path}")
    data = load_jsonl(str(path))
    return filter_for_subtask(data, subtask)


def prepare_dataset(config: dict, tokenizer, max_len: int = 256):
    """
    Prepare train/dev datasets based on config.
    Returns (train_entries, dev_entries, label2id, id2label).
    """
    ws = get_workspace()
    subtask = config["subtask"]
    track = config["track"]
    label2id, id2label = LABEL_MAPS[subtask]

    # Load real data
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    train_split, dev_split = make_splits(all_data, subtask)

    # Build training data
    train_entries = []
    if config.get("data", {}).get("real", True):
        for e in train_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                train_entries.append({
                    "id": e["id"], "text": text, "label": label,
                    "label_id": label2id[label]
                })

    # Add synthetic data
    synth_versions = config.get("data", {}).get("synth_versions", [])
    for sv in synth_versions:
        synth_data = load_synth_data(sv, subtask)
        for e in synth_data:
            label = extract_label(e, subtask)
            if label is not None and label in label2id:
                text = build_input_text(e, track, subtask)
                train_entries.append({
                    "id": e.get("id", f"synth_{sv}"),
                    "text": text, "label": label,
                    "label_id": label2id[label]
                })

    # Build dev data (always from real data only)
    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            dev_entries.append({
                "id": e["id"], "text": text, "label": label,
                "label_id": label2id[label]
            })

    return train_entries, dev_entries, label2id, id2label


def load_test_data(track: str, subtask: str) -> list[dict]:
    """Load test data for prediction."""
    ws = get_workspace()
    test_path = ws / "data" / "touchefallacy_2026_test_task.jsonl"
    all_data = load_jsonl(str(test_path))
    entries = []
    for e in all_data:
        text = build_input_text(e, track, subtask)
        entries.append({"id": e["id"], "text": text})
    return entries


def get_class_weights(train_entries: list[dict], num_classes: int) -> list[float]:
    """Compute inverse-frequency class weights for imbalanced training."""
    counts = Counter(e["label_id"] for e in train_entries)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))
    return weights
