"""
generate_submissions.py — Generate submission files for Touché 2026 Fallacy Detection.

6 TIRA slots (one output file each):

  Slot          Exp       Dev-F1   CV-F1   max_len  Notes
  ST1 base      exp_0161  0.761    —       512      raw_concat input
  ST1 enhanced  exp_0035  0.914    —       256
  ST2 base      exp_0032  0.730    —       256
  ST2 enhanced  exp_0072  0.970    —       384
  ST3 base      exp_0069  0.523    —       384
  ST3 enhanced  exp_0073  0.735    0.875   384      RoBERTa CV > Qwen-32B CV(stale=0.841)

Non-routed inference: every model runs independently on all 233 test entries.
TIRA evaluates ST2 on ground-truth fallacious entries, ST3 on non-fallacious.
Routing would lose entries where ST1 mispredicts — non-routed avoids that.

Output: submissions/submission_<subtask>_<track>.jsonl (6 files)
"""

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WORKSPACE = Path(os.environ.get(
    "FALLACY_WORKSPACE",
    "/content/drive/MyDrive/fallacy_detection"
))
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))

from shared.data_utils import (
    ST1_ID2LABEL, ST2_ID2LABEL, ST3_ID2LABEL,
    build_input_text, load_jsonl,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PATH = WORKSPACE / "data" / "touchefallacy_2026_test_task.jsonl"

# Best checkpoint per (subtask, track) — ALL RoBERTa-large
MODELS = {
    ("st1", "enhanced"): "experiments/exp_0035/ckpt/best",
    ("st1", "base"):     "experiments/exp_0161/ckpt/best",
    ("st2", "enhanced"): "experiments/exp_0072/ckpt/best",
    ("st2", "base"):     "experiments/exp_0032/ckpt/best",
    ("st3", "enhanced"): "experiments/exp_0073/ckpt/best",
    ("st3", "base"):     "experiments/exp_0069/ckpt/best",
}

# max_len must match each experiment's training config
MAX_LENS = {
    ("st1", "enhanced"): 256,   # exp_0035
    ("st1", "base"):     512,   # exp_0161 (raw_concat)
    ("st2", "enhanced"): 384,   # exp_0072
    ("st2", "base"):     256,   # exp_0032
    ("st3", "enhanced"): 384,   # exp_0073
    ("st3", "base"):     384,   # exp_0069
}

TASK_NAMES = {
    "st1": "fallacy_detection",
    "st2": "fallacy_classification",
    "st3": "scheme_classification",
}

SYSTEM_DESCRIPTION = (
    "RoBERTa-large fine-tuned on Touché 2026 training data with synthetic "
    "augmentation and pseudo-labeling. Non-routed inference on all test entries."
)

ID2LABEL = {
    "st1": ST1_ID2LABEL,
    "st2": ST2_ID2LABEL,
    "st3": ST3_ID2LABEL,
}


def build_raw_concat_text(entry):
    """exp_0161 uses raw_concat: title + parent + text_base + text_raw + argument_base."""
    parts = []
    if entry.get("text_raw_title"):
        parts.append(f"Title: {entry['text_raw_title']}")
    if entry.get("text_raw_parent"):
        parts.append(f"Parent: {entry['text_raw_parent']}")
    parts.append(f"Text: {entry.get('text_base', '')}")
    raw = entry.get("text_raw", "")
    if raw and raw != entry.get("text_base", ""):
        parts.append(f"Raw: {raw}")
    arg = entry.get("argument_base", {})
    if arg:
        parts.append(f"Claim: {arg.get('claim', '')}")
        supports = arg.get("supports", [])
        if supports:
            parts.append("Supports: " + " | ".join(supports))
    return "\n".join(parts)


def get_input_text(entry, subtask, track):
    if subtask == "st1" and track == "base":
        return build_raw_concat_text(entry)
    return build_input_text(entry, track, subtask)


class InferenceDataset(Dataset):
    def __init__(self, entries, tokenizer, max_len, subtask, track):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.subtask = subtask
        self.track = track

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        text = get_input_text(entry, self.subtask, self.track)
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


@torch.no_grad()
def predict(entries, subtask, track, batch_size=16):
    """Run inference on all entries and return predicted labels."""
    ckpt_path = str(WORKSPACE / MODELS[(subtask, track)])
    max_len = MAX_LENS[(subtask, track)]
    print(f"  Loading {ckpt_path}  (max_len={max_len})...")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.to(DEVICE).eval()

    dataset = InferenceDataset(entries, tokenizer, max_len, subtask, track)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)

    labels = [ID2LABEL[subtask][p] for p in all_preds]

    del model
    torch.cuda.empty_cache()

    from collections import Counter
    dist = Counter(labels)
    print(f"  → {len(labels)} predictions: {dict(sorted(dist.items()))}")
    return labels


def generate_submission(subtask, track, test_data):
    """
    Generate one submission file for a single subtask/track slot.

    Non-routed: runs the subtask model on ALL 233 test entries.
    TIRA selects which predictions to score using ground-truth labels.
    """
    print(f"\n{'─'*60}")
    print(f"[{subtask.upper()} {track}]  exp={Path(MODELS[(subtask, track)]).parent.parent.name}")
    print(f"{'─'*60}")

    labels = predict(test_data, subtask, track)

    out_dir = WORKSPACE / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"submission_{subtask}_{track}.jsonl"

    task_name = TASK_NAMES[subtask]
    with open(out_path, "w", encoding="utf-8") as f:
        for entry, label in zip(test_data, labels):
            f.write(json.dumps({
                "task": task_name,
                "id": entry["id"],
                "label": label,
                "tag": track,
                "system_description": SYSTEM_DESCRIPTION,
            }, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(test_data)} lines → {out_path.name}")
    return out_path


if __name__ == "__main__":
    print("Touché 2026 — Submission Generation")
    print(f"Device: {DEVICE}")
    print(f"Test file: {TEST_PATH}")

    test_data = load_jsonl(str(TEST_PATH))
    print(f"Test entries: {len(test_data)}")

    slots = [
        ("st1", "base"),
        ("st1", "enhanced"),
        ("st2", "base"),
        ("st2", "enhanced"),
        ("st3", "base"),
        ("st3", "enhanced"),
    ]

    generated = []
    for subtask, track in slots:
        path = generate_submission(subtask, track, test_data)
        generated.append(path)

    print(f"\n{'='*60}")
    print(f"DONE — {len(generated)} submission files:")
    for p in generated:
        lines = sum(1 for _ in open(p))
        print(f"  {p.name}  ({lines} lines)")
    print(f"{'='*60}")
