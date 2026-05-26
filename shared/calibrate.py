"""
Per-class threshold calibration on dev set.
Finds optimal per-class logit multipliers that maximize macro F1.

Usage:
    python calibrate.py --exp_dir experiments/exp_XXXX [--track base|enhanced]

Outputs calibration.json in exp_dir with per-class multipliers.
At inference, multiply logits by these before argmax.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    extract_label,
    load_jsonl,
    make_splits,
)


class TextDataset(Dataset):
    def __init__(self, entries, tokenizer, max_len):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        enc = self.tokenizer(
            e["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_id": e["label_id"],
        }


def get_dev_logits(exp_dir: Path, track_override: str = None):
    """Get raw logits and true labels on dev set."""
    config_path = exp_dir / "config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    subtask = config["subtask"]
    track = track_override or config.get("track", "base")
    backbone = config["backbone"]
    max_len = config.get("train", {}).get("max_len", 256)

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    ckpt_path = exp_dir / "ckpt" / "best"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), num_labels=num_labels
    ).to(device)
    model.eval()

    # Load dev split
    train_path = workspace / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    _, dev_split = make_splits(all_data, subtask)

    entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None and label in label2id:
            text = build_input_text(e, track, subtask)
            entries.append({
                "text": text,
                "label_id": label2id[label],
            })

    ds = TextDataset(entries, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.float().cpu())
            all_labels.append(batch["label_id"])

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return logits, labels, id2label, subtask


def grid_search_multipliers(logits, labels, num_labels, grid_values=None):
    """
    Grid search over per-class logit multipliers to maximize macro F1.
    Uses 3-fold cross-validation on dev to prevent overfitting.
    Range capped at [0.7, 1.4] with coarse grid.
    """
    if grid_values is None:
        grid_values = [0.7, 0.85, 1.0, 1.15, 1.4]

    n = len(labels)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    n_folds = 3
    folds = [indices[i::n_folds] for i in range(n_folds)]

    # Collect best multipliers per fold
    fold_mults = []
    fold_f1s = []

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
        train_logits = logits[train_idx]
        train_labels = labels[train_idx]

        best_f1 = -1
        best_mults = [1.0] * num_labels

        if num_labels <= 4:
            for mults in product(grid_values, repeat=num_labels):
                scaled = train_logits * np.array(mults)
                preds = scaled.argmax(axis=1)
                f1 = f1_score(train_labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_mults = list(mults)
        else:
            current_mults = [1.0] * num_labels
            for _ in range(3):
                for cls in range(num_labels):
                    best_val = current_mults[cls]
                    for v in grid_values:
                        trial = current_mults.copy()
                        trial[cls] = v
                        scaled = train_logits * np.array(trial)
                        preds = scaled.argmax(axis=1)
                        f1 = f1_score(train_labels, preds, average="macro",
                                      zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_val = v
                    current_mults[cls] = best_val
            best_mults = current_mults

        # Evaluate on held-out fold
        val_scaled = logits[val_idx] * np.array(best_mults)
        val_preds = val_scaled.argmax(axis=1)
        val_f1 = f1_score(labels[val_idx], val_preds, average="macro", zero_division=0)
        fold_mults.append(best_mults)
        fold_f1s.append(val_f1)
        print(f"  fold {fold_idx}: mults={[round(m, 2) for m in best_mults]}, "
              f"train_f1={best_f1:.4f}, val_f1={val_f1:.4f}")

    # Take median multiplier per class across folds (robust to outlier fold)
    final_mults = [float(np.median([fold_mults[f][c] for f in range(n_folds)]))
                   for c in range(num_labels)]

    # Evaluate final multipliers on full dev
    scaled = logits * np.array(final_mults)
    preds = scaled.argmax(axis=1)
    final_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return final_mults, final_f1


def main():
    parser = argparse.ArgumentParser(description="Calibrate per-class thresholds")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--track", default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    logits, labels, id2label, subtask = get_dev_logits(exp_dir, args.track)
    num_labels = logits.shape[1]

    # Baseline (no calibration)
    baseline_preds = logits.argmax(axis=1)
    baseline_f1 = f1_score(labels, baseline_preds, average="macro",
                           zero_division=0)

    # Grid search
    best_mults, best_f1 = grid_search_multipliers(logits, labels, num_labels)

    print(f"[calibrate] {subtask}: baseline_f1={baseline_f1:.5f}, "
          f"calibrated_f1={best_f1:.5f}, delta={best_f1 - baseline_f1:+.5f}")
    print(f"[calibrate] multipliers:")
    for cls_id, mult in enumerate(best_mults):
        print(f"  {id2label[cls_id]}: {mult:.2f}")

    # Save
    calib = {
        "subtask": subtask,
        "baseline_f1": round(baseline_f1, 5),
        "calibrated_f1": round(best_f1, 5),
        "multipliers": {id2label[i]: round(m, 2) for i, m in enumerate(best_mults)},
    }
    out_path = exp_dir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[calibrate] Saved to {out_path}")


if __name__ == "__main__":
    main()
