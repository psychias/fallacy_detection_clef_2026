"""
Ensemble prediction: combine logits from multiple checkpoints with optional
per-model weights and per-class calibration multipliers.

Usage:
    python ensemble_predict.py --config ensemble_config.json --output submissions/

Ensemble config JSON format:
{
    "subtask": "st3",
    "track": "base",
    "models": [
        {"exp": "exp_0057", "weight": 1.0},
        {"exp": "exp_0068", "weight": 1.2}
    ],
    "calibration": {"epistemic-external": 1.5, "practical-external": 2.0, ...}
}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    filter_for_subtask,
    load_jsonl,
)
from shared.predict_and_format import TASK_NAMES, format_submission, validate_submission


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
            "idx": idx,
        }


def get_logits(exp_dir: Path, entries: list, track: str = None):
    """Get raw logits for entries from a checkpoint."""
    config_path = exp_dir / "config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    subtask = config["subtask"]
    if track is None:
        track = config.get("track", "base")
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

    # Build text for this model's track/max_len
    model_entries = []
    for e in entries:
        text = build_input_text(e["raw"], track, subtask)
        model_entries.append({"text": text})

    ds = TextDataset(model_entries, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.float().cpu().numpy())

    return np.concatenate(all_logits, axis=0), id2label


def ensemble_predict(config: dict, workspace_dir: Path):
    """
    Run ensemble prediction.
    Returns (subtask, predictions) like predict_and_format.predict().
    """
    subtask = config["subtask"]
    track = config.get("track", "base")
    models = config["models"]
    calibration = config.get("calibration", {})

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    # Load test data once
    test_path = workspace_dir / "data" / "touchefallacy_2026_test_task.jsonl"
    test_data = load_jsonl(str(test_path))
    filtered = filter_for_subtask(test_data, subtask)
    entries = [{"id": e["id"], "raw": e} for e in filtered]

    print(f"[ensemble] {subtask}/{track}: {len(entries)} test entries, "
          f"{len(models)} models")

    # Collect weighted logits
    summed_logits = np.zeros((len(entries), num_labels))

    for m in models:
        exp_name = m["exp"]
        weight = m.get("weight", 1.0)
        model_track = m.get("track", track)
        exp_dir = workspace_dir / "experiments" / exp_name

        print(f"  Loading {exp_name} (weight={weight}, track={model_track})...")
        logits, _ = get_logits(exp_dir, entries, track=model_track)

        # Normalize to probabilities before weighting (avoids scale issues)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        summed_logits += weight * probs

    # Apply per-class calibration multipliers
    if calibration:
        calib_vector = np.array([
            calibration.get(id2label[i], 1.0) for i in range(num_labels)
        ])
        summed_logits *= calib_vector
        print(f"  Applied calibration: {calibration}")

    # Argmax
    pred_ids = summed_logits.argmax(axis=1)
    predictions = []
    for entry, pred_id in zip(entries, pred_ids):
        predictions.append((entry["id"], id2label[pred_id]))

    # Distribution
    from collections import Counter
    dist = Counter(label for _, label in predictions)
    print(f"[ensemble] distribution: {dict(sorted(dist.items()))}")

    return subtask, predictions


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction")
    parser.add_argument("--config", required=True,
                        help="Path to ensemble config JSON")
    parser.add_argument("--output", required=True,
                        help="Output submission JSONL path")
    parser.add_argument("--system_description", default="",
                        help="System description for submission")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    subtask, predictions = ensemble_predict(config, workspace)

    output_path = Path(args.output)
    tag = config.get("track", "base")
    format_submission(subtask, predictions, output_path,
                      tag=tag, system_description=args.system_description)

    errors = validate_submission(output_path, subtask)
    if errors:
        print(f"[ensemble] VALIDATION ERRORS: {errors}")
    else:
        print(f"[ensemble] Submission validated OK")


if __name__ == "__main__":
    main()
