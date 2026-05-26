"""
R1c — Predict on test set + format to shared task submission spec + validate.
Usage: python predict_and_format.py <exp_dir> [--track base|enhanced] [--output submissions/]

Loads the best checkpoint from exp_dir/ckpt/best, runs inference on the test
set, writes submission JSONL, and validates the output format.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent
sys.path.insert(0, str(workspace))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    filter_for_subtask,
    load_jsonl,
)

# Subtask → shared task name mapping
TASK_NAMES = {
    "st1": "fallacy_detection",
    "st2": "fallacy_classification",
    "st3": "scheme_classification",
}


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
            "entry_id": e["id"],
        }


def predict(exp_dir: Path, track_override: str = None):
    """Load checkpoint, predict on test, return list of (id, label) tuples."""
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
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), num_labels=num_labels
    ).to(device)
    model.eval()

    # Load test data
    test_path = workspace / "data" / "touchefallacy_2026_test_task.jsonl"
    test_data = load_jsonl(str(test_path))
    filtered = filter_for_subtask(test_data, subtask)

    entries = []
    for e in filtered:
        text = build_input_text(e, track, subtask)
        entries.append({"id": e["id"], "text": text})

    ds = TextDataset(entries, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.float().argmax(dim=-1).cpu().tolist()
            for entry_id, pred_id in zip(batch["entry_id"], preds):
                label = id2label[pred_id]
                predictions.append((entry_id, label))

    print(f"[predict] {subtask}/{track}: {len(predictions)} predictions")
    pred_dist = Counter(label for _, label in predictions)
    print(f"[predict] distribution: {dict(sorted(pred_dist.items()))}")
    return subtask, predictions


def format_submission(subtask: str, predictions: list, output_path: Path,
                      tag: str = "base", system_description: str = ""):
    """Write predictions to JSONL submission format."""
    task_name = TASK_NAMES[subtask]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry_id, label in predictions:
            line = json.dumps({
                "task": task_name,
                "id": entry_id,
                "label": label,
                "tag": tag,
                "system_description": system_description,
            }, ensure_ascii=False)
            f.write(line + "\n")

    print(f"[format] Wrote {len(predictions)} lines to {output_path}")


def validate_submission(path: Path, subtask: str) -> list[str]:
    """Validate submission JSONL format. Returns list of error messages."""
    errors = []
    task_name = TASK_NAMES[subtask]
    label2id, _ = LABEL_MAPS[subtask]
    valid_labels = set(label2id.keys())

    # Load test IDs for completeness check
    test_path = workspace / "data" / "touchefallacy_2026_test_task.jsonl"
    test_data = load_jsonl(str(test_path))
    filtered = filter_for_subtask(test_data, subtask)
    expected_ids = {e["id"] for e in filtered}

    seen_ids = set()
    line_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON: {e}")
                continue

            line_count += 1

            # Check required fields
            for field in ("task", "id", "label"):
                if field not in obj:
                    errors.append(f"Line {i}: missing field '{field}'")

            # Check task name
            if obj.get("task") != task_name:
                errors.append(f"Line {i}: expected task='{task_name}', "
                              f"got '{obj.get('task')}'")

            # Check label validity
            if obj.get("label") not in valid_labels:
                errors.append(f"Line {i}: invalid label '{obj.get('label')}' "
                              f"(valid: {valid_labels})")

            # Check for duplicates
            eid = obj.get("id")
            if eid in seen_ids:
                errors.append(f"Line {i}: duplicate id '{eid}'")
            seen_ids.add(eid)

    # Check completeness
    missing = expected_ids - seen_ids
    extra = seen_ids - expected_ids
    if missing:
        errors.append(f"Missing {len(missing)} IDs: {sorted(missing)[:5]}...")
    if extra:
        errors.append(f"Extra {len(extra)} IDs not in test set: {sorted(extra)[:5]}...")

    if not errors:
        print(f"[validate] {path.name}: PASS ({line_count} lines, "
              f"{len(expected_ids)} expected)")
    else:
        print(f"[validate] {path.name}: FAIL ({len(errors)} errors)")
        for e in errors[:10]:
            print(f"  - {e}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Predict + format + validate")
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--track", type=str, default=None,
                        help="Override track (base/enhanced)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for submission file")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} does not exist")
        sys.exit(1)

    subtask, predictions = predict(exp_dir, args.track)

    # Determine output path
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = workspace / "submissions"

    config = json.load(open(exp_dir / "config.json", encoding="utf-8-sig"))
    track = args.track or config.get("track", "base")
    filename = f"submission_{subtask}_{track}_{exp_dir.name}.jsonl"
    out_path = out_dir / filename

    format_submission(subtask, predictions, out_path)
    errors = validate_submission(out_path, subtask)

    if errors:
        print(f"\n[FAIL] {len(errors)} validation errors found!")
        sys.exit(1)
    else:
        print(f"\n[PASS] Submission validated successfully: {out_path}")


if __name__ == "__main__":
    main()
