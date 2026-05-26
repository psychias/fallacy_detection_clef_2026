"""
exp_0122 — Submission dry-run.

Loads best checkpoints for each (subtask × track) pair, generates
predictions on the test set in official JSONL format, and validates
the output schema. Saves submission_enhanced.jsonl and submission_base.jsonl.
"""

import json
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    load_jsonl,
    load_test_data,
)
from shared.train_utils import (
    Heartbeat,
    WallClockGuard,
    get_device,
    load_config,
    set_seed,
    write_status,
    write_traceback,
)
from shared.eval import write_metrics

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
            "id": e["id"],
        }


def predict_test(exp_id, subtask, track, ws, device):
    """Load checkpoint from exp_id and predict on test data."""
    src_dir = ws / "experiments" / exp_id
    ckpt_dir = src_dir / "ckpt" / "best"

    if not ckpt_dir.exists():
        print(f"[submit] WARNING: {exp_id} has no checkpoint")
        return None

    src_config = json.loads((src_dir / "config.json").read_text(encoding="utf-8-sig"))
    max_len = src_config.get("train", {}).get("max_len", 256)
    src_track = src_config.get("track", track)

    label2id, id2label = LABEL_MAPS[subtask]

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir), num_labels=len(label2id)
    ).to(device)
    model.eval()

    # Load test data using the source model's track
    test_entries = load_test_data(src_track, subtask)
    ds = TextDataset(test_entries, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            for idx, pred_id in enumerate(preds):
                predictions.append({
                    "task": TASK_NAMES[subtask],
                    "id": batch["id"][idx],
                    "label": id2label[pred_id],
                })

    del model, tokenizer
    torch.cuda.empty_cache()

    return predictions


def validate_submission(lines, expected_tasks):
    """Validate JSONL submission format."""
    errors = []
    seen_task_ids = set()

    for i, line in enumerate(lines):
        # Check required fields
        for field in ["task", "id", "label"]:
            if field not in line:
                errors.append(f"Line {i}: missing field '{field}'")

        task = line.get("task", "")
        if task not in expected_tasks:
            errors.append(f"Line {i}: unexpected task '{task}'")

        # Check for duplicate (task, id) pairs
        key = (task, line.get("id", ""))
        if key in seen_task_ids:
            errors.append(f"Line {i}: duplicate (task, id) = {key}")
        seen_task_ids.add(key)

        # Validate label for each task
        if task == "fallacy_detection":
            valid = {"fallacy", "non-fallacy"}
        elif task == "fallacy_classification":
            valid = {"authority", "black-white", "hasty_generalization", "natural",
                     "population", "slippery_slope", "tradition", "worse_problems"}
        elif task == "scheme_classification":
            valid = {"practical-internal", "practical-external",
                     "epistemic-internal", "epistemic-external"}
        else:
            valid = set()

        if line.get("label", "") not in valid:
            errors.append(f"Line {i}: invalid label '{line.get('label')}' for task '{task}'")

    return errors


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 600), margin_s=60)

    ws = workspace
    device = get_device()
    source_exps = config["source_exps"]
    results = {}

    # Load test data to get expected IDs
    test_data = load_jsonl(str(ws / "data" / "touchefallacy_2026_test_task.jsonl"))
    test_ids = [e["id"] for e in test_data]
    n_test = len(test_ids)
    print(f"[submit] Test set: {n_test} entries")

    # ── Generate predictions per track ────────────────────────────
    for track_name in ["enhanced", "base"]:
        all_lines = []

        for subtask in ["st1", "st2", "st3"]:
            key = f"{subtask}_{track_name[:3]}"
            exp_id = source_exps.get(key)
            if not exp_id:
                print(f"[submit] No source for {key}, skipping")
                continue

            if guard.exceeded():
                break

            print(f"[submit] Predicting {subtask} {track_name} from {exp_id}")
            preds = predict_test(exp_id, subtask, track_name, ws, device)

            if preds:
                all_lines.extend(preds)
                print(f"[submit] {subtask}: {len(preds)} predictions")
            else:
                print(f"[submit] {subtask}: FAILED to predict")

            heartbeat.beat()

        # Validate
        expected_tasks = {"fallacy_detection", "fallacy_classification", "scheme_classification"}
        errors = validate_submission(all_lines, expected_tasks)

        # Write JSONL
        out_path = exp_dir / f"submission_{track_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # Also write to submissions/ directory
        submissions_dir = ws / "submissions"
        submissions_dir.mkdir(exist_ok=True)
        final_path = submissions_dir / f"submission_{track_name}_dryrun.jsonl"
        with open(final_path, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # Count per task
        task_counts = {}
        for line in all_lines:
            t = line["task"]
            task_counts[t] = task_counts.get(t, 0) + 1

        results[track_name] = {
            "n_predictions": len(all_lines),
            "n_test_entries": n_test,
            "task_counts": task_counts,
            "validation_errors": errors[:20],
            "n_errors": len(errors),
            "valid": len(errors) == 0,
        }

        status = "PASS" if len(errors) == 0 else "FAIL"
        print(f"[submit] {track_name}: {len(all_lines)} lines, {len(errors)} errors → {status}")

    results["wall_clock_s"] = round(guard.elapsed(), 1)

    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("[submit] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[submit] CRASHED: {exc}")
        sys.exit(1)
