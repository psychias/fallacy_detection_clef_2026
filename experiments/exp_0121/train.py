"""
exp_0121 — Post-hoc ensemble: temperature calibration + logit averaging.

For each subtask, loads the best checkpoint(s), runs inference on dev,
calibrates temperature via Platt scaling on dev logits, then averages
calibrated probabilities if multiple models are available.

Also saves calibrated dev logits for all source models — these become
building blocks for the final ensemble in Phase 3.
"""

import json
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    extract_label,
    load_jsonl,
    make_splits,
)
from shared.eval import evaluate, write_metrics
from shared.train_utils import (
    Heartbeat,
    WallClockGuard,
    get_device,
    load_config,
    set_seed,
    write_status,
    write_traceback,
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
            "label": torch.tensor(e["label_id"], dtype=torch.long),
            "id": e["id"],
        }


def get_dev_logits(exp_id, subtask, track, ws, device):
    """Load model from exp checkpoint and get dev logits."""
    src_dir = ws / "experiments" / exp_id
    ckpt_dir = src_dir / "ckpt" / "best"
    if not ckpt_dir.exists():
        print(f"[ensemble] WARNING: {exp_id} has no checkpoint, skipping")
        return None, None, None

    src_config = json.loads((src_dir / "config.json").read_text(encoding="utf-8-sig"))
    max_len = src_config.get("train", {}).get("max_len", 256)

    label2id, id2label = LABEL_MAPS[subtask]

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir), num_labels=len(label2id)
    ).to(device)
    model.eval()

    # Build dev entries
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    _, dev_split = make_splits(all_data, subtask)

    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            dev_entries.append({
                "id": e["id"], "text": text, "label": label,
                "label_id": label2id[label],
            })

    ds = TextDataset(dev_entries, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    all_logits = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_logits.append(out.logits.float().cpu().numpy())
            all_labels.extend(batch["label"].tolist())
            all_ids.extend(batch["id"])

    logits = np.concatenate(all_logits, axis=0)
    del model, tokenizer
    torch.cuda.empty_cache()

    return logits, all_labels, all_ids


def calibrate_temperature(logits, labels):
    """Find optimal temperature T that minimizes NLL on dev set."""
    def nll(T):
        scaled = logits / T
        # Numerically stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
        nll_val = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
        return nll_val

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 600), margin_s=60)

    track = config["track"]
    source_exps = config["source_exps"]  # {"st1": [...], "st2": [...], "st3": [...]}

    ws = workspace
    device = get_device()
    results = {}

    for subtask, exp_ids in source_exps.items():
        if guard.exceeded():
            break

        label2id, id2label = LABEL_MAPS[subtask]
        print(f"\n[ensemble] === {subtask} === sources: {exp_ids}")

        all_logits_list = []
        all_temps = []
        ref_labels = None
        ref_ids = None

        for exp_id in exp_ids:
            logits, labels, ids = get_dev_logits(exp_id, subtask, track, ws, device)
            if logits is None:
                continue

            if ref_labels is None:
                ref_labels = labels
                ref_ids = ids

            # Calibrate temperature
            T = calibrate_temperature(logits, labels)
            all_temps.append({"exp": exp_id, "temperature": round(T, 4)})
            calibrated = logits / T
            all_logits_list.append(calibrated)

            # Report individual model performance
            probs = softmax(calibrated)
            preds = probs.argmax(axis=1)
            y_true = [id2label[l] for l in labels]
            y_pred = [id2label[p] for p in preds]
            m = evaluate(y_true, y_pred, subtask)
            print(f"[ensemble] {exp_id}: T={T:.3f}, calibrated F1={m['f1_macro']:.4f}")

            heartbeat.beat()

        if not all_logits_list:
            results[subtask] = {"error": "no valid source models"}
            continue

        # Average calibrated logits
        avg_logits = np.mean(all_logits_list, axis=0)
        avg_probs = softmax(avg_logits)
        preds = avg_probs.argmax(axis=1)

        y_true = [id2label[l] for l in ref_labels]
        y_pred = [id2label[p] for p in preds]
        metrics = evaluate(y_true, y_pred, subtask)

        # Save per-example calibrated probabilities for downstream use
        dev_calibrated = []
        for i in range(len(ref_ids)):
            dev_calibrated.append({
                "id": ref_ids[i],
                "true_label": y_true[i],
                "avg_probs": avg_probs[i].tolist(),
                "pred_label": y_pred[i],
            })

        results[subtask] = {
            "f1_macro": metrics["f1_macro"],
            "f1_per_class": metrics["f1_per_class"],
            "temperatures": all_temps,
            "n_sources": len(all_logits_list),
            "dev_calibrated": dev_calibrated,
        }
        print(f"[ensemble] {subtask} ensemble: F1={metrics['f1_macro']:.4f}")

    results["track"] = track
    results["wall_clock_s"] = round(guard.elapsed(), 1)

    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("\n[ensemble] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[ensemble] CRASHED: {exc}")
        sys.exit(1)
