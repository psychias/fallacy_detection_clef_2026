"""
Pseudo-labeling: predict test set with a trained checkpoint, take top-K
highest-confidence predictions per class, output as additional training data.

Usage (from Colab runner or locally):
    python pseudo_label.py --exp_dir experiments/exp_XXXX --top_k 30 --output data_synth/pseudo_v001_stN/data.jsonl

The output format matches synth data format so it integrates directly into
the training pipeline via synth_versions in config.json.
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


def _infer_probs(exp_dir: Path, raw_entries, subtask, track, device):
    """Run inference with one checkpoint, return probability matrix (N x C)."""
    config_path = exp_dir / "config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    max_len = config.get("train", {}).get("max_len", 256)
    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    ckpt_path = exp_dir / "ckpt" / "best"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), num_labels=num_labels
    ).to(device)
    model.eval()

    ds = TextDataset(raw_entries, tokenizer, max_len)
    all_probs = np.zeros((len(raw_entries), num_labels), dtype=np.float32)
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=16, shuffle=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["idx"].tolist()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
            for i, idx_val in enumerate(indices):
                all_probs[idx_val] = probs[i]

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[pseudo] Inferred {len(raw_entries)} examples from {exp_dir.name}")
    return all_probs


def pseudo_label(exp_dir, top_k: int = 30, min_confidence: float = 0.7,
                 track_override: str = None):
    """
    Predict test set with one or more checkpoints, average probabilities,
    return top-K per class by confidence. Accepts a single Path or list of Paths.
    Returns list of dicts matching synth data format.
    """
    # Normalize to list of exp dirs
    if isinstance(exp_dir, (str, Path)):
        exp_dirs = [Path(exp_dir)]
    else:
        exp_dirs = [Path(d) for d in exp_dir]

    # Read config from first experiment to get subtask/track/labels
    config_path = exp_dirs[0] / "config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    subtask = config["subtask"]
    track = track_override or config.get("track", "base")

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data — don't filter by subtask (test has no labels)
    test_path = workspace / "data" / "touchefallacy_2026_test_task.jsonl"
    test_data = load_jsonl(str(test_path))

    raw_entries = []
    for e in test_data:
        text = build_input_text(e, track, subtask)
        raw_entries.append({"id": e["id"], "text": text, "raw": e})

    # Infer with each model and average probabilities
    avg_probs = np.zeros((len(raw_entries), num_labels), dtype=np.float32)
    for ed in exp_dirs:
        avg_probs += _infer_probs(ed, raw_entries, subtask, track, device)
    avg_probs /= len(exp_dirs)

    # Build argmax predictions from averaged probs
    all_preds = []
    all_probs_list = []
    for idx_val in range(len(raw_entries)):
        prob_vec = avg_probs[idx_val]
        pred_id = int(prob_vec.argmax())
        conf = float(prob_vec[pred_id])
        label = id2label[pred_id]
        all_preds.append((idx_val, pred_id, conf, label))
        all_probs_list.append((idx_val, prob_vec))

    # For each class, rank all examples by probability for that class
    selected = []
    seen_ids = set()  # prevent same example being pseudo-labeled for multiple classes
    print(f"[pseudo] {subtask}/{track}: {len(all_preds)} test predictions (stratified)")
    for cls_id in range(num_labels):
        label = id2label[cls_id]
        # Sort all examples by probability assigned to this class (descending)
        ranked = sorted(all_probs_list, key=lambda x: -x[1][cls_id])
        # Count how many were argmax-predicted as this class
        n_argmax = sum(1 for _, _, _, lbl in all_preds if lbl == label)
        # Take top-K; for well-predicted classes use min_confidence on the
        # class probability; for under-predicted classes (n_argmax < top_k)
        # relax threshold to ensure we get some examples
        if n_argmax >= top_k:
            # Well-predicted: use standard confidence threshold
            effective_min = min_confidence
        else:
            # Under-predicted: take top-K by relative probability even if low
            effective_min = 0.0
        picked = []
        for idx_val, prob_vec in ranked[:top_k * 2]:  # scan wider to fill after dedup
            if idx_val in seen_ids:
                continue
            if prob_vec[cls_id] >= effective_min:
                picked.append((idx_val, float(prob_vec[cls_id])))
            if len(picked) >= top_k:
                break
        print(f"  {label}: {n_argmax} argmax-predicted, "
              f"{len(picked)} selected (top_k={top_k}, min_conf={effective_min:.2f})")

        for idx, conf in picked:
            seen_ids.add(idx)
            entry = raw_entries[idx]
            raw = entry["raw"]
            # Build a synth-compatible record
            source_names = "+".join(d.name for d in exp_dirs)
            record = {
                "id": f"pseudo_{entry['id']}",
                "source": "pseudo_label",
                "source_exp": source_names,
                "confidence": round(conf, 4),
                "subtask": subtask,
            }
            # Copy all raw fields to preserve text_base, text_enhanced, etc.
            for k, v in raw.items():
                if k != "id":
                    record[k] = v

            # Overwrite label fields based on predicted class
            if subtask == "st1":
                record["fallacy_exists"] = 1 if label == "fallacy" else 0
            elif subtask == "st2":
                record["fallacy_exists"] = 1
                record["fallacy_type"] = label
            elif subtask == "st3":
                record["fallacy_exists"] = 0
                goal, basis = label.split("-")
                record["classification"] = {
                    "argument_goal": goal,
                    "argument_basis": basis,
                }

            selected.append(record)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label test data")
    parser.add_argument("--exp_dir", required=True, nargs="+",
                        help="One or more experiment directories (averaged if multiple)")
    parser.add_argument("--top_k", type=int, default=30,
                        help="Max predictions per class to keep")
    parser.add_argument("--min_confidence", type=float, default=0.7,
                        help="Minimum softmax confidence to include")
    parser.add_argument("--track", default=None,
                        help="Override track (base/enhanced)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    args = parser.parse_args()

    exp_dirs = [Path(d).resolve() for d in args.exp_dir]
    output_path = Path(args.output).resolve()

    selected = pseudo_label(exp_dirs, args.top_k, args.min_confidence,
                            args.track)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[pseudo] Wrote {len(selected)} pseudo-labeled examples to {output_path}")


if __name__ == "__main__":
    main()
