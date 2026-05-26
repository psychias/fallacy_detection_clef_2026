"""
Touché 2026 — Ensemble evaluation: soft voting across multiple checkpoints.
Loads models from parent experiments' checkpoints, runs dev inference, averages
softmaxed logits across members, then argmaxes.

Soft voting is the default; it preserves calibration info and avoids the silent
alphabetical-tiebreak bias that hard majority vote has on 4/8-way tasks.
Set `ensemble_method: "majority"` in config.json to fall back to hard voting.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
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
    get_device,
    get_peak_vram_mb,
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
        }


def get_member_probs(model, tokenizer, entries, device,
                     max_len=256, batch_size=16) -> np.ndarray:
    """Run inference and return softmaxed probabilities, shape (N, num_labels).
    This is the soft-voting primitive — we want probs, not preds, so we can
    average across members before the argmax."""
    ds = TextDataset(entries, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # softmax in float32 for numerical stability
            probs = torch.softmax(outputs.logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)

    subtask = config["subtask"]
    track = config["track"]
    max_len = config.get("max_len", 256)
    batch_size = config.get("batch_size", 16)
    member_exp_ids = config["ensemble_members"]
    method = config.get("ensemble_method", "soft")  # "soft" or "majority"

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    print(f"[ensemble] subtask={subtask}, track={track}, method={method}")
    print(f"[ensemble] members={member_exp_ids}")

    # ── Dev set ───────────────────────────────────────────────────
    train_path = workspace / "data" / "touchefallacy_2026_train.jsonl"
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

    print(f"[ensemble] dev={len(dev_entries)}")
    heartbeat.beat(step=0)

    device = get_device()
    all_member_probs = []         # list of (N, C) arrays
    successful_member_ids = []    # parallel list — only members we actually loaded

    # ── Load each member and get probabilities ────────────────────
    for i, member_id in enumerate(member_exp_ids):
        ckpt_path = workspace / "experiments" / member_id / "ckpt" / "best"
        if not ckpt_path.exists():
            print(f"[ensemble] WARNING: checkpoint not found for {member_id}, skipping")
            continue

        print(f"[ensemble] Loading {member_id} from {ckpt_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_path), num_labels=num_labels
        ).to(device)

        probs = get_member_probs(model, tokenizer, dev_entries, device,
                                  max_len=max_len, batch_size=batch_size)
        member_preds = probs.argmax(axis=1).tolist()
        all_member_probs.append(probs)
        successful_member_ids.append(member_id)
        print(f"[ensemble] {member_id}: {len(set(member_preds))} unique predictions")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        heartbeat.beat(step=i + 1)

    if not all_member_probs:
        raise RuntimeError("No valid member predictions!")

    # ── Aggregate ─────────────────────────────────────────────────
    # Stack to (M, N, C) then average over M members
    stacked = np.stack(all_member_probs, axis=0)
    n_samples = stacked.shape[1]

    if method == "soft":
        avg_probs = stacked.mean(axis=0)   # (N, C)
        final_preds = avg_probs.argmax(axis=1).tolist()
    elif method == "majority":
        # Hard vote with deterministic tiebreak by averaged prob (not alphabetical)
        per_member_preds = stacked.argmax(axis=2)  # (M, N)
        avg_probs = stacked.mean(axis=0)            # (N, C) — used only for ties
        final_preds = []
        for j in range(n_samples):
            votes = per_member_preds[:, j].tolist()
            counter = Counter(votes)
            top_count = counter.most_common(1)[0][1]
            tied = [c for c, n in counter.items() if n == top_count]
            if len(tied) == 1:
                final_preds.append(tied[0])
            else:
                # Break ties by highest mean probability among tied classes
                best = max(tied, key=lambda c: avg_probs[j, c])
                final_preds.append(int(best))
    else:
        raise ValueError(f"Unknown ensemble_method: {method}. Use 'soft' or 'majority'.")

    y_true = [e["label"] for e in dev_entries]
    y_pred = [id2label[p] for p in final_preds]
    metrics = evaluate(y_true, y_pred, subtask)

    # ── Per-member F1 for comparison ──────────────────────────────
    per_member = {}
    for i, member_id in enumerate(successful_member_ids):
        mp_preds = stacked[i].argmax(axis=1).tolist()
        mp_labels = [id2label[p] for p in mp_preds]
        m_metrics = evaluate(y_true, mp_labels, subtask)
        per_member[member_id] = m_metrics["f1_macro"]

    # ── Diversity diagnostic: pairwise prediction agreement ───────
    # Useful sanity check — if all members agree on >95% of dev, the ensemble
    # is degenerate and you'd save time using a single member.
    pairwise_agreement = []
    if len(all_member_probs) >= 2:
        per_member_preds = stacked.argmax(axis=2)
        for i in range(len(all_member_probs)):
            for j in range(i + 1, len(all_member_probs)):
                agree = (per_member_preds[i] == per_member_preds[j]).mean()
                pairwise_agreement.append(round(float(agree), 4))

    metrics["ensemble_members"] = successful_member_ids
    metrics["per_member_f1"] = per_member
    metrics["ensemble_method"] = method
    metrics["pairwise_agreement"] = pairwise_agreement
    metrics["mean_pairwise_agreement"] = (
        round(float(np.mean(pairwise_agreement)), 4) if pairwise_agreement else None
    )
    metrics["peak_vram_mb"] = round(get_peak_vram_mb(), 1)
    metrics["track"] = track
    metrics["num_members"] = len(all_member_probs)

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[ensemble] DONE. Ensemble F1={metrics['f1_macro']:.4f} ({method})")
    print(f"[ensemble] Per-member: {per_member}")
    if pairwise_agreement:
        print(f"[ensemble] Pairwise agreement: {pairwise_agreement} "
              f"(mean={metrics['mean_pairwise_agreement']})")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[ensemble] CRASHED: {exc}")
        sys.exit(1)
