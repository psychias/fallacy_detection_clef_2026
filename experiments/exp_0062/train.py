"""
R1b — Leakage diagnostic: train ST2 enhanced + synth, but regex-mask
fallacy-naming words in text_enhanced to test if the enhanced track
performance is due to leakage of label information into the input text.

This is train_v4.py with a text-masking preprocessing step.
If F1 drops from ~0.937 to ~0.75, enhanced track is leakage.
If F1 holds at 0.92+, it's real signal.
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    extract_label,
    filter_for_subtask,
    get_class_weights,
    load_jsonl,
    load_synth_data,
    make_splits,
)
from shared.eval import evaluate, write_metrics
from shared.train_utils import (
    Heartbeat,
    WallClockGuard,
    count_params,
    get_device,
    get_peak_vram_mb,
    load_config,
    save_checkpoint,
    set_seed,
    write_status,
    write_traceback,
)

# ── Leakage mask ──────────────────────────────────────────────────
LEAKY_PATTERNS = [
    r"\bfallac(?:y|ies|ious)\b",
    r"\bappeal\s+to\b",
    r"\bhasty\b",
    r"\bslippery\b",
    r"\btradition(?:al)?\b",
    r"\bnaturalistic\b",
    r"\bpopular(?:ity)?\b",
    r"\bargumentum\b",
    r"\bad\s+(?:hominem|populum|verecundiam|baculum|antiquitatem|naturam)\b",
    r"\bblack[\s-]*(?:and[\s-]*)?white\b",
    r"\bfalse\s+dilemma\b",
    r"\bfalse\s+dichotomy\b",
    r"\bworse\s+problems?\b",
    r"\bred\s+herring\b",
    r"\bstraw\s*man\b",
    r"\bbandwagon\b",
    r"\bauthorit(?:y|arian|ative)\b",
]
LEAKY_RE = re.compile("|".join(LEAKY_PATTERNS), flags=re.IGNORECASE)
MASK_TOKEN = "[MASKED]"


def mask_leaky_words(text: str) -> str:
    """Replace fallacy-naming words with [MASKED]."""
    return LEAKY_RE.sub(MASK_TOKEN, text)


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
            "sample_weight": torch.tensor(e.get("sample_weight", 1.0),
                                          dtype=torch.float32),
        }


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    seed = config.get("seed", 42)
    set_seed(seed)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    time_budget = config.get("time_budget_s", 1500)
    guard = WallClockGuard(time_budget, margin_s=60)

    subtask = config["subtask"]
    track = config["track"]
    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-5)
    batch_size = train_cfg.get("batch_size", 8)
    epochs = train_cfg.get("epochs", 10)
    max_len = train_cfg.get("max_len", 256)
    loss_type = train_cfg.get("loss", "weighted_ce")
    synth_weight = train_cfg.get("synth_weight", 0.5)
    precision = train_cfg.get("precision", "fp32")
    grad_clip = train_cfg.get("grad_clip", 1.0)
    patience = train_cfg.get("patience", 0)

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    print(f"[leakage] exp_dir={exp_dir.name}, subtask={subtask}, track={track}")
    print(f"[leakage] MASKING fallacy-naming words in text")
    print(f"[leakage] backbone={backbone}, lr={lr}, epochs={epochs}")

    # ── Data ──────────────────────────────────────────────────────
    ws = workspace
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    train_split, dev_split = make_splits(all_data, subtask)

    mask_count = 0
    train_entries = []
    n_real = 0
    n_synth = 0
    if config.get("data", {}).get("real", True):
        for e in train_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                masked = mask_leaky_words(text)
                if masked != text:
                    mask_count += 1
                train_entries.append({
                    "id": e["id"], "text": masked, "label": label,
                    "label_id": label2id[label], "sample_weight": 1.0,
                })
                n_real += 1

    synth_versions = config.get("data", {}).get("synth_versions", [])
    for sv in synth_versions:
        try:
            synth_data = load_synth_data(sv, subtask)
            for e in synth_data:
                label = extract_label(e, subtask)
                if label is not None and label in label2id:
                    text = build_input_text(e, track, subtask)
                    masked = mask_leaky_words(text)
                    if masked != text:
                        mask_count += 1
                    train_entries.append({
                        "id": e.get("id", f"synth_{sv}"),
                        "text": masked, "label": label,
                        "label_id": label2id[label],
                        "sample_weight": synth_weight,
                    })
                    n_synth += 1
        except FileNotFoundError:
            print(f"[leakage] WARNING: synth version {sv} not found, skipping")

    dev_entries = []
    dev_mask_count = 0
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            masked = mask_leaky_words(text)
            if masked != text:
                dev_mask_count += 1
            dev_entries.append({
                "id": e["id"], "text": masked, "label": label,
                "label_id": label2id[label],
            })

    print(f"[leakage] train={len(train_entries)} (real={n_real}, synth={n_synth}), "
          f"dev={len(dev_entries)}")
    print(f"[leakage] masked entries: train={mask_count}, dev={dev_mask_count}")
    heartbeat.beat(step=0)

    # ── Model ─────────────────────────────────────────────────────
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=num_labels
    ).to(device)
    print(f"[leakage] params={count_params(model):,}, device={device}")

    train_ds = TextDataset(train_entries, tokenizer, max_len)
    dev_ds = TextDataset(dev_entries, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False)

    # ── Loss ──────────────────────────────────────────────────────
    class_weight_tensor = None
    if loss_type in ("weighted_ce", "focal_weighted"):
        weights = get_class_weights(train_entries, num_labels)
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    if class_weight_tensor is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor, reduction="none")
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = min(total_steps // 10, 100)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_scaler = (precision == "fp16")
    scaler = torch.amp.GradScaler(enabled=True) if use_scaler else None

    if patience > 0:
        print(f"[leakage] early_stopping: patience={patience}")

    best_dev_f1 = -1.0
    best_train_loss = float("inf")
    time_exceeded = False
    status_flags = []
    global_step = 0
    nan_count = 0
    no_improve = 0

    for epoch in range(epochs):
        if guard.exceeded():
            time_exceeded = True
            status_flags.append("time_exceeded")
            break

        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in train_loader:
            if guard.exceeded():
                time_exceeded = True
                status_flags.append("time_exceeded")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            sample_weights = batch["sample_weight"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                per_sample_loss = loss_fn(logits, labels)

            weighted_loss = (per_sample_loss.float() * sample_weights).mean()

            if not math.isfinite(weighted_loss.item()):
                nan_count += 1
                optimizer.zero_grad()
                if use_scaler:
                    scaler.update()
                if nan_count > 50:
                    status_flags.append("excessive_nan")
                    time_exceeded = True
                    break
                global_step += 1
                continue

            if use_scaler:
                scaler.scale(weighted_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += weighted_loss.item()
            epoch_batches += 1
            global_step += 1
            heartbeat.beat(step=global_step)

        if time_exceeded:
            break

        avg_loss = epoch_loss / max(epoch_batches, 1)
        if math.isfinite(avg_loss) and avg_loss < best_train_loss:
            best_train_loss = avg_loss

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.float().argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].tolist())

        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)
        print(f"[leakage] epoch={epoch}, loss={avg_loss:.4f}, "
              f"dev_f1={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            no_improve = 0
            save_checkpoint(model, tokenizer, exp_dir)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                status_flags.append("early_stopped")
                print(f"[leakage] Early stopping at epoch {epoch}")
                break

    # ── Final eval ────────────────────────────────────────────────
    metrics = {"f1_macro": 0.0}
    best_ckpt = exp_dir / "ckpt" / "best"
    if best_ckpt.exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            str(best_ckpt), num_labels=num_labels
        ).to(device)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.float().argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].tolist())
        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)

    metrics["wall_clock_s"] = round(guard.elapsed(), 1)
    metrics["peak_vram_mb"] = round(get_peak_vram_mb(), 1)
    metrics["params"] = count_params(model)
    metrics["status_flags"] = status_flags
    metrics["epochs_completed"] = min(epoch + 1, epochs) if 'epoch' in dir() else 0
    metrics["nan_batches"] = nan_count
    metrics["n_real"] = n_real
    metrics["n_synth"] = n_synth
    metrics["mask_count_train"] = mask_count
    metrics["mask_count_dev"] = dev_mask_count
    metrics["early_stopped"] = "early_stopped" in status_flags
    final_loss = best_train_loss if math.isfinite(best_train_loss) else -1.0
    metrics["best_train_loss"] = round(final_loss, 5)

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[leakage] DONE. F1={metrics['f1_macro']:.4f}, "
          f"masked_train={mask_count}, masked_dev={dev_mask_count}")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[leakage] CRASHED: {exc}")
        sys.exit(1)
