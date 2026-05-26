"""
Touché 2026 — Training script v4.
Extends v3 with: configurable seed, mixed-precision (fp32/fp16/bf16),
focal loss, and configurable gradient clipping.
"""

import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure shared/ is importable
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


def focal_loss(inputs, targets, gamma=2.0, weight=None):
    """Focal Loss (Lin et al., 2017) with optional class weights."""
    ce = torch.nn.functional.cross_entropy(
        inputs, targets, weight=weight, reduction="none"
    )
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    seed = config.get("seed", 42)
    set_seed(seed)

    # ── Status & guards ───────────────────────────────────────────
    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    time_budget = config.get("time_budget_s", 540)
    guard = WallClockGuard(time_budget, margin_s=60)

    # ── Config ────────────────────────────────────────────────────
    subtask = config["subtask"]
    track = config["track"]
    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 2e-5)
    batch_size = train_cfg.get("batch_size", 16)
    epochs = train_cfg.get("epochs", 10)
    max_len = train_cfg.get("max_len", 256)
    loss_type = train_cfg.get("loss", "ce")
    synth_weight = train_cfg.get("synth_weight", 0.5)
    precision = train_cfg.get("precision", "fp32")
    grad_clip = train_cfg.get("grad_clip", 1.0)

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    print(f"[train] exp_dir={exp_dir.name}, subtask={subtask}, track={track}")
    print(f"[train] backbone={backbone}, lr={lr}, bs={batch_size}, epochs={epochs}")
    print(f"[train] max_len={max_len}, loss={loss_type}, budget={time_budget}s")
    print(f"[train] synth_weight={synth_weight}, precision={precision}, "
          f"grad_clip={grad_clip}, seed={seed}")

    # ── Data ──────────────────────────────────────────────────────
    ws = workspace
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    train_split, dev_split = make_splits(all_data, subtask)

    train_entries = []
    n_real = 0
    n_synth = 0

    if config.get("data", {}).get("real", True):
        for e in train_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                train_entries.append({
                    "id": e["id"], "text": text, "label": label,
                    "label_id": label2id[label],
                    "sample_weight": 1.0,
                })
                n_real += 1

    # Synthetic data — with reduced weight
    synth_versions = config.get("data", {}).get("synth_versions", [])
    for sv in synth_versions:
        try:
            synth_data = load_synth_data(sv, subtask)
            for e in synth_data:
                label = extract_label(e, subtask)
                if label is not None and label in label2id:
                    text = build_input_text(e, track, subtask)
                    train_entries.append({
                        "id": e.get("id", f"synth_{sv}"),
                        "text": text, "label": label,
                        "label_id": label2id[label],
                        "sample_weight": synth_weight,
                    })
                    n_synth += 1
        except FileNotFoundError:
            print(f"[train] WARNING: synth version {sv} not found, skipping")

    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            dev_entries.append({
                "id": e["id"], "text": text, "label": label,
                "label_id": label2id[label],
            })

    print(f"[train] train={len(train_entries)} (real={n_real}, synth={n_synth}), "
          f"dev={len(dev_entries)}")
    train_dist = Counter(e["label"] for e in train_entries)
    dev_dist = Counter(e["label"] for e in dev_entries)
    print(f"[train] train_dist={dict(sorted(train_dist.items()))}")
    print(f"[train] dev_dist={dict(sorted(dev_dist.items()))}")
    heartbeat.beat(step=0)

    # ── Model ─────────────────────────────────────────────────────
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=num_labels
    )
    model.to(device)
    print(f"[train] params={count_params(model):,}, device={device}")

    # ── DataLoaders ───────────────────────────────────────────────
    train_ds = TextDataset(train_entries, tokenizer, max_len)
    dev_ds = TextDataset(dev_entries, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False)

    # ── Loss ──────────────────────────────────────────────────────
    class_weight_tensor = None
    if loss_type in ("weighted_ce", "focal_weighted"):
        weights = get_class_weights(train_entries, num_labels)
        print(f"[train] class_weights={[round(w, 3) for w in weights]}")
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32,
                                           device=device)

    if loss_type in ("focal", "focal_weighted"):
        focal_gamma = train_cfg.get("focal_gamma", 2.0)
        print(f"[train] focal_loss gamma={focal_gamma}")
        cw = class_weight_tensor  # None for "focal", tensor for "focal_weighted"

        def loss_fn(logits, labels):
            return focal_loss(logits, labels, gamma=focal_gamma, weight=cw)
    elif class_weight_tensor is not None:
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weight_tensor, reduction="none"
        )
    elif loss_type == "label_smoothing":
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # ── Optimizer & scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = min(total_steps // 10, 100)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ── AMP setup ─────────────────────────────────────────────────
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_scaler = (precision == "fp16")
    scaler = torch.amp.GradScaler(enabled=True) if use_scaler else None
    if use_amp:
        print(f"[train] AMP enabled: precision={precision}, dtype={amp_dtype}")

    # ── Early stopping ────────────────────────────────────────────
    patience = train_cfg.get("patience", 0)  # 0 = disabled
    if patience > 0:
        print(f"[train] early_stopping: patience={patience}")

    # ── Training loop ─────────────────────────────────────────────
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
            print(f"[train] Wall-clock guard hit at epoch {epoch}")
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

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                    enabled=use_amp):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                logits = outputs.logits
                per_sample_loss = loss_fn(logits, labels)

            # Apply per-sample weights and average (in fp32)
            weighted_loss = (per_sample_loss.float() * sample_weights).mean()

            # NaN safety
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

        # ── Dev evaluation ────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                        enabled=use_amp):
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                preds = outputs.logits.float().argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].tolist())

        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)

        unique_preds = set(y_pred)
        print(f"[train] epoch={epoch}, loss={avg_loss:.4f}, "
              f"dev_f1={metrics['f1_macro']:.4f}, "
              f"unique_preds={len(unique_preds)}/{num_labels}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            no_improve = 0
            save_checkpoint(model, tokenizer, exp_dir)
            print(f"[train] New best dev F1: {best_dev_f1:.4f}, checkpoint saved")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                status_flags.append("early_stopped")
                print(f"[train] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Final metrics ─────────────────────────────────────────────
    metrics = {"f1_macro": 0.0, "accuracy": 0.0, "per_class": {}}
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
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                        enabled=use_amp):
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                preds = outputs.logits.float().argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].tolist())
        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)

    metrics["wall_clock_s"] = round(guard.elapsed(), 1)
    metrics["peak_vram_mb"] = round(get_peak_vram_mb(), 1)
    metrics["params"] = count_params(model)
    metrics["track"] = track
    metrics["backbone"] = backbone
    metrics["status_flags"] = status_flags
    metrics["epochs_completed"] = min(epoch + 1, epochs) if 'epoch' in dir() else 0
    metrics["nan_batches"] = nan_count
    metrics["n_real"] = n_real
    metrics["n_synth"] = n_synth
    metrics["synth_weight"] = synth_weight
    metrics["seed"] = seed
    metrics["precision"] = precision
    metrics["patience"] = patience
    metrics["early_stopped"] = "early_stopped" in status_flags
    final_loss = best_train_loss if math.isfinite(best_train_loss) else -1.0
    metrics["best_train_loss"] = round(final_loss, 5)

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[train] DONE. F1={metrics['f1_macro']:.4f}, "
          f"wall={metrics['wall_clock_s']}s, real={n_real}, synth={n_synth}")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[train] CRASHED: {exc}")
        sys.exit(1)
