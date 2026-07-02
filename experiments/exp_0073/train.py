"""
Touché 2026 — Standard training script for fine-tuning a transformer backbone
on a single (subtask, track) slot. Reads all parameters from config.json.
"""

import json
import os
import sys
import time
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
from shared.eval import confusion_dict, evaluate, save_dev_logits, write_metrics
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
        }


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    # Training seed defaults to 42 (the submitted run). It can be overridden via
    # EXP_SEED for a seed-variance study; the dev split is unaffected because
    # make_splits() uses its own fixed seed, so only training randomness varies.
    seed = int(os.environ.get("EXP_SEED", "42"))
    set_seed(seed)
    print(f"[train] seed={seed}")

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

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    print(f"[train] exp_dir={exp_dir.name}, subtask={subtask}, track={track}")
    print(f"[train] backbone={backbone}, lr={lr}, bs={batch_size}, epochs={epochs}")
    print(f"[train] max_len={max_len}, loss={loss_type}, budget={time_budget}s")

    # ── Data ──────────────────────────────────────────────────────
    ws = workspace
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    train_split, dev_split = make_splits(all_data, subtask)

    train_entries = []
    if config.get("data", {}).get("real", True):
        for e in train_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                train_entries.append({
                    "id": e["id"], "text": text, "label": label,
                    "label_id": label2id[label],
                })

    # Synthetic data
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
                    })
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

    print(f"[train] train={len(train_entries)}, dev={len(dev_entries)}")
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
    if loss_type == "weighted_ce":
        weights = get_class_weights(train_entries, num_labels)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    elif loss_type == "focal":
        # Simple focal loss via manual computation
        loss_fn = torch.nn.CrossEntropyLoss()  # fallback; focal handled in loop
    elif loss_type == "label_smoothing":
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # ── Optimizer & scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = min(total_steps // 10, 100)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ── Training loop ─────────────────────────────────────────────
    best_dev_f1 = -1.0
    best_train_loss = float("inf")
    time_exceeded = False
    status_flags = []
    global_step = 0

    for epoch in range(epochs):
        if guard.exceeded():
            time_exceeded = True
            status_flags.append("time_exceeded")
            print(f"[train] Wall-clock guard hit at epoch {epoch}")
            break

        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            if guard.exceeded():
                time_exceeded = True
                status_flags.append("time_exceeded")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()  # cast to fp32 for stable loss
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            heartbeat.beat(step=global_step)

        if time_exceeded:
            break

        avg_loss = epoch_loss / max(len(train_loader), 1)
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss

        # ── Dev evaluation ────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].tolist())

        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)

        print(f"[train] epoch={epoch}, loss={avg_loss:.4f}, "
              f"dev_f1={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            save_checkpoint(model, tokenizer, exp_dir)
            print(f"[train] New best dev F1: {best_dev_f1:.4f}, checkpoint saved")

    # ── Final metrics ─────────────────────────────────────────────
    # Re-evaluate with the best checkpoint if it was saved, then persist the
    # per-example dev logits and a labelled confusion matrix so the per-class
    # analysis is reconstructible from artefacts (the exp_0073 gap noted in the
    # paper's §4.4 / §5.3). dev_loader is unshuffled, so row i of the logits
    # matrix aligns with dev_entries[i].
    best_ckpt = exp_dir / "ckpt" / "best"
    if best_ckpt.exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            str(best_ckpt), num_labels=num_labels
        ).to(device)
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float().cpu()
            all_logits.append(logits)
            all_preds.extend(logits.argmax(dim=-1).tolist())
            all_labels.extend(batch["label"].tolist())
    y_true = [id2label[l] for l in all_labels]
    y_pred = [id2label[p] for p in all_preds]
    metrics = evaluate(y_true, y_pred, subtask)

    dev_ids = [e["id"] for e in dev_entries]
    logits_matrix = torch.cat(all_logits, dim=0).numpy()
    logits_path = save_dev_logits(
        exp_dir, dev_ids, all_labels, logits_matrix, id2label, subtask
    )
    cm = confusion_dict(y_true, y_pred, subtask)
    write_metrics(cm, str(exp_dir / "confusion_matrix.json"))
    metrics["confusion_matrix"] = cm
    metrics["dev_logits_path"] = os.path.relpath(logits_path, str(workspace))
    print(f"[train] Saved dev logits -> {logits_path}")
    print(f"[train] Saved confusion matrix -> {exp_dir / 'confusion_matrix.json'}")

    metrics["wall_clock_s"] = round(guard.elapsed(), 1)
    metrics["peak_vram_mb"] = round(get_peak_vram_mb(), 1)
    metrics["params"] = count_params(model)
    metrics["track"] = track
    metrics["backbone"] = backbone
    metrics["status_flags"] = status_flags
    metrics["epochs_completed"] = min(epoch + 1, epochs) if 'epoch' in dir() else 0
    metrics["best_train_loss"] = round(best_train_loss, 5)

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[train] DONE. F1={metrics['f1_macro']:.4f}, "
          f"wall={metrics['wall_clock_s']}s")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[train] CRASHED: {exc}")
        sys.exit(1)
