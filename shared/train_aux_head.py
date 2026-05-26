"""
Touché 2026 — Training script with auxiliary classification head.

Adds a secondary classification head for resembles_fallacy (8-way)
alongside the main subtask head. Joint loss = main_loss + aux_weight * aux_loss.
The auxiliary signal injects external-basis knowledge that helps with
practical-external (pe) classification.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
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

# ── resembles_fallacy labels ─────────────────────────────────────
RESEMBLES_LABELS = [
    "authority", "black-white", "hasty_generalization", "natural",
    "population", "slippery_slope", "tradition", "worse_problems",
]
RESEMBLES_LABEL2ID = {l: i for i, l in enumerate(RESEMBLES_LABELS)}
RESEMBLES_ID2LABEL = {v: k for k, v in RESEMBLES_LABEL2ID.items()}
NO_RESEMBLES = -100  # ignore index for entries without resembles_fallacy


class DualHeadModel(nn.Module):
    """Transformer backbone with main + auxiliary classification heads."""

    def __init__(self, backbone_name, num_main_labels, num_aux_labels, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.main_head = nn.Linear(hidden, num_main_labels)
        self.aux_head = nn.Linear(hidden, num_aux_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        main_logits = self.main_head(cls)
        aux_logits = self.aux_head(cls)
        return main_logits, aux_logits

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save({
            "main_head": self.main_head.state_dict(),
            "aux_head": self.aux_head.state_dict(),
        }, os.path.join(path, "heads.pt"))

    @classmethod
    def load_pretrained(cls, path, num_main_labels, num_aux_labels):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.backbone = AutoModel.from_pretrained(path)
        hidden = model.backbone.config.hidden_size
        model.dropout = nn.Dropout(0.1)
        model.main_head = nn.Linear(hidden, num_main_labels)
        model.aux_head = nn.Linear(hidden, num_aux_labels)
        heads = torch.load(os.path.join(path, "heads.pt"), map_location="cpu")
        model.main_head.load_state_dict(heads["main_head"])
        model.aux_head.load_state_dict(heads["aux_head"])
        return model


class AuxDataset(Dataset):
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
            "aux_label": torch.tensor(e.get("aux_label_id", NO_RESEMBLES), dtype=torch.long),
        }


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    seed = config.get("seed", 42)
    set_seed(seed)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    time_budget = config.get("time_budget_s", 1800)
    guard = WallClockGuard(time_budget, margin_s=60)

    subtask = config["subtask"]
    track = config["track"]
    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-5)
    batch_size = train_cfg.get("batch_size", 8)
    epochs = train_cfg.get("epochs", 15)
    max_len = train_cfg.get("max_len", 384)
    loss_type = train_cfg.get("loss", "weighted_ce")
    patience = train_cfg.get("patience", 3)
    aux_weight = train_cfg.get("aux_weight", 0.3)
    synth_weight = train_cfg.get("synth_weight", 0.3)

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)
    num_aux = len(RESEMBLES_LABELS)

    print(f"[aux_head] exp={exp_dir.name}, subtask={subtask}, track={track}")
    print(f"[aux_head] backbone={backbone}, lr={lr}, aux_weight={aux_weight}")

    # ── Data ──────────────────────────────────────────────────────
    ws = workspace
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    train_split, dev_split = make_splits(all_data, subtask)

    train_entries = []
    n_with_aux = 0
    if config.get("data", {}).get("real", True):
        for e in train_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                # Extract resembles_fallacy if available
                resembles = e.get("resembles_fallacy")
                aux_id = NO_RESEMBLES
                if resembles and resembles in RESEMBLES_LABEL2ID:
                    aux_id = RESEMBLES_LABEL2ID[resembles]
                    n_with_aux += 1
                train_entries.append({
                    "id": e["id"], "text": text, "label": label,
                    "label_id": label2id[label],
                    "aux_label_id": aux_id,
                })

    # Synthetic data (no aux labels)
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
                        "aux_label_id": NO_RESEMBLES,
                    })
        except FileNotFoundError:
            print(f"[aux_head] WARNING: synth {sv} not found")

    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            resembles = e.get("resembles_fallacy")
            aux_id = NO_RESEMBLES
            if resembles and resembles in RESEMBLES_LABEL2ID:
                aux_id = RESEMBLES_LABEL2ID[resembles]
            dev_entries.append({
                "id": e["id"], "text": text, "label": label,
                "label_id": label2id[label],
                "aux_label_id": aux_id,
            })

    print(f"[aux_head] train={len(train_entries)} ({n_with_aux} with aux), dev={len(dev_entries)}")

    # ── Model ─────────────────────────────────────────────────────
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = DualHeadModel(backbone, num_labels, num_aux).to(device)
    print(f"[aux_head] params={count_params(model):,}")

    # ── DataLoaders ───────────────────────────────────────────────
    train_ds = AuxDataset(train_entries, tokenizer, max_len)
    dev_ds = AuxDataset(dev_entries, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False)

    # ── Loss functions ────────────────────────────────────────────
    if loss_type == "weighted_ce":
        weights = get_class_weights(train_entries, num_labels)
        main_loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    else:
        main_loss_fn = nn.CrossEntropyLoss()

    aux_loss_fn = nn.CrossEntropyLoss(ignore_index=NO_RESEMBLES)

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
    patience_counter = 0
    global_step = 0
    status_flags = []

    for epoch in range(epochs):
        if guard.exceeded():
            status_flags.append("time_exceeded")
            break

        model.train()
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0

        for batch in train_loader:
            if guard.exceeded():
                status_flags.append("time_exceeded")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            aux_labels = batch["aux_label"].to(device)

            main_logits, aux_logits = model(input_ids, attention_mask)

            m_loss = main_loss_fn(main_logits.float(), labels)
            a_loss = aux_loss_fn(aux_logits.float(), aux_labels)

            loss = m_loss + aux_weight * a_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_main_loss += m_loss.item()
            epoch_aux_loss += a_loss.item()
            global_step += 1
            heartbeat.beat(step=global_step)

        if "time_exceeded" in status_flags:
            break

        avg_main = epoch_main_loss / max(len(train_loader), 1)
        avg_aux = epoch_aux_loss / max(len(train_loader), 1)
        total_loss = avg_main + aux_weight * avg_aux
        if total_loss < best_train_loss:
            best_train_loss = total_loss

        # ── Dev evaluation (main task only) ───────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                main_logits, _ = model(input_ids, attention_mask)
                preds = main_logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].tolist())

        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)

        print(f"[aux_head] epoch={epoch}, main_loss={avg_main:.4f}, "
              f"aux_loss={avg_aux:.4f}, dev_f1={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            patience_counter = 0
            # Save checkpoint
            ckpt_dir = exp_dir / "ckpt" / "best"
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"[aux_head] New best F1: {best_dev_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status_flags.append("early_stopped")
                break

    # ── Final eval with best checkpoint ───────────────────────────
    ckpt_dir = exp_dir / "ckpt" / "best"
    if ckpt_dir.exists() and (ckpt_dir / "heads.pt").exists():
        model = DualHeadModel.load_pretrained(
            str(ckpt_dir), num_labels, num_aux
        ).to(device)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                main_logits, _ = model(input_ids, attention_mask)
                preds = main_logits.argmax(dim=-1).cpu().tolist()
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
    metrics["aux_weight"] = aux_weight
    metrics["status_flags"] = status_flags
    metrics["best_train_loss"] = round(best_train_loss, 5)

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[aux_head] DONE. F1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[aux_head] CRASHED: {exc}")
        sys.exit(1)
