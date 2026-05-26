"""
Touché 2026 — Training script with Layerwise Learning Rate Decay (LLRD).

Higher layers get lr=top_lr, lower layers get progressively smaller LR
via decay_factor per layer. This prevents catastrophic forgetting in
lower layers while allowing upper layers to specialize.
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    extract_label,
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
        }


def get_llrd_param_groups(model, top_lr, decay_factor, weight_decay=0.01):
    """Create parameter groups with layerwise learning rate decay."""
    param_groups = []

    # Identify layer indices: embeddings = layer -1, then encoder layers 0..N-1
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"}

    # Classifier head gets top_lr
    classifier_params_decay = []
    classifier_params_no_decay = []
    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name:
            if any(nd in name for nd in no_decay):
                classifier_params_no_decay.append(param)
            else:
                classifier_params_decay.append(param)

    if classifier_params_decay:
        param_groups.append({"params": classifier_params_decay, "lr": top_lr, "weight_decay": weight_decay})
    if classifier_params_no_decay:
        param_groups.append({"params": classifier_params_no_decay, "lr": top_lr, "weight_decay": 0.0})

    # Find number of encoder layers
    n_layers = 0
    for name, _ in model.named_parameters():
        # Match patterns like "layer.23." or "layers.23." or "encoder.layer.23."
        import re
        match = re.search(r'(?:layer|layers)\.(\d+)\.', name)
        if match:
            n_layers = max(n_layers, int(match.group(1)) + 1)

    # Encoder layers: top layer gets top_lr, each lower layer *= decay_factor
    for layer_idx in range(n_layers - 1, -1, -1):
        depth = n_layers - 1 - layer_idx  # 0 for top layer
        layer_lr = top_lr * (decay_factor ** depth)

        layer_decay_params = []
        layer_no_decay_params = []

        for name, param in model.named_parameters():
            if f"layer.{layer_idx}." in name or f"layers.{layer_idx}." in name:
                if "classifier" in name or "score" in name:
                    continue  # already handled
                if any(nd in name for nd in no_decay):
                    layer_no_decay_params.append(param)
                else:
                    layer_decay_params.append(param)

        if layer_decay_params:
            param_groups.append({"params": layer_decay_params, "lr": layer_lr, "weight_decay": weight_decay})
        if layer_no_decay_params:
            param_groups.append({"params": layer_no_decay_params, "lr": layer_lr, "weight_decay": 0.0})

    # Embeddings: lowest LR
    embed_lr = top_lr * (decay_factor ** n_layers)
    embed_decay = []
    embed_no_decay = []
    assigned = set()
    for pg in param_groups:
        for p in pg["params"]:
            assigned.add(id(p))

    for name, param in model.named_parameters():
        if id(param) not in assigned:
            if any(nd in name for nd in no_decay):
                embed_no_decay.append(param)
            else:
                embed_decay.append(param)

    if embed_decay:
        param_groups.append({"params": embed_decay, "lr": embed_lr, "weight_decay": weight_decay})
    if embed_no_decay:
        param_groups.append({"params": embed_no_decay, "lr": embed_lr, "weight_decay": 0.0})

    return param_groups


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    seed = config.get("seed", 42)
    set_seed(seed)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 1800), margin_s=60)

    subtask = config["subtask"]
    track = config["track"]
    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    top_lr = train_cfg.get("top_lr", 1e-5)
    bottom_lr = train_cfg.get("bottom_lr", 3e-6)
    decay_factor = train_cfg.get("llrd_decay", 0.9)
    batch_size = train_cfg.get("batch_size", 8)
    epochs = train_cfg.get("epochs", 12)
    max_len = train_cfg.get("max_len", 384)
    loss_type = train_cfg.get("loss", "ce")
    patience = train_cfg.get("patience", 3)

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    print(f"[llrd] exp={exp_dir.name}, subtask={subtask}, track={track}")
    print(f"[llrd] backbone={backbone}, top_lr={top_lr}, decay={decay_factor}")

    # ── Data (standard pipeline) ──────────────────────────────────
    ws = workspace
    all_data = load_jsonl(str(ws / "data" / "touchefallacy_2026_train.jsonl"))
    train_split, dev_split = make_splits(all_data, subtask)

    train_entries = []
    if config.get("data", {}).get("real", True):
        for e in train_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                train_entries.append({"id": e["id"], "text": text, "label": label,
                                     "label_id": label2id[label]})

    for sv in config.get("data", {}).get("synth_versions", []):
        try:
            for e in load_synth_data(sv, subtask):
                label = extract_label(e, subtask)
                if label and label in label2id:
                    text = build_input_text(e, track, subtask)
                    train_entries.append({"id": e.get("id", f"s_{sv}"), "text": text,
                                         "label": label, "label_id": label2id[label]})
        except FileNotFoundError:
            pass

    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            dev_entries.append({"id": e["id"], "text": text, "label": label,
                               "label_id": label2id[label]})

    print(f"[llrd] train={len(train_entries)}, dev={len(dev_entries)}")

    # ── Model ─────────────────────────────────────────────────────
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=num_labels
    ).to(device)

    # ── LLRD optimizer ────────────────────────────────────────────
    param_groups = get_llrd_param_groups(model, top_lr, decay_factor)
    optimizer = torch.optim.AdamW(param_groups)

    train_ds = TextDataset(train_entries, tokenizer, max_len)
    dev_ds = TextDataset(dev_entries, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False)

    total_steps = len(train_loader) * epochs
    warmup_steps = min(total_steps // 10, 100)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if loss_type == "weighted_ce":
        weights = get_class_weights(train_entries, num_labels)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # ── Training loop (standard) ──────────────────────────────────
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
        epoch_loss = 0.0
        for batch in train_loader:
            if guard.exceeded():
                status_flags.append("time_exceeded")
                break
            out = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device))
            loss = loss_fn(out.logits.float(), batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            global_step += 1
            heartbeat.beat(step=global_step)

        if "time_exceeded" in status_flags:
            break

        avg_loss = epoch_loss / max(len(train_loader), 1)
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device))
                all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(batch["label"].tolist())

        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)
        print(f"[llrd] epoch={epoch}, loss={avg_loss:.4f}, f1={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            patience_counter = 0
            save_checkpoint(model, tokenizer, exp_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status_flags.append("early_stopped")
                break

    # Final eval
    ckpt = exp_dir / "ckpt" / "best"
    if ckpt.exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt), num_labels=num_labels).to(device)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device))
                all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
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
    metrics["best_train_loss"] = round(best_train_loss, 5)
    metrics["llrd_decay"] = decay_factor
    metrics["top_lr"] = top_lr

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[llrd] DONE. F1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        sys.exit(1)
