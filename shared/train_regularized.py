"""
Touché 2026 — Training with advanced regularization techniques.

Supports (via config):
  - R-Drop: KL-divergence consistency regularization (two forward passes)
  - FGM: Fast Gradient Method adversarial training on word embeddings
  - SWA: Stochastic Weight Averaging over last N checkpoints
  - EMA: Exponential Moving Average of model weights

Each technique is toggled via config.train.regularization = {"type": "rdrop|fgm|swa|ema", ...}
"""

import copy
import json
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import torch
import torch.nn.functional as F
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
            e["text"], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(e["label_id"], dtype=torch.long),
        }


# ── FGM adversarial helper ────────────────────────────────────────
class FGM:
    """Fast Gradient Method: perturb word embeddings by epsilon in gradient direction."""
    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if self.emb_name in name and param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ── EMA helper ────────────────────────────────────────────────────
class EMAModel:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.data.clone()
                      for name, param in model.named_parameters()}

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model):
        """Replace model params with EMA params (for eval)."""
        self.backup = {name: param.data.clone()
                      for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data = self.shadow[name].clone()

    def restore(self, model):
        """Restore original params after eval."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]


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
    lr = train_cfg.get("lr", 1e-5)
    batch_size = train_cfg.get("batch_size", 8)
    epochs = train_cfg.get("epochs", 12)
    max_len = train_cfg.get("max_len", 384)
    loss_type = train_cfg.get("loss", "ce")
    patience = train_cfg.get("patience", 3)

    reg_cfg = train_cfg.get("regularization", {})
    reg_type = reg_cfg.get("type", "none")
    # R-Drop params
    rdrop_alpha = reg_cfg.get("alpha", 1.0)
    # FGM params
    fgm_epsilon = reg_cfg.get("epsilon", 1.0)
    # EMA params
    ema_decay = reg_cfg.get("decay", 0.999)
    # SWA params
    swa_start_epoch = reg_cfg.get("swa_start", 5)

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    print(f"[reg] exp={exp_dir.name}, subtask={subtask}, track={track}")
    print(f"[reg] backbone={backbone}, reg={reg_type}")

    # ── Data ──────────────────────────────────────────────────────
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
                    train_entries.append({"id": e.get("id"), "text": text,
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

    print(f"[reg] train={len(train_entries)}, dev={len(dev_entries)}")

    # ── Model ─────────────────────────────────────────────────────
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=num_labels).to(device)

    train_ds = TextDataset(train_entries, tokenizer, max_len)
    dev_ds = TextDataset(dev_entries, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False)

    if loss_type == "weighted_ce":
        weights = get_class_weights(train_entries, num_labels)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, min(total_steps // 10, 100), total_steps)

    # Initialize regularization helpers
    fgm = FGM(model, epsilon=fgm_epsilon) if reg_type == "fgm" else None
    ema = EMAModel(model, decay=ema_decay) if reg_type == "ema" else None
    swa_state_dicts = [] if reg_type == "swa" else None

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
        epoch_loss = 0.0

        for batch in train_loader:
            if guard.exceeded():
                status_flags.append("time_exceeded")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # ── Forward pass 1 ────────────────────────────────
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits1 = outputs.logits.float()
            loss = loss_fn(logits1, labels)

            # ── R-Drop: second forward pass + KL divergence ───
            if reg_type == "rdrop":
                outputs2 = model(input_ids=input_ids, attention_mask=attention_mask)
                logits2 = outputs2.logits.float()
                loss2 = loss_fn(logits2, labels)
                # Symmetric KL divergence
                p = F.log_softmax(logits1, dim=-1)
                q = F.log_softmax(logits2, dim=-1)
                kl = (F.kl_div(p, q.exp(), reduction="batchmean") +
                      F.kl_div(q, p.exp(), reduction="batchmean")) / 2
                loss = (loss + loss2) / 2 + rdrop_alpha * kl

            loss.backward()

            # ── FGM: adversarial step ─────────────────────────
            if reg_type == "fgm" and fgm is not None:
                fgm.attack()
                adv_out = model(input_ids=input_ids, attention_mask=attention_mask)
                adv_loss = loss_fn(adv_out.logits.float(), labels)
                adv_loss.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # ── EMA: update shadow weights ────────────────────
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            global_step += 1
            heartbeat.beat(step=global_step)

        if "time_exceeded" in status_flags:
            break

        avg_loss = epoch_loss / max(len(train_loader), 1)
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss

        # ── SWA: collect state dict after swa_start_epoch ─────
        if reg_type == "swa" and epoch >= swa_start_epoch:
            swa_state_dicts.append(copy.deepcopy(model.state_dict()))

        # ── Dev evaluation ────────────────────────────────────
        model.eval()
        if ema is not None:
            ema.apply(model)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device))
                all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(batch["label"].tolist())

        if ema is not None:
            ema.restore(model)

        y_true = [id2label[l] for l in all_labels]
        y_pred = [id2label[p] for p in all_preds]
        metrics = evaluate(y_true, y_pred, subtask)
        print(f"[reg] epoch={epoch}, loss={avg_loss:.4f}, f1={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            patience_counter = 0
            save_checkpoint(model, tokenizer, exp_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status_flags.append("early_stopped")
                break

    # ── SWA: average collected checkpoints ────────────────────────
    if reg_type == "swa" and swa_state_dicts:
        print(f"[reg] SWA: averaging {len(swa_state_dicts)} checkpoints")
        avg_state = {}
        for key in swa_state_dicts[0]:
            avg_state[key] = torch.stack([sd[key].float() for sd in swa_state_dicts]).mean(dim=0)
        model.load_state_dict(avg_state)
        save_checkpoint(model, tokenizer, exp_dir, label="swa")

        # Evaluate SWA model
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
        print(f"[reg] SWA final F1={metrics['f1_macro']:.4f}")
        status_flags.append(f"swa_{len(swa_state_dicts)}_ckpts")
    else:
        # Final eval with best checkpoint
        ckpt = exp_dir / "ckpt" / "best"
        if ckpt.exists():
            model = AutoModelForSequenceClassification.from_pretrained(
                str(ckpt), num_labels=num_labels).to(device)
            if ema is not None:
                ema.apply(model)
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
    metrics["regularization"] = reg_type
    metrics["status_flags"] = status_flags
    metrics["best_train_loss"] = round(best_train_loss, 5)

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[reg] DONE. F1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        sys.exit(1)
