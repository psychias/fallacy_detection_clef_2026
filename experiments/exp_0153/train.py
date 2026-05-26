"""
exp_0153 — 5-fold CV baseline for 0073 (ST3 enhanced, no pe synth).
Trains on each fold's train_ids, evaluates on fold's dev_ids.
Reports per-fold pe F1 and macro F1.
"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

from shared.data_utils import (
    LABEL_MAPS, build_input_text, extract_label,
    load_jsonl, load_synth_data,
)
from shared.eval import evaluate, write_metrics
from shared.train_utils import (
    Heartbeat, WallClockGuard, get_device, get_peak_vram_mb,
    load_config, set_seed, write_status, write_traceback,
)

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class TextDataset(Dataset):
    def __init__(self, texts, label_ids, weights, tokenizer, max_len):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_len, return_tensors="pt")
        self.label_ids = torch.tensor(label_ids, dtype=torch.long)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.label_ids[idx]
        item["weight"] = self.weights[idx]
        return item


def train_one_fold(fold_idx, train_ids_set, dev_ids_set, all_data, config, device):
    """Train and evaluate on one fold. Returns per-class F1 dict + macro F1."""
    subtask = config["subtask"]
    track = config["track"]
    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-5)
    batch_size = train_cfg.get("batch_size", 8)
    epochs = train_cfg.get("epochs", 15)
    max_len = train_cfg.get("max_len", 384)
    loss_type = train_cfg.get("loss", "weighted_ce")
    synth_weight = train_cfg.get("synth_weight", 0.5)
    patience = train_cfg.get("patience", 3)

    label2id, id2label = LABEL_MAPS[subtask]
    num_labels = len(label2id)

    # Build train/dev entries from fold IDs
    train_entries = []
    dev_entries = []
    for e in all_data:
        eid = e.get("id", "")
        label = extract_label(e, subtask)
        if label is None or label not in label2id:
            continue
        text = build_input_text(e, track, subtask)
        entry = {"text": text, "label": label, "label_id": label2id[label], "weight": 1.0}
        if eid in train_ids_set:
            train_entries.append(entry)
        elif eid in dev_ids_set:
            dev_entries.append(entry)

    # Add synth data to train (not fold-specific)
    synth_versions = config.get("data", {}).get("synth_versions", [])
    for sv in synth_versions:
        try:
            synth_data = load_synth_data(sv, subtask)
            for e in synth_data:
                label = extract_label(e, subtask)
                if label is not None and label in label2id:
                    text = build_input_text(e, track, subtask)
                    train_entries.append({
                        "text": text, "label": label,
                        "label_id": label2id[label], "weight": synth_weight,
                    })
        except FileNotFoundError:
            pass

    print(f"  Fold {fold_idx}: train={len(train_entries)} dev={len(dev_entries)}")
    dev_labels_dist = Counter(e["label"] for e in dev_entries)
    print(f"  Dev dist: {dict(dev_labels_dist)}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    train_ds = TextDataset(
        [e["text"] for e in train_entries],
        [e["label_id"] for e in train_entries],
        [e["weight"] for e in train_entries],
        tokenizer, max_len
    )
    dev_ds = TextDataset(
        [e["text"] for e in dev_entries],
        [e["label_id"] for e in dev_entries],
        [1.0] * len(dev_entries),
        tokenizer, max_len
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=num_labels)
    model.to(device)

    # Loss weights
    if loss_type == "weighted_ce":
        label_counts = Counter(e["label_id"] for e in train_entries if e["weight"] >= 1.0)
        total = sum(label_counts.values())
        class_weights = torch.tensor(
            [total / (num_labels * max(label_counts.get(i, 1), 1)) for i in range(num_labels)],
            dtype=torch.float32
        ).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_f1 = -1
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weight"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss = (loss * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Eval on dev
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)

        pred_labels = [id2label[p] for p in all_preds]
        true_labels = [e["label"] for e in dev_entries]
        metrics = evaluate(true_labels, pred_labels, subtask=subtask)

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Fold {fold_idx} early stopped at epoch {epoch+1}")
            break

    # Final eval with best state
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    all_logits = []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_logits.append(outputs.logits.cpu().numpy())

    pred_labels = [id2label[p] for p in all_preds]
    true_labels = [e["label"] for e in dev_entries]
    metrics = evaluate(true_labels, pred_labels, subtask=subtask)

    # PE confidence analysis
    all_logits = np.concatenate(all_logits, axis=0)
    probs = np.exp(all_logits - all_logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    labels_list = list(label2id.keys())
    pe_idx = labels_list.index("practical-external")
    pe_conf_on_pe = []
    pe_rank_on_pe = []
    for i, (true_l, entry) in enumerate(zip(true_labels, dev_entries)):
        if true_l == "practical-external":
            pe_prob = float(probs[i, pe_idx])
            rank = int((probs[i] > pe_prob).sum()) + 1
            pe_conf_on_pe.append(pe_prob)
            pe_rank_on_pe.append(rank)

    print(f"  Fold {fold_idx} result: F1={metrics['f1_macro']:.4f}  "
          f"pe_F1={metrics['f1_per_class'].get('practical-external', 0):.4f}  "
          f"pe_conf={pe_conf_on_pe}  pe_rank={pe_rank_on_pe}")

    # Cleanup
    del model, best_state
    torch.cuda.empty_cache()

    return {
        "fold": fold_idx,
        "f1_macro": metrics["f1_macro"],
        "f1_per_class": metrics["f1_per_class"],
        "pe_confidence": pe_conf_on_pe,
        "pe_rank": pe_rank_on_pe,
        "n_train": len(train_entries),
        "n_dev": len(dev_entries),
        "dev_pe_count": dev_labels_dist.get("practical-external", 0),
    }


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(config.get("seed", 42))
    device = get_device()

    write_status(exp_dir, "running")
    hb = Heartbeat(exp_dir)
    guard = WallClockGuard(config.get("time_budget_s", 3600))

    subtask = config["subtask"]

    # Load data and kfold splits
    ws = workspace
    all_data = load_jsonl(str(ws / "data" / "touchefallacy_2026_train.jsonl"))
    kfold_data = json.load(open(ws / "shared" / "kfold_splits_st3.json"))
    folds = kfold_data["folds"]

    print(f"Running 5-fold CV: subtask={subtask}, {len(folds)} folds")

    fold_results = []
    for fold in folds:
        if guard.exceeded():
            print("Time budget exceeded, stopping")
            break
        train_ids_set = set(fold["train_ids"])
        dev_ids_set = set(fold["dev_ids"])
        result = train_one_fold(fold["fold"], train_ids_set, dev_ids_set,
                                all_data, config, device)
        fold_results.append(result)
        hb.beat()

    # Aggregate
    macro_f1s = [r["f1_macro"] for r in fold_results]
    pe_f1s = [r["f1_per_class"].get("practical-external", 0) for r in fold_results]
    all_pe_confs = [c for r in fold_results for c in r["pe_confidence"]]
    all_pe_ranks = [rank for r in fold_results for rank in r["pe_rank"]]

    print(f"\n=== 5-FOLD CV RESULTS ===")
    print(f"Macro F1: {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}  per-fold: {[round(f,4) for f in macro_f1s]}")
    print(f"PE F1:    {np.mean(pe_f1s):.4f} +/- {np.std(pe_f1s):.4f}  per-fold: {[round(f,4) for f in pe_f1s]}")
    print(f"PE confidence on actual pe examples: {[round(c,4) for c in all_pe_confs]}")
    print(f"PE rank on actual pe examples: {all_pe_ranks}")

    final = {
        "subtask": subtask,
        "n_folds": len(fold_results),
        "macro_f1_mean": round(float(np.mean(macro_f1s)), 5),
        "macro_f1_std": round(float(np.std(macro_f1s)), 5),
        "macro_f1_per_fold": [round(f, 5) for f in macro_f1s],
        "pe_f1_mean": round(float(np.mean(pe_f1s)), 5),
        "pe_f1_std": round(float(np.std(pe_f1s)), 5),
        "pe_f1_per_fold": [round(f, 5) for f in pe_f1s],
        "pe_confidence_on_pe": [round(c, 5) for c in all_pe_confs],
        "pe_rank_on_pe": all_pe_ranks,
        "fold_details": fold_results,
        "peak_vram_mb": get_peak_vram_mb(),
        "wall_clock_s": guard.elapsed(),
    }
    write_metrics(final, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_traceback(script_dir, e)
        write_status(script_dir, "crashed", reason=str(e))
        raise
