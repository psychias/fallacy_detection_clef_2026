"""
exp_0180 — Qwen2.5-7B CoT-LoRA on ST3 enhanced with kNN retrieval augmentation.

Key innovations over shared/train_qlora.py:
  1. CoT supervision: trains on reasoning_chain + label (not just label)
  2. kNN retrieval: each training/inference example gets k=3 similar examples
     retrieved from the train set as few-shot context
  3. bf16 compute on H100 for stability + speed
  4. 5-fold CV evaluation for honest comparison with encoder models

Decision gate: CV F1 > 0.80 → proceed to exp_0182 (32B)
"""

import json
import os
import sys
import random
import numpy as np
from pathlib import Path
from collections import Counter

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from shared.data_utils import (
    LABEL_MAPS, build_input_text, extract_label, load_jsonl,
)
from shared.train_qlora import parse_response
from shared.eval import evaluate, write_metrics
from shared.train_utils import (
    Heartbeat, WallClockGuard, get_device, get_peak_vram_mb,
    load_config, set_seed, write_status, write_traceback,
    check_disk, clear_hf_cache_for_model,
)

# ── Prompt templates ──────────────────────────────────────────────

SYSTEM_PROMPT_COT = (
    "You are an expert in argumentation theory. Classify the argumentation scheme "
    "in the following non-fallacious argument. First explain your reasoning step by "
    "step, then give your final answer.\n\n"
    "The four possible labels are:\n"
    "- practical-internal: argument for action based on internal factors (goals, values)\n"
    "- practical-external: argument for action based on external factors (consequences, norms)\n"
    "- epistemic-internal: argument for belief based on internal factors (evidence, logic)\n"
    "- epistemic-external: argument for belief based on external factors (authority, testimony)\n\n"
    "End your response with: LABEL: <your label>"
)

SYSTEM_PROMPT_PLAIN = (
    "Classify the argumentation scheme. Respond with exactly one of: "
    "practical-internal, practical-external, epistemic-internal, epistemic-external"
)


def load_cot_data(workspace_dir, cot_path):
    """Load CoT data from data_synth directory."""
    full_path = Path(workspace_dir) / "data_synth" / cot_path
    data = {}
    with open(full_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            data[entry["id"]] = entry
    return data


def format_few_shot_context(neighbors):
    """Format kNN neighbors as few-shot examples."""
    if not neighbors:
        return ""
    parts = ["Here are some similar examples:\n"]
    for i, n in enumerate(neighbors, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Text: {n['text_preview'][:300]}")
        parts.append(f"Label: {n['label']}\n")
    return "\n".join(parts)


def format_cot_train_example(text, label, reasoning_chain, neighbors,
                              tokenizer, max_len):
    """Format training example with CoT reasoning and few-shot context."""
    # Build user message with optional few-shot
    user_parts = []
    few_shot = format_few_shot_context(neighbors)
    if few_shot:
        user_parts.append(few_shot)
    user_parts.append(f"Now classify this argument:\n{text[:2000]}")
    user_content = "\n".join(user_parts)

    # Build assistant response with CoT
    if reasoning_chain:
        assistant_content = f"{reasoning_chain}\n\nLABEL: {label}"
    else:
        assistant_content = f"LABEL: {label}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    enc = tokenizer(full_text, max_length=max_len, truncation=True,
                    padding="max_length", return_tensors="pt")

    # Mask prompt tokens
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_enc = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    labels = enc["input_ids"].clone().squeeze(0)
    labels[:prompt_len] = -100
    # Also mask padding
    labels[enc["attention_mask"].squeeze(0) == 0] = -100

    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "labels": labels,
    }


def format_cot_inference(text, neighbors, tokenizer, max_len):
    """Format for CoT inference with few-shot context."""
    user_parts = []
    few_shot = format_few_shot_context(neighbors)
    if few_shot:
        user_parts.append(few_shot)
    user_parts.append(f"Now classify this argument:\n{text[:2000]}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, max_length=max_len, truncation=True,
                    return_tensors="pt")
    return enc


def parse_cot_response(text, subtask="st3"):
    """Parse label from CoT response. Look for LABEL: <label> pattern first."""
    import re
    # Try structured format first
    match = re.search(r'LABEL:\s*(\S+)', text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower().rstrip('.')
        valid = {"practical-internal", "practical-external",
                 "epistemic-internal", "epistemic-external"}
        if candidate in valid:
            return candidate

    # Fall back to general parsing
    return parse_response(text, subtask)


def train_one_fold(fold_idx, train_ids_set, dev_ids_set, all_data,
                   cot_data, retriever, config, device):
    """Train on one fold, return metrics."""
    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 2e-4)
    grad_accum = train_cfg.get("grad_accum", 16)
    epochs = train_cfg.get("epochs", 3)
    max_len = train_cfg.get("max_len", 1536)
    lora_r = train_cfg.get("lora_r", 32)
    lora_alpha = train_cfg.get("lora_alpha", 64)
    lora_dropout = train_cfg.get("lora_dropout", 0.05)
    patience = train_cfg.get("patience", 1)
    knn_k = config.get("data", {}).get("knn_k", 3)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    label2id, id2label = LABEL_MAPS["st3"]

    # Build train/dev entries
    train_entries, dev_entries = [], []
    for e in all_data:
        eid = e.get("id", "")
        label = extract_label(e, "st3")
        if label is None or label not in label2id:
            continue
        text = build_input_text(e, "enhanced", "st3")
        cot_entry = cot_data.get(eid, {})
        reasoning = cot_entry.get("reasoning_chain", "")

        entry = {"id": eid, "text": text, "label": label, "reasoning_chain": reasoning}
        if eid in train_ids_set:
            train_entries.append(entry)
        elif eid in dev_ids_set:
            dev_entries.append(entry)

    print(f"  Fold {fold_idx}: train={len(train_entries)} dev={len(dev_entries)}")

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        backbone, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules="all-linear", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_entries) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, min(total_steps // 10, 30), max(total_steps, 1)
    )

    rng = random.Random(42 + fold_idx)
    train_exclude = {e["id"] for e in dev_entries}

    best_f1 = -1.0
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        indices = list(range(len(train_entries)))
        rng.shuffle(indices)
        accum_loss = 0.0

        for step_i, idx in enumerate(indices):
            entry = train_entries[idx]

            # Retrieve nearest neighbors (exclude self and dev entries)
            neighbors = []
            if retriever is not None:
                neighbors = retriever.query(
                    entry["text"], subtask="st3", k=knn_k,
                    exclude_ids=train_exclude | {entry["id"]}
                )

            batch = format_cot_train_example(
                entry["text"], entry["label"], entry["reasoning_chain"],
                neighbors, tokenizer, max_len
            )
            input_ids = batch["input_ids"].unsqueeze(0).to(model.device)
            attn_mask = batch["attention_mask"].unsqueeze(0).to(model.device)
            labels = batch["labels"].unsqueeze(0).to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step_i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if ((step_i + 1) // grad_accum) % 10 == 0:
                    print(f"    fold={fold_idx} epoch={epoch} step={step_i+1} "
                          f"loss={accum_loss:.4f}")
                    accum_loss = 0.0

        # Leftover gradients
        optimizer.step()
        optimizer.zero_grad()

        # Dev eval
        model.eval()
        preds = []
        for entry in dev_entries:
            neighbors = []
            if retriever is not None:
                neighbors = retriever.query(
                    entry["text"], subtask="st3", k=knn_k,
                    exclude_ids={entry["id"]}
                )
            enc = format_cot_inference(entry["text"], neighbors, tokenizer, max_len)
            input_ids = enc["input_ids"].to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, max_new_tokens=300,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            generated = out[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = parse_cot_response(response)
            if parsed is None:
                parsed = "epistemic-internal"  # majority fallback
            preds.append(parsed)

        true_labels = [e["label"] for e in dev_entries]
        metrics = evaluate(true_labels, preds, subtask="st3")
        pe_f1 = metrics.get("f1_per_class", {}).get("practical-external", 0)

        print(f"  Fold {fold_idx} epoch={epoch}: F1={metrics['f1_macro']:.4f} "
              f"pe_F1={pe_f1:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            # Save only LoRA adapter weights (not quantized base model weights)
            best_state = {k: v.cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  Fold {fold_idx}: early stop at epoch {epoch}")
            break
        model.train()

    # Final eval with best state (LoRA params only)
    if best_state is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in best_state:
                    param.copy_(best_state[name].to(param.device))
    model.eval()

    preds = []
    for entry in dev_entries:
        neighbors = []
        if retriever is not None:
            neighbors = retriever.query(
                entry["text"], subtask="st3", k=knn_k,
                exclude_ids={entry["id"]}
            )
        enc = format_cot_inference(entry["text"], neighbors, tokenizer, max_len)
        input_ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, max_new_tokens=300,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_cot_response(response)
        if parsed is None:
            parsed = "epistemic-internal"
        preds.append(parsed)

    true_labels = [e["label"] for e in dev_entries]
    final_metrics = evaluate(true_labels, preds, subtask="st3")

    del model, best_state
    torch.cuda.empty_cache()
    import gc; gc.collect()

    dev_dist = Counter(e["label"] for e in dev_entries)
    return {
        "fold": fold_idx,
        "f1_macro": final_metrics["f1_macro"],
        "f1_per_class": final_metrics.get("f1_per_class", {}),
        "n_train": len(train_entries),
        "n_dev": len(dev_entries),
        "dev_pe_count": dev_dist.get("practical-external", 0),
    }


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)
    device = get_device()

    write_status(exp_dir, "running")
    hb = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 9000), margin_s=300)

    check_disk(min_gb=20.0)

    print(f"[0180] Qwen-7B CoT-LoRA with kNN retrieval — 5-fold CV")
    print(f"[0180] backbone={config['backbone']}")

    ws = workspace

    # Load data
    all_data = load_jsonl(str(ws / "data" / "touchefallacy_2026_train.jsonl"))

    # Load CoT data
    cot_path = config.get("data", {}).get("cot_data", "cot_v001_st3/data_enhanced.jsonl")
    cot_data = load_cot_data(ws, cot_path)
    print(f"[0180] CoT data: {len(cot_data)} entries")

    # Load kNN retriever
    retriever = None
    if config.get("data", {}).get("use_knn_retrieval", True):
        try:
            from shared.knn_retrieval import KNNRetriever, load_bge_encoder
            print("[0180] Loading bge-large encoder for kNN retrieval...")
            encoder = load_bge_encoder(device="cuda")
            retriever = KNNRetriever.from_exp0138(ws, encoder=encoder)
            print(f"[0180] kNN retriever loaded: {len(retriever.metadata)} entries")
        except Exception as e:
            print(f"[0180] WARNING: kNN retrieval failed to load: {e}")
            print("[0180] Proceeding without retrieval")

    # Load k-fold splits
    kfold_data = json.load(open(ws / "shared" / "kfold_splits_st3.json"))
    folds = kfold_data["folds"]

    print(f"[0180] Running 5-fold CV")

    fold_results = []
    for fold in folds:
        if guard.exceeded():
            print("[0180] Time exceeded, stopping")
            break
        result = train_one_fold(
            fold["fold"], set(fold["train_ids"]), set(fold["dev_ids"]),
            all_data, cot_data, retriever, config, device
        )
        fold_results.append(result)
        hb.beat()
        print(f"  Fold {result['fold']}: F1={result['f1_macro']:.4f}")

    # Aggregate
    macro_f1s = [r["f1_macro"] for r in fold_results]
    pe_f1s = [r["f1_per_class"].get("practical-external", 0) for r in fold_results]

    mean_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0
    std_f1 = float(np.std(macro_f1s)) if macro_f1s else 0

    print(f"\n{'='*60}")
    print(f"5-FOLD CV RESULTS — Qwen-7B CoT-LoRA")
    print(f"{'='*60}")
    print(f"Macro F1: {mean_f1:.4f} ± {std_f1:.4f}  per-fold: {[round(f,4) for f in macro_f1s]}")
    print(f"PE F1:    {np.mean(pe_f1s):.4f} ± {np.std(pe_f1s):.4f}  per-fold: {[round(f,4) for f in pe_f1s]}")

    # Decision gate
    if mean_f1 > 0.85:
        print(f"\n>>> GATE: EXCELLENT (F1={mean_f1:.4f} > 0.85). Proceed to exp_0182 (32B) confidently.")
    elif mean_f1 > 0.80:
        print(f"\n>>> GATE: DECENT (F1={mean_f1:.4f} > 0.80). Proceed to exp_0182 with moderate expectations.")
    elif mean_f1 > 0.78:
        print(f"\n>>> GATE: MARGINAL (F1={mean_f1:.4f}). Consider aborting 32B, redirect to breadth.")
    else:
        print(f"\n>>> GATE: FAIL (F1={mean_f1:.4f} < 0.78). ABORT exp_0182. Redirect H100 budget.")

    final = {
        "subtask": "st3",
        "track": "enhanced",
        "backbone": config["backbone"],
        "n_folds": len(fold_results),
        "macro_f1_mean": round(mean_f1, 5),
        "macro_f1_std": round(std_f1, 5),
        "macro_f1_per_fold": [round(f, 5) for f in macro_f1s],
        "pe_f1_mean": round(float(np.mean(pe_f1s)), 5) if pe_f1s else 0,
        "pe_f1_std": round(float(np.std(pe_f1s)), 5) if pe_f1s else 0,
        "pe_f1_per_fold": [round(f, 5) for f in pe_f1s],
        "fold_details": fold_results,
        "gate_decision": "proceed" if mean_f1 > 0.78 else "abort",
        "knn_retrieval": retriever is not None,
        "cot_examples": len(cot_data),
        "peak_vram_mb": round(get_peak_vram_mb(), 1),
        "wall_clock_s": round(guard.elapsed(), 1),
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
