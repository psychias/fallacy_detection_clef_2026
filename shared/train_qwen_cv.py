"""
shared/train_qwen_cv.py — Generalized Qwen-7B CoT-LoRA 5-fold CV training.

Parameterized by subtask (st1/st2/st3), track (enhanced/base),
seed, lora_r, knn_k, etc. — all read from config.json.

Used by exp_0209–exp_0215 and future Qwen-7B CV experiments.
"""

import json
import os
import sys
import re
import random
import numpy as np
from pathlib import Path
from collections import Counter

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
    check_disk,
)


# ── Per-subtask system prompts ─────────────────────────────────────

SYSTEM_PROMPTS = {
    "st3": (
        "You are an expert in argumentation theory. Classify the argumentation scheme "
        "in the following non-fallacious argument. First explain your reasoning step by "
        "step, then give your final answer.\n\n"
        "The four possible labels are:\n"
        "- practical-internal: argument for action based on internal factors (goals, values)\n"
        "- practical-external: argument for action based on external factors (consequences, norms)\n"
        "- epistemic-internal: argument for belief based on internal factors (evidence, logic)\n"
        "- epistemic-external: argument for belief based on external factors (authority, testimony)\n\n"
        "End your response with: LABEL: <your label>"
    ),
    "st1": (
        "You are an expert in argumentation analysis. Determine whether the following "
        "text contains a logical fallacy. First explain your reasoning step by step, "
        "then give your final answer.\n\n"
        "The two possible labels are:\n"
        "- fallacy: the text contains a logical fallacy\n"
        "- non-fallacy: the text does not contain a logical fallacy\n\n"
        "End your response with: LABEL: <your label>"
    ),
    "st2": (
        "You are an expert in argumentation analysis. Identify the type of logical "
        "fallacy in the following text. First explain your reasoning step by step, "
        "then give your final answer.\n\n"
        "The possible fallacy types are:\n"
        "- authority: appeal to authority\n"
        "- black-white: false dilemma / black-or-white thinking\n"
        "- hasty_generalization: hasty generalization from limited examples\n"
        "- natural: appeal to nature\n"
        "- population: appeal to the majority\n"
        "- slippery_slope: slippery slope argument\n"
        "- tradition: appeal to tradition\n"
        "- worse_problems: relative privation / worse problems\n\n"
        "End your response with: LABEL: <your label>"
    ),
}

# Majority-class fallback per subtask
MAJORITY_FALLBACK = {
    "st1": "fallacy",
    "st2": "hasty_generalization",
    "st3": "epistemic-internal",
}

VALID_LABELS = {
    "st1": {"fallacy", "non-fallacy"},
    "st2": {"authority", "black-white", "hasty_generalization", "natural",
            "population", "slippery_slope", "tradition", "worse_problems"},
    "st3": {"practical-internal", "practical-external",
            "epistemic-internal", "epistemic-external"},
}


def load_cot_data(workspace_dir, cot_path):
    """Load CoT data from data_synth directory. Returns {} if file missing."""
    full_path = Path(workspace_dir) / "data_synth" / cot_path
    if not full_path.exists():
        return {}
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
                              tokenizer, max_len, subtask="st3"):
    """Format training example with CoT reasoning and few-shot context."""
    system_prompt = SYSTEM_PROMPTS[subtask]

    user_parts = []
    few_shot = format_few_shot_context(neighbors)
    if few_shot:
        user_parts.append(few_shot)
    user_parts.append(f"Now classify this argument:\n{text[:2000]}")
    user_content = "\n".join(user_parts)

    if reasoning_chain:
        assistant_content = f"{reasoning_chain}\n\nLABEL: {label}"
    else:
        assistant_content = f"LABEL: {label}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    enc = tokenizer(full_text, max_length=max_len, truncation=True,
                    padding="max_length", return_tensors="pt")

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_enc = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    labels = enc["input_ids"].clone().squeeze(0)
    labels[:prompt_len] = -100
    labels[enc["attention_mask"].squeeze(0) == 0] = -100

    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "labels": labels,
    }


def format_cot_inference(text, neighbors, tokenizer, max_len, subtask="st3"):
    """Format for CoT inference with few-shot context."""
    system_prompt = SYSTEM_PROMPTS[subtask]

    user_parts = []
    few_shot = format_few_shot_context(neighbors)
    if few_shot:
        user_parts.append(few_shot)
    user_parts.append(f"Now classify this argument:\n{text[:2000]}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, max_length=max_len, truncation=True,
                    return_tensors="pt")
    return enc


def parse_cot_response(text, subtask="st3"):
    """Parse label from CoT response."""
    match = re.search(r'LABEL:\s*(\S+)', text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower().rstrip('.')
        if candidate in VALID_LABELS.get(subtask, set()):
            return candidate
    return parse_response(text, subtask)


def train_one_fold(fold_idx, train_ids_set, dev_ids_set, all_data,
                   cot_data, retriever, config, device, guard, hb):
    """Train on one fold. Reads subtask/track/params from config."""
    backbone = config["backbone"]
    subtask = config["subtask"]
    track = config.get("track", "enhanced")
    seed = config.get("seed", 42)
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
    label2id, id2label = LABEL_MAPS[subtask]
    fallback_label = MAJORITY_FALLBACK.get(subtask, list(label2id.keys())[0])

    # Build train/dev entries
    train_entries, dev_entries = [], []
    for e in all_data:
        eid = e.get("id", "")
        label = extract_label(e, subtask)
        if label is None or label not in label2id:
            continue
        text = build_input_text(e, track, subtask)
        cot_entry = cot_data.get(eid, {})
        reasoning = cot_entry.get("reasoning_chain", "")
        entry = {"id": eid, "text": text, "label": label, "reasoning_chain": reasoning}
        if eid in train_ids_set:
            train_entries.append(entry)
        elif eid in dev_ids_set:
            dev_entries.append(entry)

    total_steps = (len(train_entries) // grad_accum) * epochs
    print(f"  Fold {fold_idx}: train={len(train_entries)} dev={len(dev_entries)} "
          f"steps={total_steps}")

    # Load model
    use_quant = train_cfg.get("quantize", True)  # default: 4-bit NF4
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            backbone, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # FP8-quantized models: _grouped_mm in MoE doesn't support Float8_e4m3fn,
        # so force bf16 dequantization at load time.
        is_fp8 = "fp8" in backbone.lower()
        load_dtype = torch.bfloat16 if is_fp8 else compute_dtype
        model = AutoModelForCausalLM.from_pretrained(
            backbone, torch_dtype=load_dtype,
            device_map="auto", trust_remote_code=True,
        )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # FP8 ParamWrapper layers don't support dropout — force 0 for FP8 models
    is_fp8 = "fp8" in backbone.lower()
    effective_dropout = 0.0 if is_fp8 else lora_dropout

    # LoRA target modules:
    #   - dense models (Qwen2.5-*): attention + MLP
    #   - MoE models (Qwen3-*-A3B-*): attention only — targeting gate/up/down_proj
    #     would wrap 128 experts × 3 projs per layer, with only top-k=8 routed
    #     per token (most adapters never see gradient).
    default_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "A3B" not in backbone:
        default_targets += ["gate_proj", "up_proj", "down_proj"]
    target_modules = train_cfg.get("lora_target_modules", default_targets)

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=effective_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, min(total_steps // 10, 30), max(total_steps, 1)
    )

    rng = random.Random(seed + fold_idx)
    dev_exclude = {e["id"] for e in dev_entries}

    best_f1 = -1.0
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        if guard.exceeded():
            print(f"  Fold {fold_idx}: time exceeded at epoch {epoch}")
            break

        model.train()
        model.config.use_cache = False
        indices = list(range(len(train_entries)))
        rng.shuffle(indices)
        accum_loss = 0.0

        for step_i, idx in enumerate(indices):
            if guard.exceeded():
                break

            entry = train_entries[idx]
            neighbors = []
            if retriever is not None:
                neighbors = retriever.query(
                    entry["text"], subtask=subtask, k=knn_k,
                    exclude_ids=dev_exclude | {entry["id"]}
                )

            batch = format_cot_train_example(
                entry["text"], entry["label"], entry["reasoning_chain"],
                neighbors, tokenizer, max_len, subtask=subtask
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
                hb.beat()

                global_step = (step_i + 1) // grad_accum
                if global_step % 10 == 0:
                    print(f"    fold={fold_idx} ep={epoch} step={global_step} "
                          f"loss={accum_loss:.4f}")
                    accum_loss = 0.0

        # Flush leftover gradients
        optimizer.step()
        optimizer.zero_grad()

        # Dev eval
        model.eval()
        model.config.use_cache = True
        preds = []
        for entry in dev_entries:
            neighbors = []
            if retriever is not None:
                neighbors = retriever.query(
                    entry["text"], subtask=subtask, k=knn_k,
                    exclude_ids={entry["id"]}
                )
            enc = format_cot_inference(entry["text"], neighbors, tokenizer, max_len,
                                       subtask=subtask)
            input_ids = enc["input_ids"].to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, max_new_tokens=300,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            generated = out[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = parse_cot_response(response, subtask)
            if parsed is None:
                parsed = fallback_label
            preds.append(parsed)

        true_labels = [e["label"] for e in dev_entries]
        metrics = evaluate(true_labels, preds, subtask=subtask)

        print(f"  Fold {fold_idx} epoch={epoch}: F1={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.named_parameters()
                          if v.requires_grad}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Fold {fold_idx}: early stop at epoch {epoch}")
                break

    # Final eval with best state
    if best_state is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in best_state:
                    param.copy_(best_state[name].to(param.device))
    model.eval()
    model.config.use_cache = True

    preds = []
    for entry in dev_entries:
        neighbors = []
        if retriever is not None:
            neighbors = retriever.query(
                entry["text"], subtask=subtask, k=knn_k,
                exclude_ids={entry["id"]}
            )
        enc = format_cot_inference(entry["text"], neighbors, tokenizer, max_len,
                                   subtask=subtask)
        input_ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, max_new_tokens=300,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_cot_response(response, subtask)
        if parsed is None:
            parsed = fallback_label
        preds.append(parsed)

    true_labels = [e["label"] for e in dev_entries]
    final_metrics = evaluate(true_labels, preds, subtask=subtask)

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
        "dev_dist": dict(dev_dist),
    }


def run_cv(exp_dir: Path):
    """Main entry point: run 5-fold CV from config.json in exp_dir."""
    workspace = exp_dir.parent.parent
    sys.path.insert(0, str(workspace))

    config = load_config(exp_dir)
    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device()

    subtask = config["subtask"]
    track = config.get("track", "enhanced")
    tag = f"[{exp_dir.name}]"

    write_status(exp_dir, "running")
    hb = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 14400), margin_s=300)

    check_disk(min_gb=20.0)

    print(f"{tag} Qwen-7B CoT-LoRA 5-fold CV — {subtask} {track}")
    print(f"{tag} backbone={config['backbone']} seed={seed}")
    print(f"{tag} lora_r={config['train'].get('lora_r', 32)} "
          f"knn_k={config.get('data', {}).get('knn_k', 3)} "
          f"grad_accum={config['train'].get('grad_accum', 16)}")

    # Load data
    all_data = load_jsonl(str(workspace / "data" / "touchefallacy_2026_train.jsonl"))

    # Load CoT data (may be empty if not generated yet)
    cot_path = config.get("data", {}).get("cot_data", "")
    cot_data = load_cot_data(workspace, cot_path) if cot_path else {}
    print(f"{tag} CoT data: {len(cot_data)} entries")

    # Load kNN retriever
    retriever = None
    if config.get("data", {}).get("use_knn_retrieval", True):
        try:
            from shared.knn_retrieval import KNNRetriever, load_bge_encoder
            print(f"{tag} Loading bge-large encoder for kNN retrieval...")
            encoder = load_bge_encoder(device="cuda")
            retriever = KNNRetriever.from_exp0138(workspace, encoder=encoder)
            print(f"{tag} kNN retriever loaded: {len(retriever.metadata)} entries")
        except Exception as e:
            print(f"{tag} WARNING: kNN retrieval failed: {e}")

    # Load k-fold splits
    splits_file = workspace / "shared" / f"kfold_splits_{subtask}.json"
    kfold_data = json.load(open(splits_file))
    folds = kfold_data["folds"]

    print(f"{tag} Running 5-fold CV")

    fold_results = []
    for fold in folds:
        if guard.exceeded():
            print(f"{tag} Time exceeded, stopping after {len(fold_results)} folds")
            break
        print(f"\n{tag} === FOLD {fold['fold']} ===")
        result = train_one_fold(
            fold["fold"], set(fold["train_ids"]), set(fold["dev_ids"]),
            all_data, cot_data, retriever, config, device, guard, hb
        )
        fold_results.append(result)
        hb.beat()
        print(f"{tag} Fold {result['fold']}: F1={result['f1_macro']:.4f}")

    # Aggregate
    macro_f1s = [r["f1_macro"] for r in fold_results]
    mean_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0
    std_f1 = float(np.std(macro_f1s)) if macro_f1s else 0

    # Per-class aggregation
    all_classes = set()
    for r in fold_results:
        all_classes.update(r["f1_per_class"].keys())
    class_f1s = {}
    for cls in sorted(all_classes):
        vals = [r["f1_per_class"].get(cls, 0) for r in fold_results]
        class_f1s[cls] = {"mean": round(float(np.mean(vals)), 5),
                          "std": round(float(np.std(vals)), 5)}

    print(f"\n{'='*60}")
    print(f"5-FOLD CV RESULTS — {subtask} {track}")
    print(f"{'='*60}")
    print(f"Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"Per-fold: {[round(f, 4) for f in macro_f1s]}")
    for cls, stats in class_f1s.items():
        print(f"  {cls}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    final = {
        "subtask": subtask,
        "track": track,
        "backbone": config["backbone"],
        "n_folds": len(fold_results),
        "macro_f1_mean": round(mean_f1, 5),
        "macro_f1_std": round(std_f1, 5),
        "macro_f1_per_fold": [round(f, 5) for f in macro_f1s],
        "class_f1": class_f1s,
        "fold_details": fold_results,
        "knn_retrieval": retriever is not None,
        "cot_examples": len(cot_data),
        "seed": seed,
        "lora_r": config["train"].get("lora_r", 32),
        "knn_k": config.get("data", {}).get("knn_k", 3),
        "peak_vram_mb": round(get_peak_vram_mb(), 1),
        "wall_clock_s": round(guard.elapsed(), 1),
    }
    write_metrics(final, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"{tag} DONE.")
