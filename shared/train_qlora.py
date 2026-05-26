"""
Touché 2026 — QLoRA fine-tuning for causal LLMs (Qwen2.5 series).

Loads model in 4-bit quantization, applies LoRA adapters to all linear layers,
fine-tunes on text classification by training on prompt→label sequences.

Config params:
  backbone: "Qwen/Qwen2.5-7B-Instruct"
  train.lora_r: 16
  train.lora_alpha: 32
  train.lr: 2e-4
  train.grad_accum: 16
  train.lora_target: "all" or specific module names
"""

import json
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    extract_label,
    load_jsonl,
    load_synth_data,
    make_splits,
)
from shared.eval import evaluate, write_metrics
from shared.train_utils import (
    Heartbeat,
    WallClockGuard,
    get_device,
    get_peak_vram_mb,
    load_config,
    set_seed,
    write_status,
    write_traceback,
)

# ── Prompt templates ──────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "st1": "Classify whether the following argument contains a logical fallacy. Respond with exactly: fallacy or non-fallacy",
    "st2": "Classify the type of fallacy in the following argument. Respond with exactly one of: authority, black-white, hasty_generalization, natural, population, slippery_slope, tradition, worse_problems",
    "st3": "Classify the argumentation scheme in the following non-fallacious argument. Respond with exactly one of: practical-internal, practical-external, epistemic-internal, epistemic-external",
}


def format_train_example(text, label, subtask, tokenizer, max_len=1024):
    """Format a training example as a chat template with the label as response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[subtask]},
        {"role": "user", "content": text[:2000]},
        {"role": "assistant", "content": label},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Tokenize and create labels (mask everything except assistant response)
    enc = tokenizer(full_text, max_length=max_len, truncation=True,
                    padding="max_length", return_tensors="pt")

    # Find where the assistant response starts
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[subtask]},
        {"role": "user", "content": text[:2000]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_enc = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Labels: -100 for prompt tokens, actual token ids for response
    labels = enc["input_ids"].clone().squeeze(0)
    labels[:prompt_len] = -100

    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "labels": labels,
    }


def format_inference(text, subtask, tokenizer, max_len=1024):
    """Format for inference (no label)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[subtask]},
        {"role": "user", "content": text[:2000]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, max_length=max_len, truncation=True,
                    return_tensors="pt")
    return enc


import re

VALID_LABELS = {
    "st1": {"fallacy", "non-fallacy"},
    "st2": {"authority", "black-white", "hasty_generalization", "natural",
            "population", "slippery_slope", "tradition", "worse_problems"},
    "st3": {"practical-internal", "practical-external",
            "epistemic-internal", "epistemic-external"},
}


def parse_response(text, subtask):
    text = text.strip().lower().split("\n")[0]
    text = re.sub(r'["\'.!]', '', text).strip()
    for label in VALID_LABELS[subtask]:
        if label in text:
            return label
    return None


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    seed = config.get("seed", 42)
    set_seed(seed)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 3600), margin_s=120)

    subtask = config["subtask"]
    track = config["track"]
    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 2e-4)
    batch_size = train_cfg.get("batch_size", 1)
    grad_accum = train_cfg.get("grad_accum", 16)
    epochs = train_cfg.get("epochs", 3)
    max_len = train_cfg.get("max_len", 1024)
    lora_r = train_cfg.get("lora_r", 16)
    lora_alpha = train_cfg.get("lora_alpha", 32)
    lora_dropout = train_cfg.get("lora_dropout", 0.05)

    label2id, id2label = LABEL_MAPS[subtask]

    print(f"[qlora] exp={exp_dir.name}, subtask={subtask}, backbone={backbone}")
    print(f"[qlora] lora_r={lora_r}, alpha={lora_alpha}, lr={lr}, grad_accum={grad_accum}")

    # ── Load model in 4-bit ───────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        backbone,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[qlora] VRAM after model load: {get_peak_vram_mb():.0f}MB")
    heartbeat.beat(step=0)

    # ── Data ──────────────────────────────────────────────────────
    ws = workspace
    all_data = load_jsonl(str(ws / "data" / "touchefallacy_2026_train.jsonl"))
    train_split, dev_split = make_splits(all_data, subtask)

    train_entries = []
    for e in train_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            train_entries.append({"text": text, "label": label})

    for sv in config.get("data", {}).get("synth_versions", []):
        try:
            for e in load_synth_data(sv, subtask):
                label = extract_label(e, subtask)
                if label and label in label2id:
                    text = build_input_text(e, track, subtask)
                    train_entries.append({"text": text, "label": label})
        except FileNotFoundError:
            pass

    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            dev_entries.append({"id": e["id"], "text": text, "label": label})

    print(f"[qlora] train={len(train_entries)}, dev={len(dev_entries)}")

    # ── Training ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_entries) // (batch_size * grad_accum)) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, min(total_steps // 10, 50), total_steps
    )

    model.train()
    global_step = 0
    accum_loss = 0.0
    best_dev_f1 = -1.0
    status_flags = []

    import random
    rng = random.Random(seed)

    for epoch in range(epochs):
        if guard.exceeded():
            status_flags.append("time_exceeded")
            break

        indices = list(range(len(train_entries)))
        rng.shuffle(indices)

        for i, idx in enumerate(indices):
            if guard.exceeded():
                status_flags.append("time_exceeded")
                break

            entry = train_entries[idx]
            batch = format_train_example(
                entry["text"], entry["label"], subtask, tokenizer, max_len
            )
            input_ids = batch["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(model.device)
            labels = batch["labels"].unsqueeze(0).to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                heartbeat.beat(step=global_step)

                if global_step % 20 == 0:
                    print(f"[qlora] step={global_step}, loss={accum_loss:.4f}")
                    accum_loss = 0.0

        if "time_exceeded" in status_flags:
            break

        # ── Dev eval ──────────────────────────────────────────
        model.eval()
        preds = []
        for entry in dev_entries:
            enc = format_inference(entry["text"], subtask, tokenizer, max_len)
            input_ids = enc["input_ids"].to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = out[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = parse_response(response, subtask)
            if parsed is None:
                parsed = list(label2id.keys())[0]
            preds.append(parsed)

        true_labels = [e["label"] for e in dev_entries]
        metrics = evaluate(true_labels, preds, subtask)
        print(f"[qlora] epoch={epoch}, dev_f1={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            model.save_pretrained(str(exp_dir / "ckpt" / "best"))
            tokenizer.save_pretrained(str(exp_dir / "ckpt" / "best"))
            print(f"[qlora] New best F1: {best_dev_f1:.4f}")

        model.train()

    # ── Save results ──────────────────────────────────────────────
    metrics["wall_clock_s"] = round(guard.elapsed(), 1)
    metrics["peak_vram_mb"] = round(get_peak_vram_mb(), 1)
    metrics["backbone"] = backbone
    metrics["track"] = track
    metrics["lora_r"] = lora_r
    metrics["lora_alpha"] = lora_alpha
    metrics["status_flags"] = status_flags

    write_metrics(metrics, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"[qlora] DONE. F1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        sys.exit(1)
