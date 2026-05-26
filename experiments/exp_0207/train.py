"""
exp_0207 — Qwen2.5-32B CoT-LoRA on ST3 enhanced — HYPERPARAM FIX.

Follow-up to exp_0182 (F1=0.626) which undertrained badly:
  - grad_accum 32→4  (~300 optimizer steps vs ~25)
  - lr 1e-4→2e-4     (match the successful 7B recipe)
  - epochs 2→3       (more passes over the data)
  - patience 1→2     (allow recovery from bad epoch)

Everything else identical: 32B 4-bit NF4, LoRA r=32, kNN retrieval, CoT.
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
    LABEL_MAPS, build_input_text, extract_label, load_jsonl, make_splits,
)
from shared.eval import evaluate, write_metrics
from shared.train_utils import (
    Heartbeat, WallClockGuard, get_device, get_peak_vram_mb,
    load_config, set_seed, write_status, write_traceback,
    check_disk, clear_hf_cache_for_model,
)

# Import from exp_0180's shared logic
sys.path.insert(0, str(script_dir.parent / "exp_0180"))
from train import (
    load_cot_data, format_cot_train_example, format_cot_inference,
    parse_cot_response, format_few_shot_context,
    SYSTEM_PROMPT_COT,
)

TAG = "[0207]"


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)
    device = get_device()

    write_status(exp_dir, "running")
    hb = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 21600), margin_s=600)

    # Clear previous model caches to make room for 32B
    clear_hf_cache_for_model("Qwen/Qwen2.5-7B-Instruct")
    check_disk(min_gb=30.0)

    backbone = config["backbone"]
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 2e-4)
    grad_accum = train_cfg.get("grad_accum", 4)
    epochs = train_cfg.get("epochs", 3)
    max_len = train_cfg.get("max_len", 1536)
    lora_r = train_cfg.get("lora_r", 32)
    lora_alpha = train_cfg.get("lora_alpha", 64)
    lora_dropout = train_cfg.get("lora_dropout", 0.05)
    patience = train_cfg.get("patience", 2)
    knn_k = config.get("data", {}).get("knn_k", 3)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"{TAG} Qwen-32B CoT-LoRA — HYPERPARAM FIX (grad_accum={grad_accum}, lr={lr}, epochs={epochs})")
    print(f"{TAG} backbone={backbone}, max_len={max_len}")

    ws = workspace
    label2id, id2label = LABEL_MAPS["st3"]

    # Check gate: exp_0180 must have passed
    gate_path = ws / "experiments" / "exp_0180" / "metrics.json"
    if gate_path.exists():
        gate_metrics = json.load(open(gate_path))
        gate_f1 = gate_metrics.get("macro_f1_mean", 0)
        gate_decision = gate_metrics.get("gate_decision", "unknown")
        print(f"{TAG} Gate check: exp_0180 F1={gate_f1:.4f}, decision={gate_decision}")
        if gate_f1 < 0.78:
            print(f"{TAG} GATE FAILED. Aborting.")
            write_metrics({"aborted": True, "reason": f"exp_0180 F1={gate_f1} < 0.78"},
                         str(exp_dir / "metrics.json"))
            write_status(exp_dir, "done")
            return
    else:
        print(f"{TAG} WARNING: exp_0180 metrics not found. Proceeding anyway.")

    # Load data — FULL train/dev split (not CV)
    all_data = load_jsonl(str(ws / "data" / "touchefallacy_2026_train.jsonl"))
    train_split, dev_split = make_splits(all_data, "st3")

    # Load CoT data
    cot_path = config.get("data", {}).get("cot_data", "cot_v001_st3/data_enhanced.jsonl")
    cot_data = load_cot_data(ws, cot_path)
    print(f"{TAG} CoT data: {len(cot_data)} entries")

    # Build entries
    train_entries = []
    for e in train_split:
        label = extract_label(e, "st3")
        if label is not None and label in label2id:
            text = build_input_text(e, "enhanced", "st3")
            eid = e.get("id", "")
            cot_entry = cot_data.get(eid, {})
            reasoning = cot_entry.get("reasoning_chain", "")
            train_entries.append({
                "id": eid, "text": text, "label": label,
                "reasoning_chain": reasoning,
            })

    dev_entries = []
    for e in dev_split:
        label = extract_label(e, "st3")
        if label is not None and label in label2id:
            text = build_input_text(e, "enhanced", "st3")
            dev_entries.append({
                "id": e.get("id", ""), "text": text, "label": label,
            })

    total_steps = (len(train_entries) // grad_accum) * epochs
    print(f"{TAG} train={len(train_entries)}, dev={len(dev_entries)}")
    print(f"{TAG} total optimizer steps: {total_steps} ({len(train_entries)}//{grad_accum} * {epochs})")
    print(f"{TAG} train dist: {dict(Counter(e['label'] for e in train_entries))}")

    # Load kNN retriever
    retriever = None
    if config.get("data", {}).get("use_knn_retrieval", True):
        try:
            from shared.knn_retrieval import KNNRetriever, load_bge_encoder
            print(f"{TAG} Loading bge-large encoder for kNN retrieval...")
            encoder = load_bge_encoder(device="cuda")
            retriever = KNNRetriever.from_exp0138(ws, encoder=encoder)
            print(f"{TAG} kNN retriever loaded: {len(retriever.metadata)} entries")
        except Exception as e:
            print(f"{TAG} WARNING: kNN retrieval failed: {e}")

    # Load model
    print(f"{TAG} Loading {backbone} in 4-bit NF4 (compute={compute_dtype})...")
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
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules="all-linear", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"{TAG} VRAM after load: {get_peak_vram_mb():.0f}MB")
    hb.beat(step=0)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, min(total_steps // 10, 30), max(total_steps, 1)
    )

    rng = random.Random(42)
    dev_ids = {e["id"] for e in dev_entries}
    best_dev_f1 = -1.0
    no_improve = 0
    status_flags = []

    for epoch in range(epochs):
        if guard.exceeded():
            status_flags.append("time_exceeded")
            break

        model.train()
        indices = list(range(len(train_entries)))
        rng.shuffle(indices)
        accum_loss = 0.0

        for step_i, idx in enumerate(indices):
            if guard.exceeded():
                status_flags.append("time_exceeded")
                break

            entry = train_entries[idx]
            neighbors = []
            if retriever is not None:
                neighbors = retriever.query(
                    entry["text"], subtask="st3", k=knn_k,
                    exclude_ids=dev_ids | {entry["id"]}
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
                hb.beat()

                global_step = (step_i + 1) // grad_accum
                if global_step % 10 == 0:
                    print(f"{TAG} epoch={epoch} step={global_step}/{len(train_entries)//grad_accum} "
                          f"loss={accum_loss:.4f}")
                    accum_loss = 0.0

        if "time_exceeded" in status_flags:
            break

        # Flush any remaining accumulated gradients
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
        metrics = evaluate(true_labels, preds, subtask="st3")
        pe_f1 = metrics.get("f1_per_class", {}).get("practical-external", 0)

        print(f"{TAG} epoch={epoch}: dev F1={metrics['f1_macro']:.4f} pe_F1={pe_f1:.4f}")

        if metrics["f1_macro"] > best_dev_f1:
            best_dev_f1 = metrics["f1_macro"]
            no_improve = 0
            # Save checkpoint
            ckpt_dir = exp_dir / "ckpt" / "best"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"{TAG} New best F1={best_dev_f1:.4f}, checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                status_flags.append("early_stopped")
                print(f"{TAG} Early stopped at epoch {epoch}")
                break
        model.config.use_cache = False

    # Final metrics
    final = {
        "subtask": "st3",
        "track": "enhanced",
        "backbone": backbone,
        "f1_macro": round(best_dev_f1, 5),
        "status_flags": status_flags,
        "n_train": len(train_entries),
        "n_dev": len(dev_entries),
        "knn_retrieval": retriever is not None,
        "cot_examples": len(cot_data),
        "peak_vram_mb": round(get_peak_vram_mb(), 1),
        "wall_clock_s": round(guard.elapsed(), 1),
        "lora_r": lora_r,
        "lr": lr,
        "grad_accum": grad_accum,
        "total_optimizer_steps": total_steps,
    }
    write_metrics(final, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print(f"{TAG} DONE. Best dev F1={best_dev_f1:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_traceback(script_dir, e)
        write_status(script_dir, "crashed", reason=str(e))
