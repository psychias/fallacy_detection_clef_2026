"""
exp_0091 — Qwen2.5-3B-Instruct zero-shot classification on ST1/ST2/ST3.

Pipeline validation for LLM approach on T4 (15GB VRAM).
Qwen2.5-3B-Instruct at float16 needs ~6GB. Runs zero-shot prompts
on the dev set for all three subtasks and reports F1.

Primary focus: ST3 scheme classification (goal × basis), where
the fine-tuned model struggles on the basis axis with pe F1=0.
"""

import json
import os
import re
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    extract_label,
    filter_for_subtask,
    load_jsonl,
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

ST1_SYSTEM = """You are an expert in argumentation and informal fallacies. Your task is to determine whether an argument contains a logical fallacy."""

ST1_USER = """Analyze the following argument and classify it as either "fallacy" or "non-fallacy".

{text}

Respond with ONLY one word: fallacy or non-fallacy"""

ST2_SYSTEM = """You are an expert in argumentation and informal fallacies. Your task is to identify the specific type of fallacy in an argument."""

ST2_USER = """Classify the fallacy type of the following argument. Choose exactly one from:
- authority (appeal to authority)
- black-white (false dilemma)
- hasty_generalization
- natural (appeal to nature)
- population (appeal to popularity/bandwagon)
- slippery_slope
- tradition (appeal to tradition)
- worse_problems (relative privation)

Argument:
{text}

Respond with ONLY the fallacy type label (e.g., "authority")."""

ST3_SYSTEM = """You are an expert in argumentation theory and argument schemes. Your task is to classify a non-fallacious argument along two dimensions:
1. Goal: Is the argument "practical" (about what to do) or "epistemic" (about what to believe)?
2. Basis: Is the argument supported by "external" sources (authority, tradition, popular opinion) or "internal" reasoning (logic, analogy, cause-effect)?"""

ST3_USER = """Classify the following non-fallacious argument:

{text}

Respond with ONLY the label in the format: goal-basis
Where goal is "practical" or "epistemic", and basis is "internal" or "external".
Example: practical-internal"""

# Separate basis-only prompt for targeted analysis
ST3_BASIS_SYSTEM = """You are an expert in argumentation theory. Your task is to determine whether an argument's support comes from external sources or internal reasoning."""

ST3_BASIS_USER = """Analyze the following argument and determine if its basis is:
- "external": The argument appeals to outside sources like authority, expert opinion, popular belief, tradition, or established practice.
- "internal": The argument relies on internal reasoning like logical deduction, analogy, cause-and-effect analysis, or examples.

Argument:
{text}

Respond with ONLY one word: external or internal"""

PROMPTS = {
    "st1": (ST1_SYSTEM, ST1_USER),
    "st2": (ST2_SYSTEM, ST2_USER),
    "st3": (ST3_SYSTEM, ST3_USER),
    "st3_basis": (ST3_BASIS_SYSTEM, ST3_BASIS_USER),
}

# Label normalization/parsing
VALID_ST1 = {"fallacy", "non-fallacy"}
VALID_ST2 = {"authority", "black-white", "hasty_generalization", "natural",
             "population", "slippery_slope", "tradition", "worse_problems"}
VALID_ST3 = {"practical-internal", "practical-external",
             "epistemic-internal", "epistemic-external"}
VALID_BASIS = {"external", "internal"}


def parse_response(text: str, subtask: str) -> str | None:
    """Parse model response to extract a valid label."""
    text = text.strip().lower()
    # Remove quotes, periods, extra whitespace
    text = re.sub(r'["\'.!]', '', text).strip()

    if subtask == "st1":
        if "non-fallacy" in text or "non_fallacy" in text or "nonfallacy" in text:
            return "non-fallacy"
        if "fallacy" in text:
            return "fallacy"
        return None

    elif subtask == "st2":
        # Check each valid label
        for label in VALID_ST2:
            if label in text or label.replace("-", "") in text or label.replace("_", " ") in text:
                return label
        # Common aliases
        if "appeal to authority" in text:
            return "authority"
        if "false dilemma" in text or "black and white" in text:
            return "black-white"
        if "hasty" in text:
            return "hasty_generalization"
        if "appeal to nature" in text or "naturalistic" in text:
            return "natural"
        if "bandwagon" in text or "popularity" in text or "ad populum" in text:
            return "population"
        if "slippery" in text:
            return "slippery_slope"
        if "tradition" in text:
            return "tradition"
        if "worse" in text or "privation" in text or "whatabout" in text:
            return "worse_problems"
        return None

    elif subtask == "st3":
        for label in VALID_ST3:
            if label in text:
                return label
        # Try to parse "goal-basis" pattern
        goals = re.findall(r'(practical|epistemic)', text)
        bases = re.findall(r'(internal|external)', text)
        if goals and bases:
            return f"{goals[0]}-{bases[0]}"
        return None

    elif subtask == "st3_basis":
        if "external" in text:
            return "external"
        if "internal" in text:
            return "internal"
        return None

    return None


def generate_batch(texts, system_prompt, user_template, tokenizer, model, device,
                   max_new_tokens=32):
    """Generate predictions for a batch of texts (one at a time for simplicity)."""
    predictions = []
    for text in texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(text=text)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=2048).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(response)
    return predictions


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 1200), margin_s=120)

    backbone = config["backbone"]
    track = config["track"]

    print(f"[llm_zeroshot] backbone={backbone}, track={track}")

    # ── Load model ────────────────────────────────────────────────
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        backbone,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[llm_zeroshot] Model loaded. VRAM={get_peak_vram_mb():.0f}MB")
    heartbeat.beat(step=0)

    # ── Load data ─────────────────────────────────────────────────
    ws = workspace
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))

    results = {}

    # ── Evaluate on each subtask ──────────────────────────────────
    for subtask in ["st3", "st1", "st2"]:
        if guard.exceeded():
            print(f"[llm_zeroshot] Time exceeded, skipping {subtask}")
            break

        label2id, id2label = LABEL_MAPS[subtask]
        _, dev_split = make_splits(all_data, subtask)

        dev_entries = []
        for e in dev_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                dev_entries.append({"id": e["id"], "text": text, "label": label})

        print(f"[llm_zeroshot] {subtask}: {len(dev_entries)} dev entries")

        system_prompt, user_template = PROMPTS[subtask]
        dev_texts = [e["text"] for e in dev_entries]
        true_labels = [e["label"] for e in dev_entries]

        # Generate predictions
        raw_preds = generate_batch(dev_texts, system_prompt, user_template,
                                   tokenizer, model, device)

        # Parse predictions
        parsed_preds = []
        parse_failures = 0
        for raw in raw_preds:
            parsed = parse_response(raw, subtask)
            if parsed is None:
                parse_failures += 1
                # Default fallback: most frequent label in training
                parsed = list(label2id.keys())[0]
            parsed_preds.append(parsed)

        # Evaluate
        metrics = evaluate(true_labels, parsed_preds, subtask)

        results[subtask] = {
            "f1_macro": metrics["f1_macro"],
            "f1_per_class": metrics["f1_per_class"],
            "parse_failures": parse_failures,
            "n_dev": len(dev_entries),
        }

        # Save per-example details
        per_example = []
        for i, entry in enumerate(dev_entries):
            per_example.append({
                "id": entry["id"],
                "true": entry["label"],
                "raw_pred": raw_preds[i][:200],
                "parsed_pred": parsed_preds[i],
            })
        results[f"{subtask}_details"] = per_example

        print(f"[llm_zeroshot] {subtask}: F1={metrics['f1_macro']:.4f}, "
              f"parse_fail={parse_failures}/{len(dev_entries)}")

        heartbeat.beat(step=["st3", "st1", "st2"].index(subtask) + 1)

    # ── ST3 basis-only analysis ───────────────────────────────────
    if not guard.exceeded():
        subtask = "st3"
        _, dev_split = make_splits(all_data, subtask)
        dev_entries_st3 = []
        for e in dev_split:
            label = extract_label(e, subtask)
            if label is not None:
                text = build_input_text(e, track, subtask)
                goal, basis = label.split("-")
                dev_entries_st3.append({
                    "id": e["id"], "text": text, "label": label,
                    "goal": goal, "basis": basis,
                })

        system_prompt, user_template = PROMPTS["st3_basis"]
        dev_texts = [e["text"] for e in dev_entries_st3]
        true_bases = [e["basis"] for e in dev_entries_st3]

        raw_basis_preds = generate_batch(dev_texts, system_prompt, user_template,
                                         tokenizer, model, device)
        parsed_bases = []
        for raw in raw_basis_preds:
            parsed = parse_response(raw, "st3_basis")
            if parsed is None:
                parsed = "internal"  # majority fallback
            parsed_bases.append(parsed)

        basis_correct = sum(1 for p, t in zip(parsed_bases, true_bases) if p == t)
        basis_acc = basis_correct / len(true_bases) if true_bases else 0

        results["st3_basis_only"] = {
            "accuracy": round(basis_acc, 4),
            "correct": basis_correct,
            "total": len(true_bases),
        }
        print(f"[llm_zeroshot] ST3 basis-only accuracy: {basis_acc:.4f}")

    # ── Summary ───────────────────────────────────────────────────
    results["backbone"] = backbone
    results["track"] = track
    results["peak_vram_mb"] = round(get_peak_vram_mb(), 1)
    results["wall_clock_s"] = round(guard.elapsed(), 1)

    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("[llm_zeroshot] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[llm_zeroshot] CRASHED: {exc}")
        sys.exit(1)
