"""
exp_0143/generate.py — kNN-retrieval few-shot classification with Qwen-3B.

Uses bge-large embeddings (from exp_0138) to retrieve nearest-neighbour
training examples as few-shot demonstrations, then prompts Qwen-3B
for ST3 classification.
"""

import json
import os
import sys
import numpy as np
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


# ── kNN retrieval ─────────────────────────────────────────────────

def load_embedding_index(embedding_exp: str):
    """Load embeddings and metadata from the embedding index experiment."""
    emb_dir = workspace / "experiments" / embedding_exp / "outputs"
    embeddings = np.load(str(emb_dir / "embeddings.npy"))
    metadata = []
    with open(emb_dir / "metadata.jsonl") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    return embeddings, metadata


def encode_query(texts: list[str], model_name="BAAI/bge-large-en-v1.5"):
    """Encode query texts with the same embedding model."""
    import subprocess
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"]
        )
        from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)
    # Free embedding model VRAM before loading the LLM
    del model
    torch.cuda.empty_cache()
    return np.array(embs, dtype=np.float32)


def retrieve_knn(query_embs, index_embs, index_meta, subtask, n_per_class=2):
    """
    For each query, retrieve n_per_class nearest examples per target class.
    Returns list of lists of (text_preview, label, score) tuples.
    """
    # Filter index to only matching subtask
    mask = [i for i, m in enumerate(index_meta) if m["subtask"] == subtask]
    sub_embs = index_embs[mask]
    sub_meta = [index_meta[i] for i in mask]

    # Get unique labels
    labels = sorted(set(m["label"] for m in sub_meta))

    # Group indices by label
    label_indices = {}
    for i, m in enumerate(sub_meta):
        label_indices.setdefault(m["label"], []).append(i)

    results = []
    for q_emb in query_embs:
        examples = []
        for label in labels:
            idxs = label_indices.get(label, [])
            if not idxs:
                continue
            # Compute cosine similarity (embeddings are normalized)
            l_embs = sub_embs[idxs]
            sims = l_embs @ q_emb
            top_k = min(n_per_class, len(idxs))
            top_indices = np.argsort(sims)[-top_k:][::-1]
            for ti in top_indices:
                idx = idxs[ti]
                examples.append((
                    sub_meta[idx]["text_preview"],
                    sub_meta[idx]["label"],
                    float(sims[ti]),
                ))
        results.append(examples)
    return results


# ── Prompt construction ───────────────────────────────────────────

ST3_SYSTEM = """You are an expert in argumentation theory and argument schemes. Your task is to classify a non-fallacious argument along two dimensions:
1. Goal: Is the argument "practical" (about what to do) or "epistemic" (about what to believe)?
2. Basis: Is the argument supported by "external" sources (authority, tradition, popular opinion) or "internal" reasoning (logic, analogy, cause-effect)?"""

def build_fewshot_prompt(text: str, examples: list[tuple], subtask: str = "st3") -> str:
    """Build a few-shot prompt with retrieved examples."""
    parts = ["Here are some example classifications:\n"]
    for ex_text, ex_label, _ in examples:
        parts.append(f"Argument: {ex_text}\nClassification: {ex_label}\n")
    parts.append(f"Now classify the following argument:\n\nArgument: {text}")
    parts.append("\nRespond with ONLY the label in the format: goal-basis")
    parts.append("Where goal is \"practical\" or \"epistemic\", and basis is \"internal\" or \"external\".")
    return "\n".join(parts)


# ── Response parsing ──────────────────────────────────────────────

import re

VALID_ST3 = {"practical-internal", "practical-external",
             "epistemic-internal", "epistemic-external"}

def parse_response(text: str) -> str | None:
    text = text.strip().lower()
    text = re.sub(r'["\'.!]', '', text).strip()
    for label in VALID_ST3:
        if label in text:
            return label
    goals = re.findall(r'(practical|epistemic)', text)
    bases = re.findall(r'(internal|external)', text)
    if goals and bases:
        return f"{goals[0]}-{bases[0]}"
    return None


# ── Main ──────────────────────────────────────────────────────────

def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)
    device = get_device()

    hb = Heartbeat(exp_dir)
    guard = WallClockGuard(config.get("time_budget_s", 1200))

    subtask = config["subtask"]
    track = config["track"]
    few_shot_cfg = config.get("few_shot", {})
    n_per_class = few_shot_cfg.get("n_per_class", 2)
    embedding_exp = few_shot_cfg.get("embedding_exp", "exp_0138")

    label2id, id2label = LABEL_MAPS[subtask]
    ws = workspace

    try:
        # Load data
        train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
        raw = load_jsonl(str(train_path))
        filtered = filter_for_subtask(raw, subtask)
        _, dev_data = make_splits(filtered, subtask, seed=42)
        print(f"Dev set: {len(dev_data)} examples")

        # Build input texts for dev
        dev_texts = [build_input_text(e, track, subtask) for e in dev_data]
        dev_labels = [extract_label(e, subtask) for e in dev_data]

        # Load embedding index and encode queries
        print("Loading embedding index...")
        index_embs, index_meta = load_embedding_index(embedding_exp)
        print(f"Index: {len(index_meta)} entries, shape {index_embs.shape}")

        print("Encoding dev queries...")
        query_embs = encode_query(dev_texts)
        hb.beat()

        # Retrieve kNN examples
        print("Retrieving kNN examples...")
        all_examples = retrieve_knn(query_embs, index_embs, index_meta,
                                    subtask, n_per_class)
        hb.beat()

        # Load LLM
        model_name = config["backbone"]
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)
        model.eval()
        print(f"Model loaded. VRAM: {get_peak_vram_mb():.0f} MB")
        hb.beat()

        # Generate predictions
        predictions = []
        for i, (text, examples) in enumerate(zip(dev_texts, all_examples)):
            if guard.exceeded():
                print(f"Time budget expired at {i}/{len(dev_texts)}")
                break

            user_prompt = build_fewshot_prompt(text, examples, subtask)
            messages = [
                {"role": "system", "content": ST3_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=2048).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            pred = parse_response(response)
            if pred is None:
                pred = list(label2id.keys())[0]  # fallback
            predictions.append(pred)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(dev_texts)}] last pred: {pred}")
                hb.beat()

        # Evaluate
        true_labels = dev_labels[:len(predictions)]

        metrics = evaluate(true_labels, predictions, subtask=subtask)
        metrics["n_predicted"] = len(predictions)
        metrics["n_total"] = len(dev_texts)
        metrics["n_per_class"] = n_per_class
        metrics["retrieval"] = "knn"
        metrics["peak_vram_mb"] = get_peak_vram_mb()

        write_metrics(metrics, str(exp_dir / "metrics.json"))
        print(f"\n=== Results ({subtask}, {track}, kNN few-shot) ===")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        write_status(exp_dir, "done")

    except Exception as e:
        write_traceback(exp_dir, e)
        write_status(exp_dir, "crashed", reason=str(e))
        raise


if __name__ == "__main__":
    main()
