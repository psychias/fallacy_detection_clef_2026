"""
exp_0081 — Pseudo-label leak audit.

Checks whether exp_0072's 0.970 ST2 dev F1 could be inflated by data leakage
between pseudo-labeled test entries (pseudo_v001_st2) and the dev split.

Diagnostics:
  1. ID overlap: any pseudo IDs in the dev split?
  2. Exact text match: any identical text_raw between pseudo & dev?
  3. Token Jaccard: pairwise token overlap between pseudo entries and dev entries.
  4. Embedding cosine similarity: using the backbone's [CLS] embeddings.
  5. Reports min/max/mean/p95 for all overlap measures.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from shared.data_utils import (
    LABEL_MAPS,
    build_input_text,
    extract_label,
    filter_for_subtask,
    load_jsonl,
    load_synth_data,
    make_splits,
)
from shared.eval import write_metrics
from shared.train_utils import (
    Heartbeat,
    WallClockGuard,
    get_device,
    load_config,
    set_seed,
    write_status,
    write_traceback,
)


def tokenize_simple(text: str) -> set[str]:
    """Whitespace + lowercase tokenization for Jaccard."""
    return set(text.lower().split())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def encode_texts(texts: list[str], tokenizer, model, device, max_len=384, batch_size=16):
    """Encode texts to [CLS] embeddings."""
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, max_length=max_len, truncation=True,
                padding=True, return_tensors="pt"
            ).to(device)
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :]  # [CLS]
            cls_emb = torch.nn.functional.normalize(cls_emb, dim=-1)
            all_embs.append(cls_emb.cpu())
    return torch.cat(all_embs, dim=0)  # (N, hidden)


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 600), margin_s=60)

    subtask = config["subtask"]
    track = config["track"]
    backbone = config["backbone"]

    label2id, id2label = LABEL_MAPS[subtask]

    print(f"[leak_audit] subtask={subtask}, track={track}, backbone={backbone}")

    # ── Load real data splits ─────────────────────────────────────
    ws = workspace
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    train_split, dev_split = make_splits(all_data, subtask)

    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            dev_entries.append({"id": e["id"], "text": text, "text_raw": e.get("text_raw", "")})

    # ── Load pseudo data ──────────────────────────────────────────
    pseudo_data = load_synth_data("pseudo_v001_st2", subtask)
    pseudo_entries = []
    for e in pseudo_data:
        label = extract_label(e, subtask)
        if label is not None and label in label2id:
            text = build_input_text(e, track, subtask)
            pseudo_entries.append({"id": e["id"], "text": text, "text_raw": e.get("text_raw", "")})

    # ── Load test data for ID cross-check ─────────────────────────
    test_path = ws / "data" / "touchefallacy_2026_test_task.jsonl"
    test_data = load_jsonl(str(test_path))
    test_ids = {e["id"] for e in test_data}
    test_raws = {e.get("text_raw", "") for e in test_data}

    print(f"[leak_audit] dev={len(dev_entries)}, pseudo={len(pseudo_entries)}, test={len(test_data)}")

    results = {}

    # ── Check 1: ID overlap ───────────────────────────────────────
    dev_ids = {e["id"] for e in dev_entries}
    pseudo_ids = {e["id"] for e in pseudo_entries}
    train_ids = {e["id"] for e in train_split}

    id_overlap_dev_pseudo = dev_ids & pseudo_ids
    id_overlap_dev_test = dev_ids & test_ids
    id_overlap_train_test = train_ids & test_ids

    results["id_overlap"] = {
        "dev_pseudo": list(id_overlap_dev_pseudo),
        "dev_test": list(id_overlap_dev_test),
        "train_test": list(id_overlap_train_test),
        "dev_pseudo_count": len(id_overlap_dev_pseudo),
        "dev_test_count": len(id_overlap_dev_test),
        "train_test_count": len(id_overlap_train_test),
    }
    print(f"[leak_audit] ID overlap: dev∩pseudo={len(id_overlap_dev_pseudo)}, "
          f"dev∩test={len(id_overlap_dev_test)}, train∩test={len(id_overlap_train_test)}")

    # ── Check 2: Exact text_raw match ─────────────────────────────
    dev_raws = {e["text_raw"] for e in dev_entries}
    pseudo_raws = {e["text_raw"] for e in pseudo_entries}

    exact_dev_pseudo = dev_raws & pseudo_raws
    exact_dev_test = dev_raws & test_raws

    results["exact_text_overlap"] = {
        "dev_pseudo_count": len(exact_dev_pseudo),
        "dev_test_count": len(exact_dev_test),
    }
    print(f"[leak_audit] Exact text_raw match: dev∩pseudo={len(exact_dev_pseudo)}, "
          f"dev∩test={len(exact_dev_test)}")

    heartbeat.beat(step=1)

    # ── Check 3: Token Jaccard between dev and pseudo ─────────────
    if dev_entries and pseudo_entries:
        dev_tokens = [tokenize_simple(e["text"]) for e in dev_entries]
        pseudo_tokens = [tokenize_simple(e["text"]) for e in pseudo_entries]

        jaccards = []
        max_jaccard_pairs = []
        for i, dt in enumerate(dev_tokens):
            for j, pt in enumerate(pseudo_tokens):
                j_score = jaccard(dt, pt)
                jaccards.append(j_score)
                if j_score > 0.5:
                    max_jaccard_pairs.append({
                        "dev_id": dev_entries[i]["id"],
                        "pseudo_id": pseudo_entries[j]["id"],
                        "jaccard": round(j_score, 4),
                    })

        jaccards = np.array(jaccards)
        results["token_jaccard"] = {
            "min": round(float(jaccards.min()), 4),
            "max": round(float(jaccards.max()), 4),
            "mean": round(float(jaccards.mean()), 4),
            "p95": round(float(np.percentile(jaccards, 95)), 4),
            "p99": round(float(np.percentile(jaccards, 99)), 4),
            "pairs_above_0.5": len(max_jaccard_pairs),
            "high_overlap_pairs": max_jaccard_pairs[:20],
        }
        print(f"[leak_audit] Token Jaccard: mean={jaccards.mean():.4f}, "
              f"max={jaccards.max():.4f}, p95={np.percentile(jaccards, 95):.4f}")
    else:
        results["token_jaccard"] = {"note": "no dev or pseudo entries"}

    heartbeat.beat(step=2)

    # ── Check 4: Embedding cosine similarity ──────────────────────
    if not guard.exceeded() and dev_entries and pseudo_entries:
        device = get_device()
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        model = AutoModel.from_pretrained(backbone).to(device)

        dev_texts = [e["text"] for e in dev_entries]
        pseudo_texts = [e["text"] for e in pseudo_entries]

        dev_embs = encode_texts(dev_texts, tokenizer, model, device)
        pseudo_embs = encode_texts(pseudo_texts, tokenizer, model, device)

        # Cosine similarity matrix (dev x pseudo)
        cos_sim = torch.mm(dev_embs, pseudo_embs.t()).numpy()  # (n_dev, n_pseudo)

        max_per_dev = cos_sim.max(axis=1)  # for each dev entry, max similarity to any pseudo
        max_per_pseudo = cos_sim.max(axis=0)

        results["embedding_cosine"] = {
            "overall_mean": round(float(cos_sim.mean()), 4),
            "overall_max": round(float(cos_sim.max()), 4),
            "dev_max_sim_mean": round(float(max_per_dev.mean()), 4),
            "dev_max_sim_p95": round(float(np.percentile(max_per_dev, 95)), 4),
            "dev_max_sim_max": round(float(max_per_dev.max()), 4),
            "pseudo_max_sim_mean": round(float(max_per_pseudo.mean()), 4),
        }

        # Flag any suspiciously similar pairs
        suspicious = []
        for i in range(cos_sim.shape[0]):
            for j in range(cos_sim.shape[1]):
                if cos_sim[i, j] > 0.95:
                    suspicious.append({
                        "dev_id": dev_entries[i]["id"],
                        "pseudo_id": pseudo_entries[j]["id"],
                        "cosine": round(float(cos_sim[i, j]), 4),
                    })
        results["embedding_cosine"]["suspicious_pairs_above_0.95"] = suspicious[:20]
        results["embedding_cosine"]["n_suspicious"] = len(suspicious)

        print(f"[leak_audit] Embedding cosine: mean={cos_sim.mean():.4f}, "
              f"max={cos_sim.max():.4f}, suspicious(>0.95)={len(suspicious)}")

        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        results["embedding_cosine"] = {"note": "skipped (time exceeded or no data)"}

    heartbeat.beat(step=3)

    # ── Check 5: Dev split composition ────────────────────────────
    dev_labels = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            dev_labels.append(label)
    results["dev_split_info"] = {
        "n_dev": len(dev_entries),
        "n_train": len(train_split),
        "label_dist": dict(Counter(dev_labels)),
    }

    # ── Summary verdict ───────────────────────────────────────────
    leak_detected = (
        len(id_overlap_dev_pseudo) > 0
        or len(id_overlap_dev_test) > 0
        or len(exact_dev_pseudo) > 0
        or len(exact_dev_test) > 0
    )
    results["verdict"] = {
        "leak_detected": leak_detected,
        "id_leak": len(id_overlap_dev_pseudo) > 0 or len(id_overlap_dev_test) > 0,
        "text_leak": len(exact_dev_pseudo) > 0 or len(exact_dev_test) > 0,
        "summary": "LEAK FOUND — investigate before trusting dev F1" if leak_detected
                   else "No direct leak found. exp_0072 dev F1 appears legitimate."
    }
    print(f"[leak_audit] VERDICT: {results['verdict']['summary']}")

    # ── Write results ─────────────────────────────────────────────
    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("[leak_audit] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[leak_audit] CRASHED: {exc}")
        sys.exit(1)
