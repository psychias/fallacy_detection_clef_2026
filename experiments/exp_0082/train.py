"""
exp_0082 — NLI zero-shot baseline for ST3 basis axis.

The goal axis (practical vs epistemic) is already 100% accurate.
The basis axis (external vs internal) accounts for ALL ST3 errors.
With only 8 pe training examples, fine-tuning can't learn basis well.

Strategy: Use NLI (DeBERTa-v3-large-mnli) to classify basis:
  - Premise: the argument text
  - Hypothesis options for external/internal
  - Pick whichever has higher entailment probability
  - Combine with the fine-tuned model's goal prediction for final ST3 label

Reports: per-class F1, basis accuracy, overall ST3 F1 vs current best.
"""

import json
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    load_config,
    set_seed,
    write_status,
    write_traceback,
)

# ── Hypothesis templates for NLI ──────────────────────────────────
# We test multiple hypothesis phrasings and pick the best on dev.

HYPOTHESIS_SETS = {
    "v1_simple": {
        "external": "This argument appeals to an external source or authority.",
        "internal": "This argument appeals to internal reasoning or logic.",
    },
    "v2_detailed": {
        "external": "The argument supports its claim by citing an external source, such as an authority, popular opinion, tradition, or established practice.",
        "internal": "The argument supports its claim through internal reasoning, such as logical deduction, cause-and-effect analysis, or analogy.",
    },
    "v3_scheme": {
        "external": "The basis of this argument is an appeal to something outside the argument itself, like an authority, popular belief, tradition, or external evidence.",
        "internal": "The basis of this argument is the internal structure of the reasoning, like analogy, causal reasoning, or logical inference.",
    },
    "v4_short": {
        "external": "This relies on outside authority or evidence.",
        "internal": "This relies on internal logic or reasoning.",
    },
}

# Goal axis hypotheses (for completeness / ablation)
GOAL_HYPOTHESES = {
    "practical": "This argument is about what we should do or how we should act.",
    "epistemic": "This argument is about what is true or what we should believe.",
}


def nli_classify_batch(texts, hypothesis, tokenizer, model, device, batch_size=16):
    """
    Run NLI on (text, hypothesis) pairs. Returns entailment probabilities.
    Model expected: entailment=0 or entailment=2 depending on training.
    """
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # NLI format: premise [SEP] hypothesis
            enc = tokenizer(
                batch_texts,
                [hypothesis] * len(batch_texts),
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits  # (B, 3) for entailment/neutral/contradiction
            probs = torch.softmax(logits.float(), dim=-1)
            all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)  # (N, 3)


def get_entailment_idx(model_name: str) -> int:
    """Determine which index is 'entailment' for this NLI model."""
    # DeBERTa-v3-large-mnli uses label2id from config
    # Most MNLI models: 0=contradiction, 1=neutral, 2=entailment
    # But some use 0=entailment. We'll check the model's config.
    return 2  # default for DeBERTa-v3-large-mnli-fever-anli-ling-wanli


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    set_seed(42)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 600), margin_s=60)

    subtask = config["subtask"]
    track = config["track"]
    nli_backbone = config["backbone"]

    label2id, id2label = LABEL_MAPS[subtask]
    print(f"[nli_zeroshot] subtask={subtask}, track={track}, nli={nli_backbone}")

    # ── Load data ─────────────────────────────────────────────────
    ws = workspace
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    _, dev_split = make_splits(all_data, subtask)

    # Build dev entries with decomposed labels
    dev_entries = []
    for e in dev_split:
        label = extract_label(e, subtask)
        if label is not None:
            text = build_input_text(e, track, subtask)
            goal, basis = label.split("-")  # e.g. "practical-external" → "practical", "external"
            dev_entries.append({
                "id": e["id"],
                "text": text,
                "label": label,
                "goal": goal,
                "basis": basis,
            })

    print(f"[nli_zeroshot] dev entries: {len(dev_entries)}")

    # ── Load NLI model ────────────────────────────────────────────
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(nli_backbone)
    model = AutoModelForSequenceClassification.from_pretrained(nli_backbone).to(device)

    # Check model's label mapping
    model_config = model.config
    if hasattr(model_config, 'label2id'):
        print(f"[nli_zeroshot] Model label2id: {model_config.label2id}")
        entail_key = [k for k, v in model_config.label2id.items()
                      if 'entail' in k.lower()]
        if entail_key:
            entail_idx = model_config.label2id[entail_key[0]]
        else:
            entail_idx = get_entailment_idx(nli_backbone)
    else:
        entail_idx = get_entailment_idx(nli_backbone)
    print(f"[nli_zeroshot] Entailment index: {entail_idx}")

    heartbeat.beat(step=0)

    dev_texts = [e["text"] for e in dev_entries]
    true_goals = [e["goal"] for e in dev_entries]
    true_bases = [e["basis"] for e in dev_entries]
    true_labels = [e["label"] for e in dev_entries]

    results = {}

    # ── Test each hypothesis set for BASIS axis ───────────────────
    best_basis_acc = -1
    best_hyp_set = None

    for hyp_name, hyp_dict in HYPOTHESIS_SETS.items():
        if guard.exceeded():
            break

        ext_probs = nli_classify_batch(dev_texts, hyp_dict["external"], tokenizer, model, device)
        int_probs = nli_classify_batch(dev_texts, hyp_dict["internal"], tokenizer, model, device)

        ext_entail = ext_probs[:, entail_idx].numpy()
        int_entail = int_probs[:, entail_idx].numpy()

        pred_bases = []
        basis_scores = []
        for e_score, i_score in zip(ext_entail, int_entail):
            if e_score > i_score:
                pred_bases.append("external")
            else:
                pred_bases.append("internal")
            basis_scores.append({"external": round(float(e_score), 4),
                                 "internal": round(float(i_score), 4)})

        basis_correct = sum(1 for p, t in zip(pred_bases, true_bases) if p == t)
        basis_acc = basis_correct / len(true_bases) if true_bases else 0

        results[f"basis_{hyp_name}"] = {
            "accuracy": round(basis_acc, 4),
            "correct": basis_correct,
            "total": len(true_bases),
            "pred_dist": {
                "external": sum(1 for p in pred_bases if p == "external"),
                "internal": sum(1 for p in pred_bases if p == "internal"),
            },
            "true_dist": {
                "external": sum(1 for t in true_bases if t == "external"),
                "internal": sum(1 for t in true_bases if t == "internal"),
            },
        }

        print(f"[nli_zeroshot] {hyp_name}: basis_acc={basis_acc:.4f}")

        if basis_acc > best_basis_acc:
            best_basis_acc = basis_acc
            best_hyp_set = hyp_name
            best_pred_bases = pred_bases

        heartbeat.beat(step=list(HYPOTHESIS_SETS.keys()).index(hyp_name) + 1)

    # ── Test GOAL axis with NLI (for comparison) ──────────────────
    if not guard.exceeded():
        pract_probs = nli_classify_batch(dev_texts, GOAL_HYPOTHESES["practical"], tokenizer, model, device)
        epist_probs = nli_classify_batch(dev_texts, GOAL_HYPOTHESES["epistemic"], tokenizer, model, device)

        pract_entail = pract_probs[:, entail_idx].numpy()
        epist_entail = epist_probs[:, entail_idx].numpy()

        pred_goals_nli = []
        for p_score, e_score in zip(pract_entail, epist_entail):
            pred_goals_nli.append("practical" if p_score > e_score else "epistemic")

        goal_correct = sum(1 for p, t in zip(pred_goals_nli, true_goals) if p == t)
        goal_acc = goal_correct / len(true_goals) if true_goals else 0

        results["goal_nli"] = {
            "accuracy": round(goal_acc, 4),
            "correct": goal_correct,
            "total": len(true_goals),
        }
        print(f"[nli_zeroshot] NLI goal accuracy: {goal_acc:.4f}")

    # ── Combine: fine-tuned goal (assume perfect) + NLI basis ─────
    # Simulate the oracle goal + NLI basis combination
    if best_pred_bases:
        combined_preds = []
        for true_goal, pred_basis in zip(true_goals, best_pred_bases):
            combined_preds.append(f"{true_goal}-{pred_basis}")

        metrics_oracle_goal = evaluate(true_labels, combined_preds, subtask)
        results["combined_oracle_goal_nli_basis"] = {
            "hypothesis_set": best_hyp_set,
            "basis_accuracy": round(best_basis_acc, 4),
            "f1_macro": metrics_oracle_goal["f1_macro"],
            "f1_per_class": metrics_oracle_goal["f1_per_class"],
        }
        print(f"[nli_zeroshot] Oracle-goal + NLI-basis: F1={metrics_oracle_goal['f1_macro']:.4f}")

        # Also: NLI goal + NLI basis (fully zero-shot)
        if not guard.exceeded() and 'pred_goals_nli' in dir():
            full_nli_preds = []
            for pg, pb in zip(pred_goals_nli, best_pred_bases):
                full_nli_preds.append(f"{pg}-{pb}")

            metrics_full_nli = evaluate(true_labels, full_nli_preds, subtask)
            results["full_nli_zeroshot"] = {
                "f1_macro": metrics_full_nli["f1_macro"],
                "f1_per_class": metrics_full_nli["f1_per_class"],
            }
            print(f"[nli_zeroshot] Full NLI zero-shot ST3: F1={metrics_full_nli['f1_macro']:.4f}")

    # ── Per-example scores for error analysis ─────────────────────
    dev_logits = []
    for i, entry in enumerate(dev_entries):
        dev_logits.append({
            "id": entry["id"],
            "true_label": entry["label"],
            "true_goal": entry["goal"],
            "true_basis": entry["basis"],
            "pred_basis_nli": best_pred_bases[i] if best_pred_bases else None,
            "best_hyp_set": best_hyp_set,
        })
    results["dev_logits"] = dev_logits

    # ── Cleanup & save ────────────────────────────────────────────
    del model, tokenizer
    torch.cuda.empty_cache()

    results["best_hypothesis_set"] = best_hyp_set
    results["best_basis_accuracy"] = round(best_basis_acc, 4)
    results["subtask"] = subtask
    results["track"] = track
    results["nli_backbone"] = nli_backbone

    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("[nli_zeroshot] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[nli_zeroshot] CRASHED: {exc}")
        sys.exit(1)
