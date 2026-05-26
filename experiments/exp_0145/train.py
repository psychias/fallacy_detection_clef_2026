"""
exp_0145 -- Per-class threshold sweep on ST3 dev probabilities.

Uses calibrated dev probs from exp_0121 (which loaded exp_0073 checkpoint).
Sweeps per-class thresholds independently for each ST3 class, with special
focus on practical-external (pe) which is stuck at F1=0 under argmax.

Strategy:
1. Load calibrated dev probs from exp_0121/metrics.json
2. For each class, sweep threshold from 0.05 to 0.50
3. Find the threshold combo that maximizes macro-F1 (greedy per-class)
4. Also try: if max_prob < global_confidence_threshold, fall back to
   class with lowest threshold (pe-biased)
"""

import json
import sys
import itertools
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

import numpy as np

from shared.data_utils import LABEL_MAPS
from shared.eval import evaluate, write_metrics
from shared.train_utils import (
    Heartbeat,
    WallClockGuard,
    load_config,
    write_status,
    write_traceback,
)


ST3_CLASSES = ["epistemic-external", "epistemic-internal",
               "practical-external", "practical-internal"]


def predict_with_thresholds(probs_list, thresholds, class_names):
    """
    Predict using per-class thresholds. For each example:
    1. Check if any class exceeds its threshold
    2. If multiple do, pick the one with highest prob/threshold ratio
    3. If none do, fall back to argmax
    """
    preds = []
    for probs in probs_list:
        # Compute how far above threshold each class is (relative)
        ratios = [probs[i] / thresholds[i] for i in range(len(class_names))]
        # Classes above their threshold
        above = [i for i in range(len(class_names)) if probs[i] >= thresholds[i]]

        if above:
            # Pick highest ratio
            best = max(above, key=lambda i: ratios[i])
        else:
            # Fallback: argmax
            best = int(np.argmax(probs))

        preds.append(class_names[best])
    return preds


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 300), margin_s=30)

    subtask = config["subtask"]
    calibrated_source = config["calibrated_source"]
    label2id, id2label = LABEL_MAPS[subtask]

    # Load calibrated dev probs from exp_0121
    cal_metrics = json.loads(
        (workspace / "experiments" / calibrated_source / "metrics.json")
        .read_text(encoding="utf-8-sig")
    )
    dev_cal = cal_metrics["st3"]["dev_calibrated"]
    print(f"[thresh] Loaded {len(dev_cal)} calibrated dev predictions from {calibrated_source}")

    probs_list = [d["avg_probs"] for d in dev_cal]
    true_labels = [d["true_label"] for d in dev_cal]
    ids = [d["id"] for d in dev_cal]

    # Show class distribution
    from collections import Counter
    dist = Counter(true_labels)
    print(f"[thresh] Class distribution: {dict(dist)}")

    # ── Analysis: what does the model think about pe examples? ────
    pe_idx = ST3_CLASSES.index("practical-external")
    pe_examples = [(i, p, t) for i, (p, t) in enumerate(zip(probs_list, true_labels))
                   if t == "practical-external"]
    print(f"\n[thresh] practical-external analysis ({len(pe_examples)} examples):")
    for idx, probs, _ in pe_examples:
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])
        ranks_str = " > ".join(f"{ST3_CLASSES[c]}:{p:.3f}" for c, p in ranked)
        print(f"  {ids[idx]}: {ranks_str}")
        print(f"    pe rank: {[c for c, _ in ranked].index(pe_idx) + 1}, pe prob: {probs[pe_idx]:.4f}")

    heartbeat.beat(step=0)

    # ── Strategy 1: Greedy per-class threshold sweep ──────────────
    print("\n[thresh] Strategy 1: Greedy per-class sweep")
    threshold_range = np.arange(0.05, 0.55, 0.025)
    best_thresholds = [0.25] * len(ST3_CLASSES)  # default = argmax-like
    best_f1 = -1

    # Sweep each class independently (greedy)
    for class_idx in range(len(ST3_CLASSES)):
        best_t = best_thresholds[class_idx]
        for t in threshold_range:
            trial = best_thresholds.copy()
            trial[class_idx] = float(t)
            preds = predict_with_thresholds(probs_list, trial, ST3_CLASSES)
            metrics = evaluate(true_labels, preds, subtask)
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                best_t = float(t)
        best_thresholds[class_idx] = best_t
        print(f"  {ST3_CLASSES[class_idx]}: best_t={best_t:.3f}, running_F1={best_f1:.5f}")

    heartbeat.beat(step=1)

    # ── Strategy 2: Grid search pe threshold specifically ─────────
    print("\n[thresh] Strategy 2: Focused pe threshold grid")
    pe_thresholds = np.arange(0.01, 0.30, 0.005)
    pe_results = []
    for pe_t in pe_thresholds:
        trial = [0.25, 0.25, float(pe_t), 0.25]
        preds = predict_with_thresholds(probs_list, trial, ST3_CLASSES)
        metrics = evaluate(true_labels, preds, subtask)
        pe_results.append({
            "pe_threshold": round(float(pe_t), 4),
            "f1_macro": metrics["f1_macro"],
            "f1_per_class": metrics["f1_per_class"],
        })
        if metrics["f1_per_class"].get("practical-external", 0) > 0:
            print(f"  pe_t={pe_t:.3f}: macro-F1={metrics['f1_macro']:.4f}, "
                  f"pe-F1={metrics['f1_per_class']['practical-external']:.4f}")

    best_pe_result = max(pe_results, key=lambda r: r["f1_macro"])
    print(f"  Best pe-only sweep: t={best_pe_result['pe_threshold']}, "
          f"F1={best_pe_result['f1_macro']:.5f}")

    heartbeat.beat(step=2)

    # ── Strategy 3: Exhaustive small grid on all 4 classes ────────
    print("\n[thresh] Strategy 3: Small exhaustive grid")
    grid_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    pe_grid = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    best_grid_f1 = -1
    best_grid_thresholds = None

    for ee_t in grid_vals:
        for ei_t in grid_vals:
            for pe_t in pe_grid:
                for pi_t in grid_vals:
                    trial = [ee_t, ei_t, pe_t, pi_t]
                    preds = predict_with_thresholds(probs_list, trial, ST3_CLASSES)
                    metrics = evaluate(true_labels, preds, subtask)
                    if metrics["f1_macro"] > best_grid_f1:
                        best_grid_f1 = metrics["f1_macro"]
                        best_grid_thresholds = trial.copy()

    print(f"  Best grid: thresholds={best_grid_thresholds}, F1={best_grid_f1:.5f}")

    # ── Baseline: argmax ──────────────────────────────────────────
    argmax_preds = [ST3_CLASSES[int(np.argmax(p))] for p in probs_list]
    argmax_metrics = evaluate(true_labels, argmax_preds, subtask)
    print(f"\n[thresh] Baseline argmax: F1={argmax_metrics['f1_macro']:.5f}")
    print(f"  per-class: {argmax_metrics['f1_per_class']}")

    # ── Choose best strategy ──────────────────────────────────────
    strategies = [
        ("greedy", best_thresholds, best_f1),
        ("pe_only", [0.25, 0.25, best_pe_result["pe_threshold"], 0.25],
         best_pe_result["f1_macro"]),
        ("grid", best_grid_thresholds, best_grid_f1),
        ("argmax", [0.25, 0.25, 0.25, 0.25], argmax_metrics["f1_macro"]),
    ]
    winner = max(strategies, key=lambda s: s[2])
    print(f"\n[thresh] WINNER: {winner[0]} with F1={winner[2]:.5f}")
    print(f"  thresholds: {dict(zip(ST3_CLASSES, winner[1]))}")

    # Final evaluation with winner thresholds
    final_preds = predict_with_thresholds(probs_list, winner[1], ST3_CLASSES)
    final_metrics = evaluate(true_labels, final_preds, subtask)

    # ── Write results ─────────────────────────────────────────────
    results = {
        "subtask": subtask,
        "source_exp": config["source_exp"],
        "calibrated_source": calibrated_source,
        "best_strategy": winner[0],
        "best_thresholds": dict(zip(ST3_CLASSES, winner[1])),
        "best_f1_macro": winner[2],
        "argmax_f1_macro": argmax_metrics["f1_macro"],
        "improvement": round(winner[2] - argmax_metrics["f1_macro"], 5),
        "final_f1_per_class": final_metrics["f1_per_class"],
        "strategies": {s[0]: {"thresholds": dict(zip(ST3_CLASSES, s[1])), "f1_macro": s[2]}
                       for s in strategies},
        "pe_analysis": {
            "n_pe_examples": len(pe_examples),
            "pe_sweep_best": best_pe_result,
        },
        "wall_clock_s": round(guard.elapsed(), 1),
    }

    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("[thresh] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[thresh] CRASHED: {exc}")
        sys.exit(1)
