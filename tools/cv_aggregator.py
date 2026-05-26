"""
CV Results Aggregator — compiles all CV experiment results into a comparison table.
Computes paired fold-by-fold differences against baseline.
Usage: python tools/cv_aggregator.py
"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict

ws = Path("G:/My Drive/fallacy_detection")

# ── Define experiments to compare ────────────────────────────────────
# Each entry: (exp_id, label, subtask, metric_extractor)
# metric_extractor: function(metrics_dict) -> (macro_f1_per_fold, pe_f1_per_fold_or_None)

def extract_st3_cv(m):
    """Standard ST3 CV metrics: macro + pe per fold."""
    macro = m.get("macro_f1_per_fold", [])
    pe = m.get("pe_f1_per_fold", [])
    per_class = None
    if "fold_details" in m:
        per_class = {}
        for fold_detail in m["fold_details"]:
            for cls, f1 in fold_detail.get("f1_per_class", {}).items():
                per_class.setdefault(cls, []).append(f1)
    return macro, pe, per_class


def extract_st2_cv(m):
    """ST2 CV metrics: macro per fold + per-class."""
    macro = m.get("macro_f1_per_fold", [])
    per_class = {}
    if "per_class_f1" in m:
        for cls, info in m["per_class_f1"].items():
            per_class[cls] = info.get("per_fold", [])
    elif "fold_details" in m:
        for fd in m["fold_details"]:
            for cls, f1 in fd.get("f1_per_class", {}).items():
                per_class.setdefault(cls, []).append(f1)
    return macro, None, per_class


def extract_st1_cv(m):
    """ST1 CV metrics: argmax + threshold variants."""
    macro = m.get("argmax_f1_per_fold", m.get("macro_f1_per_fold", []))
    per_class = {}
    if "fold_details" in m:
        for fd in m["fold_details"]:
            for cls, f1 in fd.get("f1_per_class", {}).items():
                per_class.setdefault(cls, []).append(f1)
    # Also extract threshold-tuned variants if present
    transfer = m.get("transfer_f1_per_fold", [])
    return macro, transfer if transfer else None, per_class


# ── Experiment registry ──────────────────────────────────────────────
EXPERIMENTS = OrderedDict()

# ST3 experiments
st3_exps = [
    ("exp_0153", "Baseline (no synth)", extract_st3_cv),
    ("exp_0151", "+ pe synth enh", extract_st3_cv),
    ("exp_0152", "+ pe synth base", extract_st3_cv),
    ("exp_0156", "+ logit adjustment", extract_st3_cv),
    ("exp_0159", "+ aux head", extract_st3_cv),
    ("exp_0160", "+ oversampling 10x", extract_st3_cv),
]

# ST2 experiments
st2_exps = [
    ("exp_0170", "ST2 enh (0072 config)", extract_st2_cv),
]

# ST1 experiments
st1_exps = [
    ("exp_0171", "ST1 enh (0035 config)", extract_st1_cv),
]


def load_metrics(exp_id):
    path = ws / "experiments" / exp_id / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def paired_comparison(baseline_folds, intervention_folds):
    """Compute paired fold-by-fold differences and win rate."""
    if len(baseline_folds) != len(intervention_folds):
        return None
    diffs = [i - b for b, i in zip(baseline_folds, intervention_folds)]
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    losses = sum(1 for d in diffs if d < 0)
    return {
        "diffs": [round(d, 5) for d in diffs],
        "mean_diff": round(float(np.mean(diffs)), 5),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "record": f"{wins}W-{ties}T-{losses}L",
    }


def print_table(title, experiments, extractor_list, baseline_id=None):
    """Print comparison table for a group of experiments."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    results = {}
    for exp_id, label, extractor in extractor_list:
        m = load_metrics(exp_id)
        if m is None:
            print(f"  {exp_id} ({label}): NOT DONE YET")
            continue
        macro_folds, secondary_folds, per_class = extractor(m)
        results[exp_id] = {
            "label": label,
            "macro_folds": macro_folds,
            "secondary_folds": secondary_folds,
            "per_class": per_class,
        }

    if not results:
        print("  No results available yet.")
        return

    # Main comparison table
    print(f"\n  {'Approach':<30} {'CV Macro F1':>12} {'Std':>8} {'Worst':>8} ", end="")
    # Check if we have PE (ST3)
    has_pe = any(r["secondary_folds"] is not None and len(r.get("secondary_folds", [])) > 0
                 for r in results.values())
    if has_pe:
        print(f"{'PE F1':>10} {'PE Std':>8} {'PE Worst':>10}", end="")
    print()
    print(f"  {'-'*28}   {'-'*10}   {'-'*6}   {'-'*6}   ", end="")
    if has_pe:
        print(f"{'-'*8}   {'-'*6}   {'-'*8}", end="")
    print()

    for exp_id, r in results.items():
        macro = r["macro_folds"]
        if not macro:
            continue
        label = r["label"]
        mean_m = np.mean(macro)
        std_m = np.std(macro)
        worst_m = min(macro)

        print(f"  {label:<30} {mean_m:>12.4f} {std_m:>8.4f} {worst_m:>8.4f} ", end="")

        if has_pe and r["secondary_folds"] is not None and len(r["secondary_folds"]) > 0:
            pe = r["secondary_folds"]
            mean_pe = np.mean(pe)
            std_pe = np.std(pe)
            worst_pe = min(pe)
            print(f"{mean_pe:>10.4f} {std_pe:>8.4f} {worst_pe:>10.4f}", end="")
        elif has_pe:
            print(f"{'N/A':>10} {'N/A':>8} {'N/A':>10}", end="")
        print()

    # Paired comparisons vs baseline
    if baseline_id and baseline_id in results:
        base = results[baseline_id]
        base_macro = base["macro_folds"]
        base_pe = base["secondary_folds"]

        print(f"\n  Paired fold-by-fold comparison vs {base['label']}:")
        print(f"  {'Approach':<30} {'Mean Diff':>10} {'Record':>10} {'Per-fold diffs'}")
        print(f"  {'-'*28}   {'-'*8}   {'-'*8}   {'-'*40}")

        for exp_id, r in results.items():
            if exp_id == baseline_id:
                continue
            if not r["macro_folds"] or len(r["macro_folds"]) != len(base_macro):
                continue

            pc = paired_comparison(base_macro, r["macro_folds"])
            print(f"  {r['label']:<30} {pc['mean_diff']:>+10.4f} {pc['record']:>10} "
                  f"{pc['diffs']}")

            # PE paired comparison if available
            if base_pe and r["secondary_folds"] and len(r["secondary_folds"]) == len(base_pe):
                pc_pe = paired_comparison(base_pe, r["secondary_folds"])
                print(f"    {'(PE F1)':<28} {pc_pe['mean_diff']:>+10.4f} {pc_pe['record']:>10} "
                      f"{pc_pe['diffs']}")

    # Per-class detail for the best non-baseline if available
    print(f"\n  Per-fold details:")
    for exp_id, r in results.items():
        if not r["macro_folds"]:
            continue
        macro = r["macro_folds"]
        print(f"  {r['label']}: macro_per_fold={[round(f,4) for f in macro]}", end="")
        if r["secondary_folds"]:
            print(f"  secondary_per_fold={[round(f,4) for f in r['secondary_folds']]}", end="")
        print()


def main():
    print("CV RESULTS AGGREGATOR")
    print(f"Generated: {__import__('datetime').datetime.now().isoformat()}")

    # ST3 Enhanced comparison
    print_table(
        "ST3 Enhanced — Intervention Comparison",
        st3_exps,
        st3_exps,
        baseline_id="exp_0153",
    )

    # ST2 Enhanced validation
    print_table(
        "ST2 Enhanced — Flagship Validation",
        st2_exps,
        st2_exps,
    )

    # ST1 Enhanced validation
    print_table(
        "ST1 Enhanced — Flagship Validation",
        st1_exps,
        st1_exps,
    )

    # Summary: overall queue status
    print(f"\n{'='*90}")
    print(f"  QUEUE STATUS")
    print(f"{'='*90}")
    all_exps = [e[0] for e in st3_exps + st2_exps + st1_exps]
    for exp_id in all_exps:
        status_path = ws / "experiments" / exp_id / "status.json"
        if status_path.exists():
            with open(status_path, encoding="utf-8-sig") as f:
                text = f.read().strip()
                state = json.loads(text).get("state", "unknown") if text else "empty"
        else:
            state = "missing"
        has_metrics = (ws / "experiments" / exp_id / "metrics.json").exists()
        print(f"  {exp_id}: {state}  {'[HAS METRICS]' if has_metrics else ''}")


if __name__ == "__main__":
    main()
