"""
exp_0124 — Build stratified 5-fold CV splitter for ST3 with pe-aware distribution.

Problem: Standard dev split has only 1 pe (practical-external) example, making
pe F1 measurement binary (0 or 1). This creates a splitter that distributes
all pe examples across folds so each fold has at least 1 pe in its dev portion.

Outputs: shared/kfold_splits_st3.json with fold definitions.
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

from shared.data_utils import (
    LABEL_MAPS,
    extract_label,
    filter_for_subtask,
    load_jsonl,
)
from shared.eval import write_metrics
from shared.train_utils import load_config, write_status, write_traceback


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    write_status(exp_dir, "running")

    ws = workspace
    n_folds = config.get("n_folds", 5)
    seed = config.get("seed", 42)
    subtask = config["subtask"]

    label2id, id2label = LABEL_MAPS[subtask]

    # Load all data
    train_path = ws / "data" / "touchefallacy_2026_train.jsonl"
    all_data = load_jsonl(str(train_path))
    filtered = filter_for_subtask(all_data, subtask)

    # Attach labels
    entries_with_labels = []
    for e in filtered:
        label = extract_label(e, subtask)
        if label is not None:
            entries_with_labels.append({"id": e["id"], "label": label})

    print(f"[kfold] total entries: {len(entries_with_labels)}")
    print(f"[kfold] label distribution: {Counter(e['label'] for e in entries_with_labels)}")

    # Group by label
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for e in entries_with_labels:
        by_label[e["label"]].append(e["id"])

    # Shuffle within each label group
    for label in by_label:
        rng.shuffle(by_label[label])

    # Build folds: deal entries round-robin within each label group
    # This ensures each fold has approximately proportional representation
    # and pe examples are distributed (>=1 per fold if possible)
    folds = [[] for _ in range(n_folds)]

    for label in sorted(by_label.keys()):
        ids = by_label[label]
        for i, eid in enumerate(ids):
            folds[i % n_folds].append(eid)

    # Shuffle each fold
    for fold in folds:
        rng.shuffle(fold)

    # Build train/dev splits for each fold
    fold_splits = []
    for fold_idx in range(n_folds):
        dev_ids = set(folds[fold_idx])
        train_ids = set()
        for j in range(n_folds):
            if j != fold_idx:
                train_ids.update(folds[j])

        fold_splits.append({
            "fold": fold_idx,
            "dev_ids": sorted(dev_ids),
            "train_ids": sorted(train_ids),
            "n_dev": len(dev_ids),
            "n_train": len(train_ids),
        })

    # Report per-fold label distributions
    id_to_label = {e["id"]: e["label"] for e in entries_with_labels}
    fold_stats = []
    for fs in fold_splits:
        dev_labels = Counter(id_to_label[eid] for eid in fs["dev_ids"])
        train_labels = Counter(id_to_label[eid] for eid in fs["train_ids"])
        fold_stats.append({
            "fold": fs["fold"],
            "n_dev": fs["n_dev"],
            "n_train": fs["n_train"],
            "dev_labels": dict(dev_labels),
            "train_labels": dict(train_labels),
            "pe_in_dev": dev_labels.get("practical-external", 0),
            "pe_in_train": train_labels.get("practical-external", 0),
        })
        print(f"[kfold] fold {fs['fold']}: dev={fs['n_dev']} "
              f"(pe={dev_labels.get('practical-external', 0)}), "
              f"train={fs['n_train']}")

    # Save fold definitions
    output = {
        "subtask": subtask,
        "n_folds": n_folds,
        "seed": seed,
        "folds": fold_splits,
    }
    output_path = ws / "shared" / "kfold_splits_st3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[kfold] Saved to {output_path}")

    # Metrics
    results = {
        "subtask": subtask,
        "n_folds": n_folds,
        "n_total": len(entries_with_labels),
        "label_distribution": dict(Counter(e["label"] for e in entries_with_labels)),
        "fold_stats": fold_stats,
        "output_path": str(output_path),
    }

    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("[kfold] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[kfold] CRASHED: {exc}")
        sys.exit(1)
