"""Generate stratified 5-fold splits for ST1 and ST2."""
import json, sys, random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_utils import load_jsonl, extract_label, filter_for_subtask

data = load_jsonl("data/touchefallacy_2026_train.jsonl")

for subtask in ["st1", "st2"]:
    filtered = filter_for_subtask(data, subtask)
    groups = {}
    for e in filtered:
        label = extract_label(e, subtask)
        if label is not None:
            groups.setdefault(label, []).append(e["id"])

    n_folds = 5
    seed = 42
    rng = random.Random(seed)
    for label in sorted(groups):
        rng.shuffle(groups[label])

    fold_dev_ids = [[] for _ in range(n_folds)]
    for label in sorted(groups):
        ids = groups[label]
        for i, eid in enumerate(ids):
            fold_dev_ids[i % n_folds].append(eid)

    all_ids = set()
    for label in groups:
        all_ids.update(groups[label])

    folds = []
    for f in range(n_folds):
        dev_ids = set(fold_dev_ids[f])
        train_ids = all_ids - dev_ids
        folds.append({
            "fold": f,
            "dev_ids": sorted(dev_ids),
            "train_ids": sorted(train_ids),
        })

    result = {"subtask": subtask, "n_folds": n_folds, "seed": seed, "folds": folds}
    out_path = f"shared/kfold_splits_{subtask}.json"
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)

    for fold in folds:
        dev_set = set(fold["dev_ids"])
        dev_labels = Counter()
        for e in filtered:
            if e["id"] in dev_set:
                dev_labels[extract_label(e, subtask)] += 1
        print(f"{subtask} fold {fold['fold']}: dev={len(fold['dev_ids'])} "
              f"train={len(fold['train_ids'])} dev_dist={dict(dev_labels)}")
    print()
