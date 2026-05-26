"""
exp_0146 -- LLM basis-axis meta-learner for ST3.

Combines fine-tuned encoder logits (from exp_0073 via exp_0121 calibrated probs)
with Qwen-3B basis-axis predictions (from exp_0091) to train a logistic regression
meta-classifier on dev data.

The key insight: RoBERTa is perfect on goal axis but 0% on pe class.
Qwen-3B achieves 85.5% basis accuracy but weaker on goal. These are
textbook-complementary signals for ensemble stacking.

Features for meta-learner:
  - 4 calibrated class probs from encoder (ee, ei, pe, pi)
  - 1 binary LLM basis prediction (0=internal, 1=external)
  - 1 binary LLM goal prediction (0=epistemic, 1=practical)

Uses leave-one-out cross-validation since dev set is only 69 examples.
"""

import json
import sys
from collections import Counter
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


def extract_goal_basis(label):
    """Extract goal (practical/epistemic) and basis (internal/external) from ST3 label."""
    parts = label.split("-")
    return parts[0], parts[1]  # goal, basis


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)

    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)
    guard = WallClockGuard(config.get("time_budget_s", 300), margin_s=30)

    calibrated_source = config["calibrated_source"]
    llm_source = config["llm_source"]

    # Load calibrated encoder probs
    cal_metrics = json.loads(
        (workspace / "experiments" / calibrated_source / "metrics.json")
        .read_text(encoding="utf-8-sig")
    )
    dev_cal = cal_metrics["st3"]["dev_calibrated"]
    print(f"[meta] Loaded {len(dev_cal)} calibrated dev probs from {calibrated_source}")

    # Load LLM predictions
    llm_metrics = json.loads(
        (workspace / "experiments" / llm_source / "metrics.json")
        .read_text(encoding="utf-8-sig")
    )
    llm_details = llm_metrics["st3_details"]
    print(f"[meta] Loaded {len(llm_details)} LLM ST3 predictions from {llm_source}")

    # Also load basis-only predictions if available
    llm_basis = llm_metrics.get("st3_basis_only", {})
    print(f"[meta] LLM basis-only accuracy: {llm_basis.get('accuracy', 'N/A')}")

    # Align by ID
    llm_by_id = {d["id"]: d for d in llm_details}
    aligned = []
    for cal in dev_cal:
        eid = cal["id"]
        if eid in llm_by_id:
            aligned.append((cal, llm_by_id[eid]))
        else:
            print(f"  Warning: ID {eid} not in LLM predictions")

    print(f"[meta] Aligned: {len(aligned)} examples")

    # Build feature matrix
    X = []
    y = []
    ids = []
    for cal, llm in aligned:
        encoder_probs = cal["avg_probs"]  # 4 floats (ee, ei, pe, pi)
        llm_pred = llm["parsed_pred"]

        if llm_pred is None:
            llm_goal_feat = 0.5
            llm_basis_feat = 0.5
        else:
            llm_goal, llm_basis = extract_goal_basis(llm_pred)
            llm_goal_feat = 1.0 if llm_goal == "practical" else 0.0
            llm_basis_feat = 1.0 if llm_basis == "external" else 0.0

        features = encoder_probs + [llm_goal_feat, llm_basis_feat]
        X.append(features)

        true_label = cal["true_label"]
        y.append(ST3_CLASSES.index(true_label))
        ids.append(cal["id"])

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    print(f"[meta] Features shape: {X.shape}, classes: {Counter(y)}")

    heartbeat.beat(step=0)

    # ── Strategy 1: Logistic Regression with LOO-CV ───────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    loo_preds = []
    loo_probs = []
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            multi_class="multinomial", solver="lbfgs"
        )
        clf.fit(X_train_s, y_train)
        loo_preds.append(clf.predict(X_test_s)[0])
        loo_probs.append(clf.predict_proba(X_test_s)[0].tolist())

    loo_labels = [ST3_CLASSES[p] for p in loo_preds]
    true_labels = [ST3_CLASSES[yi] for yi in y]
    logreg_metrics = evaluate(true_labels, loo_labels, "st3")
    print(f"\n[meta] Strategy 1 (LogReg LOO): F1={logreg_metrics['f1_macro']:.5f}")
    print(f"  per-class: {logreg_metrics['f1_per_class']}")

    heartbeat.beat(step=1)

    # ── Strategy 2: Simple rule-based routing ─────────────────────
    # If encoder predicts external-basis class (ee or pe) AND LLM says external,
    # use LLM full prediction; otherwise use encoder
    route_preds = []
    for i, (cal, llm) in enumerate(aligned):
        encoder_pred = cal["pred_label"]
        encoder_probs_i = cal["avg_probs"]
        llm_pred = llm["parsed_pred"]

        enc_goal, enc_basis = extract_goal_basis(encoder_pred)

        if llm_pred is not None:
            llm_goal, llm_basis = extract_goal_basis(llm_pred)
        else:
            llm_goal, llm_basis = None, None

        # Route: if encoder says external-basis, defer to LLM for
        # basis decision, keep encoder's goal
        if enc_basis == "external" and llm_pred is not None:
            route_preds.append(f"{enc_goal}-{llm_basis}")
        elif llm_basis == "external" and encoder_probs_i[ST3_CLASSES.index("practical-external")] > 0.01:
            # LLM thinks external and encoder gives non-zero pe probability
            route_preds.append(llm_pred)
        else:
            route_preds.append(encoder_pred)

    route_metrics = evaluate(true_labels, route_preds, "st3")
    print(f"\n[meta] Strategy 2 (rule routing): F1={route_metrics['f1_macro']:.5f}")
    print(f"  per-class: {route_metrics['f1_per_class']}")

    # ── Strategy 3: Weighted average of encoder probs + LLM signal ─
    # Convert LLM prediction to soft distribution and average
    llm_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
    best_wavg_f1 = -1
    best_wavg_w = 0
    best_wavg_preds = None

    for w in llm_weights:
        wavg_preds = []
        for i, (cal, llm) in enumerate(aligned):
            enc_probs = np.array(cal["avg_probs"])
            llm_pred = llm["parsed_pred"]

            if llm_pred is not None and llm_pred in ST3_CLASSES:
                llm_onehot = np.zeros(4)
                llm_onehot[ST3_CLASSES.index(llm_pred)] = 1.0
            else:
                llm_onehot = np.ones(4) / 4.0

            combined = (1 - w) * enc_probs + w * llm_onehot
            wavg_preds.append(ST3_CLASSES[int(np.argmax(combined))])

        wm = evaluate(true_labels, wavg_preds, "st3")
        if wm["f1_macro"] > best_wavg_f1:
            best_wavg_f1 = wm["f1_macro"]
            best_wavg_w = w
            best_wavg_preds = wavg_preds

    wavg_metrics = evaluate(true_labels, best_wavg_preds, "st3")
    print(f"\n[meta] Strategy 3 (weighted avg, w={best_wavg_w}): F1={wavg_metrics['f1_macro']:.5f}")
    print(f"  per-class: {wavg_metrics['f1_per_class']}")

    heartbeat.beat(step=2)

    # ── Baseline: encoder argmax only ─────────────────────────────
    baseline_preds = [cal["pred_label"] for cal, _ in aligned]
    baseline_metrics = evaluate(true_labels, baseline_preds, "st3")
    print(f"\n[meta] Baseline (encoder only): F1={baseline_metrics['f1_macro']:.5f}")
    print(f"  per-class: {baseline_metrics['f1_per_class']}")

    # LLM-only baseline
    llm_only_preds = [llm["parsed_pred"] or "epistemic-internal" for _, llm in aligned]
    llm_only_metrics = evaluate(true_labels, llm_only_preds, "st3")
    print(f"\n[meta] LLM only: F1={llm_only_metrics['f1_macro']:.5f}")
    print(f"  per-class: {llm_only_metrics['f1_per_class']}")

    # ── Choose best ───────────────────────────────────────────────
    strategies = [
        ("logreg_loo", logreg_metrics["f1_macro"], logreg_metrics["f1_per_class"]),
        ("rule_routing", route_metrics["f1_macro"], route_metrics["f1_per_class"]),
        ("weighted_avg", wavg_metrics["f1_macro"], wavg_metrics["f1_per_class"]),
        ("encoder_only", baseline_metrics["f1_macro"], baseline_metrics["f1_per_class"]),
        ("llm_only", llm_only_metrics["f1_macro"], llm_only_metrics["f1_per_class"]),
    ]
    winner = max(strategies, key=lambda s: s[1])
    print(f"\n[meta] WINNER: {winner[0]} with F1={winner[1]:.5f}")

    # ── Write results ─────────────────────────────────────────────
    results = {
        "subtask": "st3",
        "best_strategy": winner[0],
        "best_f1_macro": winner[1],
        "best_f1_per_class": winner[2],
        "encoder_baseline_f1": baseline_metrics["f1_macro"],
        "llm_baseline_f1": llm_only_metrics["f1_macro"],
        "improvement_over_encoder": round(winner[1] - baseline_metrics["f1_macro"], 5),
        "strategies": {
            s[0]: {"f1_macro": s[1], "f1_per_class": s[2]}
            for s in strategies
        },
        "logreg_details": {
            "C": 1.0,
            "class_weight": "balanced",
            "n_features": X.shape[1],
            "feature_names": ["enc_ee", "enc_ei", "enc_pe", "enc_pi",
                             "llm_goal", "llm_basis"],
        },
        "weighted_avg_best_w": best_wavg_w,
        "n_aligned": len(aligned),
        "wall_clock_s": round(guard.elapsed(), 1),
    }

    write_metrics(results, str(exp_dir / "metrics.json"))
    write_status(exp_dir, "done")
    print("[meta] DONE.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[meta] CRASHED: {exc}")
        sys.exit(1)
