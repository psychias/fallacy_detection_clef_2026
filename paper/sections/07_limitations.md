# 7. Limitations

Several limitations should accompany the numbers in this notebook.

First, the test-set leaderboard had not been released at the time of writing; every number in the paper is from our held-out dev partition or from 5-fold cross-validation on it. Where a single 5-fold split was available (ST3 enhanced under our submitted recipe, exp_0153; the Qwen2.5-7B CoT-LoRA comparator, exp_0180; the Qwen3-4B-Thinking comparators, exp_0217/exp_0218) we use the CV mean as the decision-relevant estimate. The other dev numbers in Tables 1-3 are single-run point estimates, and the ranking of close-together rows should be read with appropriate caution.

Second, we did not run a separate seed-variance ablation. Several runs used the lab-default seed without separate seed-variance ablation; reported single-run dev numbers should be read as point estimates. For runs whose individual seeds are not recorded, we used the lab-default seed=42; CV runs used fold seeds 42-46.

Third, the enhanced-track results are subject to the leakage caveat developed in 6.2. The regex-mask diagnostic of exp_0062 rules out the dominant trivial-string channel but does not clear the softer semantic channel, and we have flagged this everywhere the enhanced numbers appear.

Fourth, the LLM comparison ended with a stale Qwen2.5-32B cross-validation: the single-run exp_0207 reached 0.841 on ST3 enhanced but the 5-fold CV was still running at the deadline, and we chose not to wait. This is a real opportunity cost — it is possible the 32B model under CV would have come out ahead — and a reader who cares about that comparison should treat the Qwen-32B row as undervalidated rather than dismissed.

Fifth, the per-class F1 breakdown in Appendix B reflects two different decoding regimes: ST2 values are computed from default argmax over class logits, while ST3 values reflect the per-class threshold calibration of exp_0145 (also used for the headline ST3 numbers in Table 1). Readers comparing per-class numbers across subtasks should keep this asymmetry in mind.

We note one further reproducibility caveat: several experiment-tracking rows in our internal log were marked `status=queued` during the writing window. We have either refreshed those rows or removed citations to them; the frozen `experiments.tsv` accompanying our TIRA submission is the authoritative reference.
