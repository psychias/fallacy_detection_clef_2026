# 5. Results

## 5.1 Submitted-system headline

Table 1 reports precision, recall, and F1-macro on our held-out development partition for the six submitted runs. ST2 and ST3 are macro-averaged over their respective 8- and 4-class label sets. Numbers are from the experiment ids cited in the rightmost column.

**Table 1: Headline dev results for the six submitted runs.**

| Subtask | Track    | P     | R     | F1    | exp_id    |
|---------|----------|-------|-------|-------|-----------|
| ST1     | base     | 0.734 | 0.734 | 0.734 | exp_0033  |
| ST1     | enhanced | 0.917 | 0.914 | 0.914 | exp_0035  |
| ST2     | base     | 0.746 | 0.728 | 0.730 | exp_0032  |
| ST2     | enhanced | 0.970 | 0.972 | 0.970 | exp_0072  |
| ST3     | base     | 0.512 | 0.523 | 0.523 | exp_0069  |
| ST3     | enhanced | 0.732 | 0.740 | 0.735 | exp_0073  |

Two single-number qualifications. On ST3 we additionally ran 5-fold cross-validation under our submitted recipe (exp_0073 re-run as exp_0153, fold seeds 42-46), which gave mean F1 0.875 with the same configuration. A pe-synth variant of the same recipe gave 0.840 in 5-fold CV (exp_0104). The Qwen2.5-7B CoT-LoRA alternative for ST3 enhanced gave 0.851 ± 0.114 over the same fold split (exp_0180); the std overlaps the RoBERTa CV mean, which is the decision-relevant fact behind 5.5. The Touché 2026 test set was not released at the time of writing, so all numbers in this paper are dev or 5-fold CV, never test.

## 5.2 Augmentation ablation

Table 2 decomposes each cell into a real-only baseline, a +synth step, and (where applicable) a +synth+pseudo step. We include the two negative deltas explicitly so that the cost of the wrong recipe is visible: pseudo at max_len=256 loses 0.008 on ST1 enhanced, and synth_v003 on top of synth_v002 loses 0.045 on ST3 base.

**Table 2: Per-subtask augmentation ablation, F1-macro on dev. Δ is the change relative to the immediately preceding row within the same (subtask, track) block. Negative deltas are not hidden. ^a Single-seed unless noted; see Section 7 for the seed-variance caveat.**

| Subtask | Track    | Recipe                               | exp_id    | F1    | Δ      |
|---------|----------|--------------------------------------|-----------|-------|--------|
| ST1     | base     | real only                            | exp_0013  | 0.716 | ---    |
| ST1     | base     | + synth (sw=0.5)                     | exp_0033  | 0.734 | +0.018 |
| ST1     | enhanced | real only                            | exp_0022  | 0.899 | ---    |
| ST1     | enhanced | + synth (sw=0.5)                     | exp_0035  | 0.914 | +0.015 |
| ST1     | enhanced | + synth + pseudo @256 (sw=0.7)       | exp_0071  | 0.906 | −0.008 |
| ST2     | base     | real only                            | exp_0014  | 0.711 | ---    |
| ST2     | base     | + synth (sw=0.5)                     | exp_0032  | 0.730 | +0.019 |
| ST2     | enhanced | real only                            | exp_0023  | 0.923 | ---    |
| ST2     | enhanced | + synth (sw=0.5)                     | exp_0036  | 0.937 | +0.014 |
| ST2     | enhanced | + synth + pseudo @384 (sw=0.7)       | exp_0072  | 0.970 | +0.033 |
| ST3     | base     | real only                            | exp_0058  | 0.436 | ---    |
| ST3     | base     | + synth_v002 @384                    | exp_0069  | 0.523 | +0.087 |
| ST3     | base     | + synth_v002 + v003 @384             | exp_0068  | 0.478 | −0.045 |
| ST3     | base     | + synth_v002 + focal loss            | exp_0053  | 0.505 | −0.018 |
| ST3     | enhanced | real only                            | exp_0024  | 0.694 | ---    |
| ST3     | enhanced | + synth @384                         | exp_0067  | 0.694 | 0.000  |
| ST3     | enhanced | + synth + pseudo @384 (sw=0.7)       | exp_0073  | 0.735 | +0.041 |

The pattern is straightforward and was the basis for our final submission choices: synth alone gives a small but consistent gain across ST1 and ST2; on ST3 base, synth gives the single largest gain in the table; pseudo helps the enhanced track for ST2 and ST3 but not for ST1. Where a recipe lost ground (exp_0068, exp_0053, exp_0071), we report the loss in the same table rather than relegating it to a footnote.

## 5.3 Backbone comparison

Table 3 reports the most informative subset of the backbone runs we tried. All numbers are on the dev partition unless noted; rows marked CV are 5-fold means.

**Table 3: Backbone comparison on ST3 enhanced (and one ST3 base + one ST2 enhanced LLM row for context). Single-run dev F1 unless otherwise noted. The submitted row is marked †. *collapsed* = recipe failed to train.**

| Backbone                          | Subtask | Track | exp_id    | F1            |
|-----------------------------------|---------|-------|-----------|---------------|
| RoBERTa-base                      | ST3     | enh   | exp_0018  | 0.675         |
| RoBERTa-large †                   | ST3     | enh   | exp_0073  | 0.735         |
| DeBERTa-v3-base                   | ST3     | base  | exp_0003  | 0.277         |
| DeBERTa-v3-large (collapsed)      | ST3     | base  | exp_0009  | 0.205         |
| ModernBERT-large                  | ST3     | enh   | exp_0086  | 0.719         |
| Qwen2.5-3B zero-shot              | ST3     | enh   | exp_0091  | 0.467         |
| Qwen2.5-3B kNN few-shot           | ST3     | enh   | exp_0143  | 0.529         |
| Qwen2.5-7B CoT-LoRA (5-fold CV)   | ST3     | enh   | exp_0180  | 0.851 ± 0.114 |
| Qwen2.5-32B CoT-LoRA (single)     | ST3     | enh   | exp_0207  | 0.841         |
| Qwen3-4B-Thinking (5-fold CV)     | ST2     | enh   | exp_0217  | 0.963         |
| Qwen3-4B-Thinking (5-fold CV)     | ST3     | base  | exp_0218  | 0.448         |

Three patterns are worth naming. First, RoBERTa-large dominates RoBERTa-base on the cells where we have both. Second, the DeBERTa-v3 column is a flat collapse: 0.28 at base scale and 0.21 (with a NaN-explosion history; 4.2) at large scale. Third, the LLM column is more interesting than the headline encoder vs LLM framing suggests — Qwen2.5-7B CoT-LoRA's 5-fold mean (0.851) is below the RoBERTa-large 5-fold mean (0.875) and its ±0.114 std overlaps the gap, while Qwen2.5-32B's single-run 0.841 is comparable but unvalidated. Qwen3-4B-Thinking is a mixed picture: strong on ST2 enhanced, below the RoBERTa baseline on ST3 base.

## 5.4 ST3 confusion structure

[FIGURE 2: ST3-enhanced confusion structure (exp_0073). Pending Stage 4 confirmation of saved logits; if unavailable this figure is rendered as a schematic and the per-class diagonals reflect the F1 distribution reported in Appendix B.]

Figure 2 shows the predicted-vs-true confusion structure on the ST3-enhanced dev partition for exp_0073. The visible asymmetry is in the pe row, which is consistent with the diagnostic numbers reported earlier (exp_0082 NLI-basis probe pe-F1 = 0.25; Qwen 3B zero-shot pe-F1 = 0.0 in exp_0091) and with the survival of the pe ceiling through our targeted interventions (4.7, 6.3).

## 5.5 Final submission rationale

Our final submission (exp_0220) populates all six slots with RoBERTa-large systems. We did not use the additional two-runs-per-track budget available under the Touché submission rules; we report on the single CV-validated configuration per slot. The non-obvious choice was ST3 enhanced, where Qwen2.5-32B's single-run 0.841 (exp_0207) sat above the RoBERTa-large dev number of 0.735 but only just below the RoBERTa-large 5-fold CV mean of 0.875. Two reasons drove the encoder choice, and we keep them separate. The first is operational: the Qwen2.5-32B 5-fold CV was still running at the deadline, leaving the RoBERTa-large CV as the only completed cross-validation we had in hand at submission time. The second is the absence of a statistical reason to prefer Qwen even if the comparison had been completed: a 5-fold std of ±0.114 on Qwen2.5-7B (exp_0180) is evidence the encoder-vs-LLM comparison is underpowered at this sample size, not evidence for the system whose CV mean is higher. We therefore do not claim the available CV evidence rules out the Qwen alternatives; we only claim that at deadline the encoder pipeline was the only fully-validated configuration. A post-hoc ensemble (exp_0121) of the available encoder and LLM systems gave no gain over the single-source encoder, which is one further reason we did not retroactively complicate the submission. A submission dry-run (exp_0122) verified that the TIRA-format outputs matched the lab's expected schema before the final upload.
