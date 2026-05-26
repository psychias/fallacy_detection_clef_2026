# Content inventory — current Touché 2026 notebook draft

Source: `paper.tex` + `sections/01_introduction.tex` through `08_conclusion.tex` + `references.bib` + `per_class_f1_st2_st3.csv`, all in `G:\My Drive\fallacy_detection\touche26_paper\`. Post-Round-3 state (revision_log_r3.md applied). The PDF reference `Psychias_at_Touche-2.pdf` is not on disk in this workspace; the `.tex` sources are the content authority.

This inventory is the Phase-0 ledger that the Phase-2 draft must preserve in full. Every line below must survive into the rewritten paper.

---

## A. Submitted runs (Table 1 cells)

All six are dev-partition F1-macro for the submitted RoBERTa-large systems.

| Subtask | Track | P | R | F1 | exp_id |
|---|---|---|---|---|---|
| ST1 | base | 0.734 | 0.734 | 0.734 | `exp_0033` |
| ST1 | enhanced | 0.917 | 0.914 | 0.914 | `exp_0035` |
| ST2 | base | 0.746 | 0.728 | 0.730 | `exp_0032` |
| ST2 | enhanced | 0.970 | 0.972 | 0.970 | `exp_0072` |
| ST3 | base | — | — | 0.523 | `exp_0069` (P/R dropped; data-quality artefact) |
| ST3 | enhanced | 0.732 | 0.740 | 0.735 | `exp_0073` |

---

## B. Cross-validation numbers

- ST3-enh CV under submitted recipe: mean F1 **0.875** (`exp_0073` re-run as `exp_0153`, fold seeds 42–46)
- Qwen2.5-7B CoT-LoRA ST3-enh CV: **0.851 ± 0.114** (`exp_0180`)
- Qwen3-4B-Thinking ST2-enh CV: **0.963** (`exp_0217`)
- Qwen3-4B-Thinking ST3-base CV: **0.448** (`exp_0218`)

---

## C. Ablation table (Table 2) — all rows

Every row's F1 and Δ must be preserved.

| Subtask | Track | Recipe | exp_id | F1 | Δ |
|---|---|---|---|---|---|
| ST1 | base | real only | `exp_0013` | 0.716 | — |
| ST1 | base | + synth (sw=0.5) | `exp_0033` | 0.734 | +0.018 |
| ST1 | enh | real only | `exp_0022` | 0.899 | — |
| ST1 | enh | + synth (sw=0.5) | `exp_0035` | 0.914 | +0.015 |
| ST1 | enh | + synth + pseudo @256 (sw=0.7) | `exp_0071` | 0.906 | **−0.008** |
| ST2 | base | real only | `exp_0014` | 0.711 | — |
| ST2 | base | + synth (sw=0.5) | `exp_0032` | 0.730 | +0.019 |
| ST2 | enh | real only | `exp_0023` | 0.923 | — |
| ST2 | enh | + synth (sw=0.5) | `exp_0036` | 0.937 | +0.014 |
| ST2 | enh | + synth + pseudo @384 (sw=0.7) | `exp_0072` | 0.970 | +0.033 |
| ST3 | base | real only | `exp_0058` | 0.436 | — |
| ST3 | base | + synth_v002 @384 | `exp_0069` | 0.523 | +0.087 |
| ST3 | base | + synth_v002 + v003 @384 | `exp_0068` | 0.478 | **−0.045** |
| ST3 | base | + synth_v002 + focal loss | `exp_0053` | 0.505 | **−0.018** |
| ST3 | enh | real only | `exp_0024` | 0.694 | — |
| ST3 | enh | + synth @384 | `exp_0067` | 0.694 | 0.000 |
| ST3 | enh | + synth + pseudo @384 (sw=0.7) | `exp_0073` | 0.735 | +0.041 |

The three bolded **negative deltas** must remain surfaced in body prose (R1's standing audit anchor).

---

## D. Backbone comparison (Table 3) — all rows

| Backbone | Subtask | Track | exp_id | F1 |
|---|---|---|---|---|
| RoBERTa-base | ST3 | enh | `exp_0018` | 0.675 |
| **RoBERTa-large (submitted)** | ST3 | enh | `exp_0073` | **0.735** |
| DeBERTa-v3-base | ST3 | base | `exp_0003` | 0.277 |
| DeBERTa-v3-large (collapsed) | ST3 | base | `exp_0009` | 0.205 |
| ModernBERT-large | ST3 | enh | `exp_0086` | 0.719 |
| Qwen2.5-3B zero-shot | ST3 | enh | `exp_0091` | 0.467 |
| Qwen2.5-3B kNN few-shot | ST3 | enh | `exp_0143` | 0.529 |
| Qwen2.5-7B CoT-LoRA (5-fold CV) | ST3 | enh | `exp_0180` | 0.851 ± 0.114 |
| Qwen2.5-32B CoT-LoRA (single) | ST3 | enh | `exp_0207` | 0.841 |
| Qwen3-4B-Thinking (5-fold CV) | ST2 | enh | `exp_0217` | 0.963 |
| Qwen3-4B-Thinking (5-fold CV) | ST3 | base | `exp_0218` | 0.448 |

---

## E. RoBERTa-base vs RoBERTa-large narrative numbers (§4.2)

- ST1 base: 0.689 (`exp_0004`) vs 0.716 (`exp_0013`) → Δ +0.027
- ST1 enh: 0.892 (`exp_0016`) vs 0.899 (`exp_0022`) → Δ +0.007
- ST2 base: 0.656 (`exp_0005`) vs 0.711 (`exp_0014`) → Δ +0.055
- ST2 enh: 0.925 (`exp_0017`) vs 0.923 (`exp_0023`) → Δ −0.002 (inside seed-variance, treat as tie)

---

## F. DeBERTa-v3-large failure log (§4.2 narrative — must be retained)

Plain dev collapse:
- `exp_0007` ST1 base → 0.351
- `exp_0008` ST2 base → 0.030
- `exp_0009` and re-run `exp_0040` ST3 base → both 0.205

Mixed-precision NaN explosions: `exp_0041`, `exp_0042`, `exp_0043`, `exp_0044`.

bf16 + lr=5e-6 silent failures: `exp_0049`, `exp_0050`, `exp_0051`.

**Aggregate framing:** "four NaN-explosion incidents + seven collapse-or-fail runs" — preserve this count.

---

## G. Per-class F1 (Appendix C / Table 5)

Source: `per_class_f1_st2_st3.csv` (threshold-calibrated ST3; ST2 argmax).

ST2-enhanced (`exp_0072`, argmax):
| label | P | R | F1 | n |
|---|---|---|---|---|
| authority | 0.972 | 0.986 | 0.979 | 72 |
| black_white | 0.967 | 0.967 | 0.967 | 30 |
| hasty_generalization | 0.964 | 0.958 | 0.961 | 72 |
| natural | 1.000 | 1.000 | 1.000 | 21 |
| population | 0.948 | 0.971 | 0.959 | 68 |
| slippery_slope | 1.000 | 0.957 | 0.978 | 46 |
| tradition | 0.971 | 0.971 | 0.971 | 35 |
| worse_problems | 0.940 | 0.957 | 0.948 | 47 |

ST3-enhanced (`exp_0073` under `exp_0145` thresholds):
| label | P | R | F1 | n |
|---|---|---|---|---|
| ei (epistemic-internal) | 0.755 | 0.833 | 0.792 | 48 |
| ee (epistemic-external) | 0.790 | 0.830 | 0.810 | 53 |
| pi (practical-internal) | 0.821 | 0.718 | 0.766 | 39 |
| pe (practical-external) | **0.563** | **0.583** | **0.573** | **24** |

Anchor numbers (locked from Round 2.5):
- pe F1 = **0.573**
- gap to other three: **0.19–0.24** (0.766 − 0.573 = 0.193; 0.810 − 0.573 = 0.237)

---

## H. pe-class evidence chain (§4.7 + §6.3)

- `exp_0082` NLI-style basis probe: pe-F1 = **0.25** (pre-classifier diagnostic)
- `exp_0091` Qwen 3B zero-shot ST3 pe = 0.0
- `exp_0103` pe-targeted synthetic batch (100 pe + practical-anchor pairs)
- `exp_0145` per-class threshold sweep: best `{ee=0.1, ei=0.1, pe=0.05, pi=0.4}` → +1.0 pp aggregate F1 over default argmax
- `exp_0146` LLM basis-axis meta-learner → no improvement

---

## I. Leakage diagnostic chain (§6.2)

- `exp_0062` regex-mask sanity check: F1 retained at **0.937** after masking; unmasked baseline `exp_0036` = 0.937. **Null delta** is the diagnostic outcome.
- `exp_0081` leak-audit gate (before pseudo-labelling) → no detectable leak under the test we ran.

---

## J. Synthetic data inventory

- `synth_v002`: 130 contrastive pairs for ST1/ST2 (`exp_0025`); 110 for ST3 (`exp_0030`).
- `synth_v003`: rare-class top-up for ST3 (`exp_0056`) — 100 ee + 50 pe pairs.
- `pe`-targeted batch: 100 pe + practical-anchor pairs (`exp_0103`).

---

## K. Other infrastructure / diagnostic runs

- `exp_0061`: token-length diagnostic motivating max_len=384 for ST3, 256 for ST1/ST2.
- `exp_0121`: post-hoc ensemble — no gain over single-source.
- `exp_0122`: TIRA submission dry-run — schema verified.
- `exp_0124`: stratified 5-fold splits for ST3 (473 examples).
- `exp_0138`: bge-large-en-v1.5 sentence embedding index for kNN few-shot (1876 × 1024).
- `exp_0220`: final TIRA submission record (six slots, all RoBERTa-large).

---

## L. Total unique exp_ids

**57 distinct `exp_NNNN` references.** Round-3 spot-check confirmed: 57 before round 3, 57 after — none added, none removed. The audit comments (`% from exp_NNNN`) on every numeric claim are the integrity anchor and must be preserved through any rewrite.

---

## M. Figures

| # | Name | Where | What it shows |
|---|---|---|---|
| Fig 1 | Pipeline | §4.1 | TikZ block diagram: 3 input streams (Real / Synthetic / Pool→LeakAudit→Pseudo) → Mini-batch mix (sw 1.0/0.5/0.7) → RoBERTa-large encoder → Per-task head |
| Fig 2 | ST3 confusion (schematic) | §5.4 | TikZ 4×4 grid; ee/ei/pi diagonals 0.81/0.79/0.77; pe diagonal 0.57; off-diagonal illustrative only (logits not checkpointed for `exp_0073`) |
| Fig 3 | Citation graph | Appendix D | TikZ bipartite: 10 citation nodes (blue/teal/gray/purple/amber) → 5 pipeline-component nodes; solid arrows = shipped routes; dashed = dropped (DeBERTa-v3, ModernBERT, LLM detour) |

---

## N. Limitations (current §7, 6 numbered + closing)

1. Test-set leaderboard not released at writing time; every number is dev partition or 5-fold CV.
2. **P1 verbatim (locked):** "Several runs used the lab-default seed without separate seed-variance ablation; reported single-run dev numbers should be read as point estimates."
3. Enhanced-track leakage caveat (cross-references §6.2).
4. Qwen2.5-32B 5-fold CV stale at deadline; treated as undervalidated rather than dismissed.
5. Per-class F1 (Appendix C) reflects two decoding regimes: ST2 argmax vs ST3 threshold-calibrated under `exp_0145`.
6. TSV reconciliation caveat: `status=queued` rows refreshed or removed; frozen `experiments.tsv` in TIRA submission package is authoritative.
- Closing reflection: "Taken together, these limitations do not invalidate the headline numbers, but the enhanced-track results in particular should be read as upper bounds on what a fair pipeline would extract from the same inputs."

---

## O. Appendices

- **App A — Prompts:** TODO-placeholder structure (3 paragraphs: synth template, pe-targeted variant, pseudo-label τ=0.9 selection). Senior author will paste verbatim prompt text before camera-ready.
- **App B — Hyperparameter grid (Table 4):** 6 rows (one per submitted run) × 8 cols (Subtask, Track, exp_id, max_len, batch, lr, epochs, extras).
- **App C — Per-class F1 (Table 5):** 12 rows (8 ST2 + 4 ST3) × 5 cols (Label, P, R, F1, Support). Values per §G above.
- **App D — Citation graph (Figure 3):** TikZ bipartite as in §M above.

---

## P. References (`references.bib`, 33 entries)

Active citation keys (29 used in body; 4 currently uncited but retained for future use):

**Direct task:** `sahai2021invisiblewall`, `macagno2022argumentation`, `heinrich2026touche`

**Encoders:** `liu2019roberta`, `he2021debertav3`, `warner2024modernbert`, `clark2020electra` (uncited)

**LLM track:** `qwen25`, `qwen3`, `hu2022lora`, `wei2022cot`, `dettmers2023qlora`

**Training techniques:** `lee2013pseudolabel`, `lin2017focal`, `izmailov2018swa` (uncited), `reimers2019sbert` (uncited)

**Fallacy / argumentation prior work:** `habernal2018arc`, `dasanmartino2019propaganda`, `goffredo2022political`, `alhindi2022multitask`, `walton2008schemes`

**Infrastructure:** `frobe2023tira`, `froebe:2023b` (template-style alias, uncited), `wolf2020transformers`

**Lit-review additions:** `helwe2024mafalda`, `pan2024zeroshotfallacy`, `bondarenko2022touche`, `bondarenko2023touche`, `kiesel2024touche`, `kiesel2025touche`, `li2023synthdata`, `mukherjee2020ust`, `jo2021schemes`

Bib is **frozen**: do not add or remove entries during the rewrite. The 4 uncited entries (`clark2020electra`, `izmailov2018swa`, `reimers2019sbert`, `froebe:2023b`) stay in `references.bib` — BibTeX prints only cited entries, so they cost nothing in the rendered paper.

---

## Q. Locked verbatim sentences (frozen across all rewrites)

- **§6.2(c) non-negotiable:** "we cannot rule out softer leakage from the rewriter's semantic choices"
- **§6.3 pe anchor:** "The practical-external cell sits at F1 = 0.573 on dev (precision 0.563, recall 0.583, n=24, `exp_0073` under the per-class thresholds from `exp_0145`), against 0.766–0.810 for the other three ST3 classes; the pe ceiling is a concrete 0.19–0.24 F1 gap that the targeted synth augmentation, the per-class threshold sweep, and the basis-axis meta-learner did not close."
- **R1 (§3.3):** "Following the official Touché 2026 release, we hold out a stratified development partition from the published training data; the ST3 partition contains 473 examples."
- **R2 (§4.4):** "We generated synthetic contrastive pairs using a frontier LLM via API."
- **R3 (§1):** "We submit one run per subtask and track, totalling six TIRA runs."
- **R3 (§5.5):** "Our final submission (`exp_0220`) populates all six slots with RoBERTa-large systems."
- **P1 (§7):** "Several runs used the lab-default seed without separate seed-variance ablation; reported single-run dev numbers should be read as point estimates."
- **Touché overview mandatory cite (§1):** sentence containing `\cite{heinrich2026touche}` in the form "Our submission and this notebook contribute to the lab's overall record as documented in the Touché 2026 overview paper".

---

## R. §6.2 four-part structure (frozen scaffold)

The four labelled parts (a)/(b)/(c)/(d) must be preserved as a sub-argument under Discussion:
- **(a)** The concern (label-aware rewrite is structural, not accidental).
- **(b)** The sanity check (regex-mask, F1=0.937 retained, what it does NOT establish).
- **(c)** The interpretation (narrow result; defence rests on (d); non-closure sentence above).
- **(d)** The commitment (Therefore, separate base/enhanced reporting; anchor cross-comparisons on base + CV).

Plus a trailing paragraph on the `exp_0081` leak-audit gate explicitly naming its limit.

---

## S. Declaration on Generative AI (frozen)

Multi-sentence paragraph in current paper.tex (longer than BIT.UA's one-sentence form, but matches CEUR-WS policy template). Keep as-is unless the senior author signals otherwise during final length-cut.

---

## T. Open factual questions to surface in `revision-questions.md` (Phase 5)

These are pre-existing TODOs from prior rounds that the rewrite must NOT silently resolve:

1. **ST1/ST2 dev split sizes** — currently deferred to camera-ready (TODO comment at `sections/03_task_data.tex` line 14).
2. **Author block fields** — ORCID (`0000-0000-0000-0000`), email (`<EMAIL_TO_FILL>`), affiliation street/city/postcode/country all placeholders pending camera-ready.
3. **Appendix A prompt text** — verbatim text held by the senior author; current draft has `[TODO senior: paste verbatim ...]` blocks.
4. **GitHub / TIRA package URL** — analogue of BIT.UA's abstract closing line; senior author has not yet confirmed whether a public repository will accompany this submission.
5. **Acknowledgments paragraph (Phase-0 sign-off, 2026-05-25).** BIT.UA's acks names a specific FCT grant with DOI. For a sole-author submission with no grant funding, two acceptable forms: (a) omit the acks section entirely (some CEUR notebooks do this), or (b) include a one-line generic acks naming only the lab organizers (current paper has *"The author thanks the Touché 2026 lab organizers for providing the dataset, evaluation infrastructure, and submission platform."* in `paper.tex.preLENGTH.bak` — already removed in the length pass but recoverable). The Phase-2 rewrite should DEFAULT to (a) — omit the acks section — and Phase-5 `revision-questions.md` should surface the choice between (a) and (b) for senior-author confirmation before camera-ready.
