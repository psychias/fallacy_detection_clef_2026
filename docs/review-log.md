# Review log — Round 1 (Phase 3, 2026-05-25)

Five-reviewer + EIC panel; four ordered passes. The pre-rubric integrity pass runs first; the three substantive rubrics fire only if pre-pass clears. Mapping Part 1 (CO-1 through CO-7) is closed architecture — any reviewer recommendation that would re-litigate those carry-overs is logged in the "Out-of-scope suggestions" section, not actioned.

---

## Pre-rubric integrity pass

### 1.1 Locked verbatim sentences (§Q)

| # | Lock | Target location | Status | File:line |
|---|---|---|---|---|
| Q1 | §5.2(c) "we cannot rule out softer leakage from the rewriter's semantic choices" | sections/05_discussion.tex | **PASS** | sections/05_discussion.tex:19 |
| Q2 | §5.3 pe anchor (F1=0.573 / 0.563 / 0.583 / n=24 / exp_0073 / exp_0145 / 0.766--0.810 / 0.19--0.24 gap) | sections/05_discussion.tex | **PASS** | sections/05_discussion.tex:31 |
| Q3 | R1 §4.1 opener "Following the official Touché 2026 release... the ST3 partition contains 473 examples." | sections/04_results.tex §4.1 | **PASS** | sections/04_results.tex:7 |
| Q4 | R2 synth opener "We generated synthetic contrastive pairs using a frontier LLM via API." | sections/03_methodology.tex §3.3 | **PASS** | sections/03_methodology.tex:68 |
| Q5 | R3 §1 "We submit one run per subtask and track, totalling six TIRA runs." | sections/01_introduction.tex P2 | **PASS** | sections/01_introduction.tex:7 |
| Q6 | CO-6 §3.5 close "Our final submission (`exp\_0220`) populates all six slots with RoBERTa-large systems." | sections/03_methodology.tex §3.5 LLM detour close | **PASS** (sentence-initial "Our" capitalization applied) | sections/03_methodology.tex:101 |
| Q7 | P1 §5.4 seed-variance "Several runs used the lab-default seed without separate seed-variance ablation; reported single-run dev numbers should be read as point estimates." | sections/05_discussion.tex §5.4 | **PASS** | sections/05_discussion.tex:40 |
| Q8 | CO-6 §5.4 closing "Taken together, these limitations do not invalidate the headline numbers, but the enhanced-track results in particular should be read as upper bounds on what a fair pipeline would extract from the same inputs." | sections/05_discussion.tex §5.4 close | **PASS** | sections/05_discussion.tex:52 |
| Q9 | Touché overview cite "Our submission and this notebook contribute to the lab's overall record as documented in the Touché 2026 overview paper~\cite{heinrich2026touche}." | sections/01_introduction.tex §1 P1 | **PASS** | sections/01_introduction.tex:5 |

All 9 locked sentences present byte-identically (modulo benign LaTeX whitespace and the sentence-initial "Our" capitalization on Q6 applied by the parent harness).

### 1.2 exp_id audit comments

- Total `% from` audit-comment lines across active section files (01–06): **86**
  - 01_introduction.tex: 0
  - 02_previous_work.tex: 0
  - 03_methodology.tex: 23
  - 04_results.tex: 51
  - 05_discussion.tex: 12
  - 06_conclusion.tex: 0
- Plus paper.tex contributions (Appendix tables + Hyperparameter Grid prompts subsection): **9**
- **Grand total: 95** (active sections + paper.tex)
- Pre-Phase-2 baseline (sections/*.prePHASE2.bak + paper.tex.prePHASE2.bak): 81 + 9 = **90**
- Phase 2 self-report: 94 (sections only or grand total ambiguous; +1 vs current section total is safe-direction drift)
- 86 in sections alone is below the 89 sections+paper.tex pre-Phase-2 mark only if "pre-Phase-2 was 89" referred to sections+paper.tex combined; grand-total 95 ≥ 89.
- Unique `exp_NNNN` references across all active section files + paper.tex: **57** (inventory §L target = 57)
- **Verdict: PASS.** Audit-comment count ≥89, unique exp_id count = 57 exactly.

### 1.3 §5.2 four-part scaffold

- `\emph{(a) The concern.}` — count 1 at sections/05_discussion.tex:14
- `\emph{(b) A sanity check, and what it does not establish.}` — count 1 at sections/05_discussion.tex:16
- `\emph{(c) The interpretation.}` — count 1 at sections/05_discussion.tex:19
- `\emph{(d) The commitment (primary defence).}` — count 1 at sections/05_discussion.tex:21
- "**Therefore**" opener of (d): present, bold-emphasised (`\textbf{Therefore}`) at sections/05_discussion.tex:21
- **Verdict: PASS.** All four markers exactly once, "Therefore" opener of (d) preserved with light typographic boldening — substance unchanged.

### 1.4 Frozen-bib resolution

- Unique `\cite{}` keys across paper.tex + sections/*.tex: **29**
- references.bib entries: **33** (4 uncited-but-retained per inventory §P: `clark2020electra`, `izmailov2018swa`, `reimers2019sbert`, `froebe:2023b`)
- Missing keys (cited but not in bib): **NONE**
- All 29 cited keys present in references.bib.
- **Verdict: PASS.**

### Integrity pre-pass verdict

**PASS** on all four sub-passes. Proceeding to rubric passes 2–4.

---

## Rubric 1 — Style fidelity to BIT.UA (round captain: R2 writing/style)

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Abstract template (dossier §2): 8–9 sentences in BIT.UA order; S1 domain opener (no numbers); S6 key insight narratively framed; no (i)/(ii)/(iii) enumeration | **VERIFIED** | Eight sentences (S1 domain opener — no numbers; S2 benchmark introduction; S3 participation; S4 method 1; S5 method 2; S6 key insight "A key consideration for this year's submission was the structural status..."; S7 outcome "competitively across all six (subtask, track) cells"; S8 forward-pointer). No S9 (default per inventory §T.4). No enumeration; no F1 numbers. |
| 2 | Introduction 5-paragraph rhythm (dossier §4): P1 domain+benchmark; P2 high-level system+R3 lock; P3 measured central-tension; P4 methodology reshaping; P5 theme+roadmap | **VERIFIED** | P1 (lines 1–5) opens domain+benchmark with mandated Touché overview cite at close. P2 (line 7) carries the R3 lock "six TIRA runs" + system+backbone-grid overview. P3 (line 9) opens "A central consideration for this year's submission was..." — measured register, no op-ed framing. P4 (line 11) reshapes via negative results listing. P5 (line 13) names the theme "honest reading of small-corpus comparisons" + roadmap matching BIT.UA pattern. |
| 3 | Section openings (dossier §6): §2 lineage anchor (CO-1); §3 changes-this-year; §4 results frame; §5 reflective | **VERIFIED** | §2:4 opens with the Sahai 2021 lineage anchor exactly per CO-1 template. §3:4 "Our main changes this year focused on building a uniform fine-tuning pipeline..." mirrors BIT.UA §3 ("Our main changes this year focused on generation..."). §4:1 opens with a single-paragraph framing sentence (no table in first line) per BIT.UA §4. §5:1 "This year we would like to reflect on the structural shape of the Touché 2026 evaluation..." mirrors BIT.UA "This year, we would like to reflect more broadly on the competition itself...". |
| 4 | Section closings (dossier §7): §3 forward-pointer to App A; §4 forward-pointer to §5; §5 close hedge before future-work | **VERIFIED** | §3.3 line 90 "The exact synthetic-pair generation prompts can be seen in Appendix~\ref{app:prompts}." (matches BIT.UA "The prompts can be seen in Appendix A"). §4.4 line 185 "We turn to the interpretation of the enhanced-track numbers and the rare-class behaviour we observed on ST3 in \S\ref{sec:discussion}." §5 closes §5.4 with the locked CO-6 reflection (substantive claim first), then a forward-looking future-work paragraph (prose, not bullets). |
| 5 | Table introduction pattern (dossier §8): 3-sentence frame (name + column grouping + cross-cell variability), walk by setup descriptor (NOT by exp_id) | **VERIFIED with one residual** | Table 1 intro (lines 11–13) names headline; mentions macro-averaging; mentions base-to-enhanced variation. Table 2 caption + body prose; Table 3 frame at line 101 is three-sentence (name table + describe row content + acknowledge cross-row variability). Walks are by setup descriptor ("the +synth recipe", "the synth\_v003 top-up", "the deadline-stale 32B run") not by exp_id. **Residual (m-1):** Table 1 intro paragraph at lines 11–12 has only two framing sentences before the table; a third sentence acknowledging the across-cell variability (cf. BIT.UA "Some techniques were developed between batches...") would tighten the BIT.UA fit. |
| 6 | Voice and tense (dossier §10): "we" throughout; past for experiments / present for findings / future for proposed work | **VERIFIED** | First-person plural "we" consistent across all six section files. Past tense for completed experiments ("We compared", "We generated", "We ran"). Present tense for findings ("The practical-external cell sits at F1 = 0.573", "Encoders did not look like the obvious choice"). Future/conditional for proposed work ("would let us decide more cleanly"). |
| 7 | Hedging register (dossier §10): substantive claim first, caveat after; no defensive lead-ins | **VERIFIED** | Sample audit: §5.1 line 7 "the strongest honest summary is that the LLM track did not show a benefit at this dataset scale, not that LLMs are categorically worse" (substantive then caveat). §5.2 line 21 "Therefore, we present base-track and enhanced-track numbers separately..." (commitment first; caveat embedded in (c) earlier). §5.4 line 38 "every number in the paper is from our held-out dev partition or from 5-fold cross-validation on it" (claim first). |
| 8 | §14 anti-pattern compliance: no F1 in abstract; no (i)/(ii)/(iii) enumeration; no "central tension" op-ed; in-prose exp_NNNN count ≤8 | **VERIFIED** | Abstract: no F1 numbers, no enumeration. Intro P3: opens "A central consideration..." not "The central tension of this notebook is not which backbone won..." — op-ed framing removed. Rendered in-prose `\texttt{exp\_NNNN}` mentions in body: 4 total (Q6 `exp_0220` once, Q2 carve-out exception with `exp_0073` and `exp_0145`, §5.4 carve-out (b) `exp_0207`). Per-paragraph ceiling of 1 respected except for the Q2-exception paragraph. |
| 9 | Acknowledgments omitted (inventory §T.5 default) | **VERIFIED** | No `Acknowledgments` block or `\section*{Acknowledgments}` in paper.tex. Phase 5 `revision-questions.md` will surface the omit-vs-include choice. |
| 10 | Declaration on Generative AI kept byte-identical to pre-Phase-2 (inventory §S) | **VERIFIED** | paper.tex lines 106–118: multi-sentence declaration unchanged from pre-Phase-2 baseline. CEUR-WS GenAI taxonomy categories cited. |

**Rubric 1 summary:** 10 / 10 items VERIFIED; **1 RESIDUAL ISSUE** (m-1, table-1 intro three-sentence frame).

---

## Rubric 2 — Content fidelity to source draft (R1 methodology lead)

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Table 1 headline: all six F1 cells from inventory §A present; P/R/F1 byte-identical (ST3-base P/R = ---) | **VERIFIED** | sections/04_results.tex lines 25–30: ST1-base 0.734/0.734/0.734 / ST1-enh 0.917/0.914/0.914 / ST2-base 0.746/0.728/0.730 / ST2-enh 0.970/0.972/0.970 / ST3-base ---/---/0.523 / ST3-enh 0.732/0.740/0.735 — all six cells preserved exactly. |
| 2 | Table 2 ablation: all 17 rows in inventory §C order | **VERIFIED** | sections/04_results.tex lines 73–94: 17 rows, inventory §C order preserved cell-for-cell (ST1-base × 2 / ST1-enh × 3 / ST2-base × 2 / ST2-enh × 3 / ST3-base × 4 / ST3-enh × 3). |
| 3 | Table 3 backbone: all 11 rows from inventory §D | **VERIFIED** | sections/04_results.tex lines 115–125: 11 rows. RoBERTa-base / RoBERTa-large† / DeBERTa-v3-base / DeBERTa-v3-large / ModernBERT-large / Qwen2.5-3B zero-shot / Qwen2.5-3B kNN few-shot / Qwen2.5-7B CoT-LoRA / Qwen2.5-32B CoT-LoRA / Qwen3-4B-Thinking ST2 / Qwen3-4B-Thinking ST3-base. |
| 4 | CV numbers: ST3-enh exp_0073 re-run = 0.875; Qwen-7B 0.851 ± 0.114; Qwen-32B 0.841; Qwen3-4B-Thinking 0.963 / 0.448 | **VERIFIED** | sections/04_results.tex:35 "mean F1 0.875"; line 36 + line 122 "0.851 $\pm$ 0.114"; line 123 + line 191 "0.841"; line 124 "0.963"; line 125 "0.448". |
| 5 | RoBERTa-base vs large narrative (inventory §E): four comparison cells | **VERIFIED** | sections/03_methodology.tex:54 "0.027 on ST1 base, 0.007 on ST1 enhanced, and 0.055 on ST2 base"; line 55 "0.925 vs.\ 0.923" on ST2 enhanced (treated as tie inside seed-variance). All four cells present. |
| 6 | DeBERTa failure log (inventory §F): "four NaN-explosion incidents + seven collapse-or-fail runs"; plain dev collapse 0.351 / 0.030 / 0.205 | **VERIFIED** | sections/03_methodology.tex:58 plain dev collapse triple (0.351 / 0.030 / 0.205); line 60 "four NaN-explosion incidents and seven collapse-or-fail runs"; line 59 mixed-precision NaN + bf16 silent-failure framing preserved. |
| 7 | Negative-delta call-outs in body: synth_v003 −0.045, focal-loss −0.018, pseudo @256 −0.008 | **VERIFIED** | sections/04_results.tex:59 "$-$0.045", "$-$0.018"; line 44 "0.008" (pseudo @256). Plus sections/03_methodology.tex:74 (−0.045, −0.018), line 83 (0.008). Triple anchor preserved in §3 body AND in §4 body. |
| 8 | pe-class evidence chain (inventory §H): exp_0082 NLI probe pe-F1=0.25; exp_0103 synth + exp_0145 threshold + exp_0146 meta-learner | **VERIFIED** | sections/05_discussion.tex:29 "NLI-style basis probe found pe-F1 of 0.25"; line 30 "three interventions: a pe-targeted synthetic batch, a per-class threshold sweep that pushed the pe threshold to 0.05, and an LLM basis-axis meta-learner." Plus full audit-comment trail: `% from exp_0082 ... % from exp_0103 exp_0145 exp_0146`. |
| 9 | Leakage diagnostic chain (inventory §I): exp_0062 regex-mask F1=0.937; exp_0081 leak-audit gate | **VERIFIED** | sections/05_discussion.tex:16 "F1 was retained at 0.937 after masking, against the unmasked enhanced-track baseline of 0.937" with audit comment `% from exp_0062 exp_0036`. Line 23 leak-audit gate `% from exp_0081`. Both diagnostics survive the four-part (a)–(d) scaffold. |
| 10 | Synthetic data inventory (inventory §J): synth_v002 130 ST1/ST2 + 110 ST3; synth_v003 100 ee + 50 pe; pe-targeted 100 pe+pa | **VERIFIED** | sections/03_methodology.tex:68 "an initial \texttt{synth\_v002} (130 pairs for ST1/ST2 and 110 pairs for ST3); a rare-class top-up \texttt{synth\_v003} for ST3 (100 epistemic-external + 50 practical-external pairs); and a separate \textsf{pe}-targeted batch (100 practical-external + practical-anchor pairs)". |
| 11 | Figures: Fig 1 pipeline TikZ in §3; Fig 2 ST3 confusion in §4.4; Fig 3 citation graph in App D | **VERIFIED** | `\begin{figure}` appears at sections/03_methodology.tex:6 (Fig 1 pipeline), sections/04_results.tex:132 (Fig 2 confusion), paper.tex:260 (Fig 3 citation graph, inside Appendix D). All three figures preserved as TikZ. |
| 12 | Six limitations folded into §5.4 prose paragraphs (NOT numbered list) | **VERIFIED** | sections/05_discussion.tex §5.4 (lines 33–52) renders six items as six discrete prose paragraphs: (i) test-set unreleased / dev+CV only (line 38), (ii) seed-variance with P1 lock (lines 40–41), (iii) leakage cross-ref to §5.2 (line 43), (iv) Qwen-32B CV stale at deadline (lines 45–46), (v) per-class decoding asymmetry (line 48), (vi) TSV reconciliation (line 50). Closes with the locked CO-6 reflection (line 52). All six per inventory §N. No numbered list. |
| 13 | All 33 bib entries preserved in references.bib | **VERIFIED** | `references.bib` has 33 `@…{…,` entries. All 29 cited keys resolve; 4 uncited-but-retained entries match inventory §P list (`clark2020electra`, `izmailov2018swa`, `reimers2019sbert`, `froebe:2023b`). |

**Rubric 2 summary:** 13 / 13 VERIFIED. No content-fidelity violations detected.

---

## Rubric 3 — CLEF notebook conventions (DA + R3 cross-check)

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | CEUR-WS template fit: `\documentclass{ceurart}`; `\copyrightyear{2026}`; `\conference{...September 21--24, 2026, Jena, Germany}` | **VERIFIED** | paper.tex:14 / 23 / 28–29. All three fields exact per Touché lab supplied scaffold. |
| 2 | Author block: ORCID / email / address placeholders pre-camera-ready | **VERIFIED** | paper.tex:44–49. `orcid=0000-0000-0000-0000`, `email=<EMAIL_TO_FILL>`, `\address[1]{Independent Researcher, <Street>, <City>, <Post code>, <Country>}` — all four placeholders match inventory §T item 2. |
| 3 | Title and sub-title: two-line `\title[mode=sub]{...}` pattern (dossier §1) | **VERIFIED** | paper.tex:36–38. Title "Psychias at Touché 26: Encoder Fine-Tuning, Synthetic Augmentation, and an LLM Detour" + sub-title "Notebook for the Touché Lab at CLEF 2026". |
| 4 | Abstract environment well-formed | **VERIFIED** | paper.tex:54–56. Single paragraph, 8 sentences, no enumeration, well-formed. |
| 5 | Keywords environment: 5–7 noun phrases, `\sep`-separated | **VERIFIED** | paper.tex:58–66. Seven phrases (Fallacy detection / Argument scheme classification / Touché 2026 / RoBERTa / Synthetic data augmentation / Pseudo-labelling / CLEF Working Notes) `\sep`-separated. |
| 6 | Section numbering: `\section`/`\subsection` properly nested; `\section*` only for the Declaration on Generative AI | **VERIFIED** | paper.tex `\section` count: 6 numbered body sections (Introduction / Previous Work / Methodology / Results / Discussion and Future Work / Conclusion) + 4 appendix `\section`s. The only `\section*` is line 106 "Declaration on Generative AI" per ceurart convention. |
| 7 | `\bibliography{references}` placed before `\appendix` | **VERIFIED** | paper.tex:123 `\bibliography{references}` followed at line 126 by `\appendix`. |
| 8 | Appendix structure: `\appendix` then four `\section{}` (Prompts / Hyperparameter Grid / Per-Class F1 / Citation Graph) | **VERIFIED** | paper.tex:126 `\appendix`; lines 128 / 167 / 206 / 250 are the four appendix `\section`s in the inventory §O order. |
| 9 | Tables and figures: `[h]` placement; `\caption{}` ABOVE for tables (line above tabular), BELOW for figures; `\label{}` after `\caption{}` | **VERIFIED** | Tables (tab:headline / tab:ablation / tab:backbone / tab:hyper / tab:perclass): all use `\begin{table}[h]\centering\small\caption{...}\label{tab:...}\begin{tabular}{...}` — caption-above, label-after-caption (BIT.UA / dossier §9). Figures (fig:pipeline / fig:confusion / fig:citgraph): `\begin{figure}[h]` or `[t]` then tikzpicture then `\caption{...}\label{fig:...}` — caption-below. All five tables and three figures match the convention. |
| 10 | References format: BibTeX `@inproceedings` / `@article` etc., natbib-compatible | **VERIFIED** | references.bib parseable; standard BibTeX entry types throughout. ceurart class loads natbib via the included style. The bib file is frozen per inventory §P (33 entries unchanged from pre-Phase-2 baseline). |

**Rubric 3 summary:** 10 / 10 VERIFIED. No CLEF-convention violations detected.

---

## Revision items (consolidated, prioritized)

### Critical (C-N)

None. The pre-pass integrity checks all passed, and the three rubric passes surfaced no critical content-fidelity, style, or convention violations.

### Major (M-N)

None.

### Minor (m-N)

**m-1 — Rubric 1 item 5 — Table 1 intro frame** (R2/EIC):
- Location: sections/04_results.tex lines 11–12 (paragraph immediately preceding Table 1 in §4.1).
- Reviewer: R2 (writing/style, round captain).
- Critique: The current Table 1 intro paragraph has two framing sentences ("Table~\ref{tab:headline} reports precision, recall, and F1-macro on our held-out development partition for the six submitted runs..." + "Across the six cells the easier binary task and the hardest scheme task differ by roughly two-tenths of an F1 point on each track, and the base-to-enhanced gap varies markedly across subtasks."). The BIT.UA 3-sentence frame (dossier §8) is name-table + describe-column-grouping + acknowledge-cross-cell-variability. The second sentence here already acknowledges variability; what is missing is a middle "describe column grouping / what varies across columns" sentence between the table-name sentence and the variability acknowledgement.
- Required action: Insert one transitional sentence between current sentences 1 and 2 that describes the column layout (the (subtask, track) cell structure, P/R/F1 columns macro-averaged for ST2 and ST3) so the frame reads name-table / describe-column-grouping / acknowledge-variability. Suggested form: *"Each row carries the precision, recall, and F1-macro for one (subtask, track) cell, with macro-averaging over the 8- and 4-class label sets for ST2 and ST3 respectively; the two enhanced-track values should be read against the leakage caveat developed in §5.2."* (The reference to §5.2 is already in the caption — the recommended sentence reframes the caption material into the introducing paragraph so the dossier §8 three-beat rhythm holds.)
- Acceptance criterion: Table 1 intro paragraph carries three sentences before the `\begin{table}` line, matching dossier §8.
- Priority: minor. The current two-sentence frame is functional; the third sentence is a register tightening, not a substantive fix.

### Notes on items that did NOT trigger a revision

- §3 contains the LLM-detour close at line 101 ("Our final submission (`exp\_0220`)...") with the sentence-initial "Our" correctly capitalized — the parent harness applied this fix per the briefing note.
- §5.4 limitations paragraph 5 (per-class decoding asymmetry, line 48) and paragraph 6 (TSV reconciliation, line 50) are deliberately short — BIT.UA register accepts short paragraphs in §5 mid-section reflections.
- The intro P4 (line 11) compresses three negative-result threads (DeBERTa NaN / LLM detour / pe ceiling) into one sentence each within a single paragraph — terser than BIT.UA's more spacious P4, but the content is preserved and the longer treatment lives in §3 (backbones, LLM detour) and §5.3 (pe). No revision.

---

## Out-of-scope suggestions (NOT triggering Phase 4)

Each entry below is a recommendation that would have been issued by one or more reviewers but is closed by mapping.md Part 1 carry-over CO-1 through CO-7. Logged for the audit trail; NOT actioned.

1. **R3 would have recommended splitting §4 Results into per-track sections** (one §4 per evaluation track). — Closed by **CO-2** (per-subtask layout fixed: §4.1 Validation / §4.2 ST1 / §4.3 ST2 / §4.4 ST3). Not actioned.
2. **R2 would have recommended folding §5.4 Limitations back into a standalone §7** to give the limitations a numbered top-level section. — Closed by **CO-5** (the BIT.UA template has no §7; limitations sit as §5.4, six-section TOC preserved). Not actioned.
3. **DA would have recommended moving the R1-locked "Following the official Touché 2026 release..." sentence out of §4.1 back to a standalone §3 Task and Data section** to keep dataset statistics together with the subtask description. — Closed by **CO-4** (the §3 Task and Data fold-in distributes that material across §1 P2, §3.1 pipeline overview, and §4.1 Validation opener; the R1 lock lives at the §4.1 opener). Not actioned.
4. **R1 would have recommended splitting consolidated Table 2 into three per-subtask ablation tables** for readability. — Closed by **CO-7** (per-subtask split rejected because it would break inventory §C row order, which is R1's standing integrity anchor; one Table 2 with all 17 rows, blocked by `\addlinespace` and `\midrule` per subtask, retains the audit trail). Not actioned.
5. **R3 would have recommended renaming §2 Previous Work back to "Background and Related Work"** to match the more generic notebook convention. — Closed by mapping.md Part 2 §2 mapping (the BIT.UA template title is "Previous Work" and Touché 2026 matches it; the section function changes from self-recap to lineage anchor under CO-1, but the title stays). Not actioned.
6. **DA would have recommended restoring the (i)/(ii)/(iii) contribution enumeration in the abstract** to surface the three contributions discretely for the reader. — Closed by **dossier §14** anti-pattern (BIT.UA abstracts use narrative S1–S9 structure, never enumerated contributions). Not actioned.
7. **R2 would have recommended restoring the in-prose `\texttt{exp\_NNNN}` parentheticals** for full auditability in the rendered text. — Closed by **dossier §14 carve-out** (audit trail lives in `% from exp_NNNN` end-of-line LaTeX comments, not in body prose; in-prose exp_id mentions limited to §Q-locked sentences plus structural carve-outs (b) like `exp_0207` and `exp_0220`). Not actioned.

**Total out-of-scope suggestions: 7.** Each maps to a specific closed carry-over.

---

## Per-reviewer signed checklist

- [x] **EIC sign-off — p_accept_round_phase3 = 0.96**; integrity pre-pass clean on all four sub-passes, three rubric passes returned 33 / 33 VERIFIED with one minor residual (m-1), seven out-of-scope suggestions correctly fenced against the closed carry-overs. Recommend accept-with-minor-revision (m-1 only).
- [x] **R1 (methodology) sign-off** — content-fidelity rubric clean (13 / 13). All Table 1/2/3 cells, the CV numbers, the RoBERTa-base/large narrative, DeBERTa failure aggregate, three negative-delta call-outs, pe evidence chain, leakage diagnostic chain, synth inventory, six limitations, 33 bib entries all VERIFIED. The 57 unique exp_ids and 95 audit-comment lines preserve the integrity anchor R1 has flagged through all prior rounds.
- [x] **R2 (writing/style) sign-off — round captain for style rubric** — 10 / 10 VERIFIED with one residual (m-1 Table 1 frame). Abstract, intro 5-paragraph rhythm, section openings/closings, hedging register, §14 anti-pattern compliance, acks omission, and GenAI declaration all match the BIT.UA template per dossier specification.
- [x] **R3 (related work) sign-off** — §2 Previous Work opens with the CO-1 lineage anchor; all eight prior-work entries (Sahai / Walton / Macagno / Habernal / Da San Martino / Goffredo / Alhindi / MAFALDA / Pan / Jo / Bondarenko 2022 / Bondarenko 2023 / Kiesel 2024 / Kiesel 2025) and the Touché lab history thread present. Touché overview mandatory cite (`heinrich2026touche`) appears in §1 P1 close at sections/01_introduction.tex:5.
- [x] **DA sign-off — confirms §5.2 four-part scaffold and §5.3 pe anchor survived prose-level review.** All nine locked verbatim sentences (§Q) present byte-identically modulo benign whitespace and the harness-applied sentence-initial capitalization on Q6. The (a)(b)(c)(d) markers and the bolded "Therefore" opener of (d) in §5.2 are intact. The pe anchor sentence in §5.3 carries the full data trio (F1=0.573, P=0.563, R=0.583, n=24, exp_0073 + exp_0145 references, 0.766--0.810 comparison band, 0.19--0.24 gap) verbatim. No drift detected at the prose level.

---

**Round-1 verdict: ACCEPT with one minor revision (m-1, Table 1 intro three-sentence frame).**

`p_accept_round_phase3 = 0.96`.

---

## m-1 resolution (2026-05-25, senior-author ruling)

m-1 applied directly per senior ruling: no Phase 4 spawn. The proposed insertion (R2/EIC) was applied as a single `str_replace` on `sections/04_results.tex` between current sentence 1 (name-table) and current sentence 2 (acknowledge-variability). Senior author's classification: this is more than register tightening — the forward-pointer to §5.2 in the second clause sets up the leakage context before the reader hits the table, which changes how the reader parses the 0.97 ST2-enhanced cell on first encounter. The new sentence introduces no new numerical claims; provenance tag is `% from R2-m1` (review-item id), not an `exp_NNNN` audit anchor.

**Post-edit integrity pre-pass (re-run, 2026-05-25):**
- §Q locked verbatim: **9/9 PASS** (Q1–Q9 byte-identical at assigned locations).
- Audit comment count: 87 sections + 9 paper.tex = **96 TOTAL** (was 95 pre-m-1; +1 for the R2-m1 tag, exactly as expected).
- §5.2 four-part scaffold: `\emph{(a)/(b)/(c)/(d)}` each once + `\textbf{Therefore}` opener intact.
- Cite resolution: 29 cited / 33 bib / missing: NONE.
- R2-m1 tag and the new sentence both verified present at `sections/04_results.tex`.

Phase 3 Round 1 closes here. Phase 4 was not invoked (no substantive items remaining); Phase 5 fires next and produces `diff-summary.md` + `revision-questions.md`.
