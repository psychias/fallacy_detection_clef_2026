# Structural diff summary — pre-Phase-2 draft → post-Phase-3 paper

Five sections, each scoped to ≤45-second read. Source authority: `style-dossier.md`, `content-inventory.md`, `mapping.md` Part 1 (CO-1 through CO-7), Phase 3 `review-log.md`. Backups for byte-level diff: `paper.tex.prePHASE2.bak` and `sections/*.prePHASE2.bak` + `sections/*.retired-PHASE2.bak`.

---

## 1. Abstract — before / after

**Pre-Phase-2 (`paper.tex.prePHASE2.bak`):** 4 sentences front-loading F1 numbers and an `(i)/(ii)/(iii)` enumeration.

> *"We describe our submission to the Touché 2026 Fallacy Detection lab at CLEF, which evaluates systems on three subtasks over Reddit-derived text. We submit one run per subtask and track, totalling six TIRA runs, all built around a fine-tuned RoBERTa-large encoder. On our held-out development partitions, the submitted systems reach F1-macro of 0.73 / 0.91 on ST1 (base / enhanced), 0.73 / 0.97 on ST2, and 0.52 / 0.74 on ST3. The enhanced-track scores carry an explicit caveat: the enhanced fields were rewritten with label knowledge, and although a regex-mask sanity check rules out trivial fallacy-name string-matching as the dominant signal, we cannot rule out softer semantic leakage and therefore report base and enhanced numbers separately. The notebook contributes (i) a per-subtask ablation… (ii) a backbone comparison… and (iii) a record of the negative results…"*

| Pre-Phase-2 sentence | Anti-pattern triggered (dossier §14) |
|---|---|
| S1 *"We describe our submission to the Touché 2026..."* | Participation-first opener — no domain-importance frame |
| S3 *"…F1-macro of 0.73 / 0.91 on ST1 (base / enhanced), 0.73 / 0.97 on ST2, and 0.52 / 0.74 on ST3"* | **F1 numbers in the abstract** (banned by §14) |
| S4 *"…carry an explicit caveat… we cannot rule out softer semantic leakage…"* | **Defensive lead-in** — caveat precedes the substantive key insight |
| S5 *"The notebook contributes (i)… (ii)… and (iii)…"* | **`(i)/(ii)/(iii)` enumeration** (banned by §14) |

**Post-Phase-3 (`paper.tex` lines 54–56):** 8-sentence BIT.UA narrative.

| # | Function | Touché realisation |
|---|---|---|
| S1 | Domain importance | *"Online discussions on platforms like Reddit are a productive testbed for fallacy detection…"* |
| S2 | Benchmark introduction | *"The Touché 2026 lab at CLEF formalises this as a shared task on a Reddit-derived corpus…"* |
| S3 | Participation statement | *"This paper describes our participation in the Touché 2026 Fallacy Detection lab, covering all three subtasks…"* |
| S4 | Method summary 1 | *"Our submitted systems are fine-tuned RoBERTa-large encoders with task-specific heads, augmented with a small batch of LLM-generated synthetic contrastive pairs."* |
| S5 | Method summary 2 + backbone grid | *"For ST2 and ST3 on the enhanced track we additionally extended training with a pseudo-labelled top-up…; we ran a wider backbone grid covering DeBERTa-v3, ModernBERT, and four Qwen variants…"* |
| S6 | Key insight (narrative, not defensive) | *"A key consideration for this year's submission was the structural status of the enhanced-track fields, which were rewritten by the lab organisers with knowledge of the gold labels…"* |
| S7 | High-level outcome (no per-cell numbers) | *"On our held-out development partitions our systems perform competitively across all six (subtask, track) cells, with the strongest results on the enhanced track of ST2."* |
| S8 | Discussion forward-pointer | *"We discuss these outcomes in light of the enhanced-track leakage question and the small-corpus limits of the encoder-vs-LLM comparison, and outline directions for future work."* |
| S9 | Code link | OMITTED (default per inventory §T.4 + Part 2 mapping; revisited in `revision-questions.md`) |

---

## 2. TOC restructure (section number trace, old → new)

Pre-Phase-2 TOC: 8 numbered sections + acks + GenAI + refs + 3 appendices.
Post-Phase-3 TOC: 6 numbered sections + GenAI + refs + 4 appendices.

| Pre-Phase-2 § | Post-Phase-3 § | Carry-over | Action |
|---|---|---|---|
| §1 Introduction | §1 Introduction | dossier §4 (5-paragraph rhythm) | Rewritten; subtask high-level overview added to P2 per CO-4(i) |
| §2 Background and Related Work | §2 Previous Work | CO-1 + Part 2 mapping | Renamed; lineage-anchor opener instead of self-recap (Touché has no self-recap analogue) |
| §3 Task and Data | — (no standalone section) | **CO-4 fold-in** | (i) §3.1 subtasks → §1 P2 + §3 Methodology ¶1; (ii) §3.2 base/enhanced → §3 Methodology ¶1; (iii) §3.3 dataset statistics incl. R1 lock → §4.1 Validation Results opener |
| §4 System Description | §3 Methodology | dossier §5 §6 + CO-6(i) | Renamed; absorbed CO-4 fold-in content; §3.5 LLM detour closes with R3-CO6 locked sentence |
| §5 Results | §4 Results | CO-2 + CO-7 | Renamed; per-subtask layout (§4.1 Validation, §4.2 ST1, §4.3 ST2, §4.4 ST3); tables consolidated (Table 1 in §4.1, Table 2 at §4/§5 boundary, Table 3 in §4.4) |
| §6 Discussion | §5 Discussion and Future Work | dossier §5 + CO-5 | Renamed; absorbed §7 Limitations as §5.4; closing future-work paragraph added |
| §7 Limitations | — (no standalone section) | **CO-5 fold-in** | Six limitations folded into §5.4 as prose paragraphs in BIT.UA register; closing reflection placed at §5.4 close (CO-6(ii)) |
| §8 Conclusion | §6 Conclusion | dossier §5 | Renamed; single-paragraph BIT.UA register; ~160 words matching BIT.UA §6 |
| Acknowledgments | — (omitted) | inventory §T.5 | Sole-author + no grant funding; default omit, surfaced in `revision-questions.md` |
| Declaration on Generative AI | Declaration on Generative AI | inventory §S | Byte-identical to pre-Phase-2 |
| References | References | frozen | 33 entries, unchanged |
| Appendix A Prompts | Appendix A Prompts | inventory §O App A | TODO-placeholder structure unchanged |
| Appendix B Hyperparameter Grid | Appendix B Hyperparameter Grid | inventory §O App B | Unchanged |
| Appendix C Per-Class F1 | Appendix C Per-Class F1 | inventory §O App C | Unchanged |
| Appendix D Citation Graph | Appendix D Citation Graph | inventory §O App D | Unchanged |

A reader cross-referencing the pre-Phase-2 PDF will find:
- §3 Task and Data content distributed (§1 P2 + §3.1 + §4.1).
- §7 Limitations content folded as §5.4.
- All other content preserved at the new section number.

---

## 3. §5.2 four-part scaffold migration

| Element | Pre-Phase-2 location | Post-Phase-3 location | Status |
|---|---|---|---|
| Section label | §6.2 *"The enhanced-track leakage question"* | §5.2 *"The enhanced-track leakage question"* | Same title, shifted under CO-5 four-subsection layout |
| (a) The concern | §6.2(a) | §5.2(a) | `\emph{(a)}` marker once; sentence preserved |
| (b) A sanity check, and what it does not establish | §6.2(b) | §5.2(b) | `\emph{(b)}` marker once; sentence preserved |
| (c) The interpretation + non-closure | §6.2(c) | §5.2(c) | `\emph{(c)}` marker once; **Q1-locked verbatim** "we cannot rule out softer leakage from the rewriter's semantic choices" preserved byte-identical |
| (d) The commitment (primary defence) | §6.2(d) | §5.2(d) | `\emph{(d)}` marker once; `\textbf{Therefore}` opener preserved; commitment to separate base/enhanced reporting preserved |
| Trailing leak-audit gate paragraph (exp_0081) | §6.2 closing | §5.2 closing | Preserved, with `exp_0081` parenthetical stripped per §14 carve-out (audit comment retained) |

Post-Phase-3 integrity pre-pass confirms: `\emph{(a)}` × 1, `\emph{(b)}` × 1, `\emph{(c)}` × 1, `\emph{(d)}` × 1, `\textbf{Therefore}` × 1. Scaffold intact under the rename §6.2 → §5.2.

---

## 4. §Q lock survival ledger — all 9 sentences

| # | Sentence (truncated) | Pre-Phase-2 location | Post-Phase-3 location | Byte-identical |
|---|---|---|---|---|
| Q1 | "…cannot rule out softer leakage from the rewriter's semantic choices" | `sections/06_discussion.tex` §6.2(c) | `sections/05_discussion.tex` §5.2(c) | ✓ |
| Q2 | "The practical-external cell sits at F1 = 0.573 on dev (precision 0.563, recall 0.583, n=24, exp_0073 under the per-class thresholds from exp_0145), against 0.766–0.810…" | `sections/06_discussion.tex` §6.3 | `sections/05_discussion.tex` §5.3 | ✓ |
| Q3 | "Following the official Touché 2026 release, we hold out a stratified development partition…the ST3 partition contains 473 examples." | `sections/03_task_data.tex` §3.3 | `sections/04_results.tex` §4.1 opener | ✓ (relocated per CO-4(iii)) |
| Q4 | "We generated synthetic contrastive pairs using a frontier LLM via API." | `sections/04_system.tex` §4.4 | `sections/03_methodology.tex` §3 (synth subsection) | ✓ |
| Q5 | "We submit one run per subtask and track, totalling six TIRA runs." | `sections/01_introduction.tex` P2 | `sections/01_introduction.tex` P2 | ✓ (in-place) |
| Q6 | "Our final submission (exp_0220) populates all six slots with RoBERTa-large systems." | `sections/05_results.tex` §5.5 | `sections/03_methodology.tex` §3.5 close (CO-6(i)) | ✓ (relocated + capitalization fix: harness restored "Our" sentence-initial after Phase 2 placed it mid-sentence with semicolon) |
| Q7 | "Several runs used the lab-default seed without separate seed-variance ablation; reported single-run dev numbers should be read as point estimates." | `sections/07_limitations.tex` | `sections/05_discussion.tex` §5.4 | ✓ (folded under CO-5) |
| Q8 | "Taken together, these limitations do not invalidate the headline numbers, but the enhanced-track results in particular should be read as upper bounds on what a fair pipeline would extract from the same inputs." | `sections/07_limitations.tex` close | `sections/05_discussion.tex` §5.4 close (CO-6(ii)) | ✓ |
| Q9 | "Our submission and this notebook contribute to the lab's overall record as documented in the Touché 2026 overview paper~\cite{heinrich2026touche}." | `sections/01_introduction.tex` P1 | `sections/01_introduction.tex` P1 | ✓ (in-place) |

**Total: 9/9 byte-identical** (Q3 + Q4 + Q6 + Q7 + Q8 relocated under CO-4 / CO-5 / CO-6; Q1, Q2, Q5, Q9 in-place at renamed sections; Q6 carries the harness-applied sentence-initial capitalization restoration after Phase 2's mid-sentence integration).

---

## 5. Body-prose exp_id reduction trace

**Pre-Phase-2 body-prose count:** ~44 inline `\texttt{exp\_NNNN}` mentions (R2's Round-3 audit at the start of the BIT.UA refresh cycle).
**Post-Phase-2 count:** 5.
**Post-Phase-3 (m-1) count:** 5 (m-1 added one new prose sentence but the sentence introduces no new exp_id references — only the `R2-m1` provenance tag in its `% from` comment).

**The 5 surviving in-prose exp_NNNN mentions** (each justified against dossier §14 carve-out):

| # | exp_id | Location | Carve-out justification |
|---|---|---|---|
| 1 | `exp_0081` | `sections/03_methodology.tex` Fig 1 TikZ node label ("Leak audit \\ exp_0081") | (b) the run id IS the figure's subject — the gate node carries the run id as its label |
| 2 | `exp_0220` | `sections/03_methodology.tex` §3.5 LLM detour close | (a) §Q-locked Q6 verbatim sentence ("Our final submission (exp_0220) populates all six slots with RoBERTa-large systems.") |
| 3 | `exp_0073` | `sections/05_discussion.tex` §5.3 pe anchor sentence | (a) §Q-locked Q2 verbatim — the pe-anchor sentence carries two exp_ids by design; this is the documented per-paragraph-ceiling exception |
| 4 | `exp_0145` | `sections/05_discussion.tex` §5.3 pe anchor sentence | (a) §Q-locked Q2 verbatim — same anchor sentence as #3 |
| 5 | `exp_0207` | `sections/05_discussion.tex` §5.4 limitations paragraph 4 | (b) the run id IS the entity being discussed — "the deadline-stale 32B run" is the §5.4 limitation, and `exp_0207` IS that run |

**Net reduction: 44 → 5** (≈89% reduction in in-prose exp_id mentions). Audit-trail `% from exp_NNNN` LaTeX comments unchanged in coverage — every numeric claim still resolves to its source row. The integrity check post-Phase-3 confirms 57 unique exp_ids still referenced somewhere across body prose, tables, figures, and audit comments; the only reduction is in rendered text density, which was the §14 anti-pattern target.
