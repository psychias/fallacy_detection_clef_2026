# Phase 1 mapping — BIT.UA at BioASQ 13B template → Touché 2026 content

Phase 1 deliverable per senior brief 2026-05-25. Two parts:

1. **Standalone carry-over rows** (top of file, NOT folded into the section table) — surfacing decisions made at Phase 0 sign-off that the Phase 2 rewriter must observe.
2. **Section-by-section mapping table** (BIT.UA element on the left, Touché content on the right, source pointer for every cell).

References to inventory items use the form §[letter] from `content-inventory.md`; references to dossier items use §[number] from `style-dossier.md`.

---

## Part 1 — Standalone carry-overs (Phase 0 sign-off, 2026-05-25)

These three rows are kept SEPARATE from the section mapping so they cannot drift in Phase 2.

| # | Carry-over | Decision | Where it surfaces in Phase 2 |
|---|---|---|---|
| **CO-1** | §2 opening sentence | Open with a lineage anchor, not a self-recap. Template: *"The Reddit informal-fallacy dataset that the Touché 2026 lab redistributes was introduced by Sahai et al.~\cite{sahai2021invisiblewall}, who labelled a long-tail eight-way taxonomy over comments mined from r/changemyview and adjacent subreddits and reported the first transformer baselines on the resulting corpus."* — current `sections/02_background.tex` line 4 is already close to this; the Phase 2 rewriter keeps the substance and only adjusts the leading clause to match BIT.UA's first-sentence rhythm. This section function is "situating-this-paper-inside-a-lineage", not "recapping our own prior submission" (which Touché has none of). | First paragraph of new §2 *Previous Work*. |
| **CO-2** | §4 Results subsection structure | **Per-subtask layout fixed (not combined):** §4.1 Validation, §4.2 Official Results — ST1, §4.3 Official Results — ST2, §4.4 Official Results — ST3. Mirrors BIT.UA's four-subsection rhythm (4.1 Validation, 4.2 Phase A, 4.3 Phase A+, 4.4 Phase B). Each subtask subsection carries one summary-of-systems table (analogue of BIT.UA Tables 4/6/8) and one performance table (analogue of BIT.UA Tables 5/7/9). | §4.1–§4.4 subsection headers in the rewritten paper; each subsection's table-pair. |
| **CO-3** | Best Competitor / Median rows | **Dropped entirely** from the Touché analogues of BIT.UA Tables 5, 7, 9. Test-set leaderboard not released at writing time, so there is no baseline to populate against. Do NOT render empty rows or `—` placeholders. Every Touché performance table caption carries the explicit note: *"Test-set leaderboard not released at writing time; no median or best-competitor reference available — see §5.4 Limitations."* | Table 1 caption + Table 2/3 captions. |
| **CO-4** | §3 Task and Data fold-in (Phase 1 sign-off blocking 1, 2026-05-25) | Current draft's three §3 subsections are reabsorbed: **(i)** §3.1 subtask descriptions — expand §1 P2 by one sentence naming ST1/ST2/ST3 at high level, then fold the full subtask description into §3 Methodology paragraph 1 (pipeline overview) as scaffolding BEFORE the streams-into-mini-batch sentence; **(ii)** §3.2 base/enhanced tracks — §3 Methodology paragraph 1, immediately after the subtask description so the §5.2 leakage scaffold has its setup; **(iii)** §3.3 dataset statistics (including R1 lock *"…the ST3 partition contains 473 examples."*) — opens §4.1 Validation Results, structurally parallel to BIT.UA's *"Validation was conducted on the first and second batches of the 2024 dataset"* sentence. No standalone §3 Task and Data section in the rewritten paper. | §1 P2 + §3 Methodology paragraph 1 + §4.1 opener. See Part 4 "Touché content with no direct BIT.UA position" for the dataset-statistics placement detail. |
| **CO-5** | §5 four-subsection layout (Phase 1 sign-off blocking 2, 2026-05-25) | BIT.UA has no §7; Touché folds the current §7 Limitations INTO §5 as a labelled mid-section, preserving the audit-anchor structure that R1/R2/R3/P1 review passes rely on. **§5 subsections in order:** §5.1 Encoder vs LLM at this scale; §5.2 The enhanced-track leakage question (four-part (a)/(b)/(c)/(d) scaffold, inventory §R); §5.3 The pe rare-class ceiling; **§5.4 Limitations** (six items as prose paragraphs in BIT.UA register, NOT a numbered list, but clustered as a labelled subsection so P1 lock has a home); then a future-work closing paragraph (NOT numbered as §5.5 — it's the natural closing of §5 in BIT.UA's style). Six-section TOC preserved (§1/§2/§3/§4/§5/§6 — no §7). | §5.1, §5.2, §5.3, §5.4 subsection headers and the closing future-work paragraph. |
| **CO-6** | Two §Q-locked sentences placed (Phase 1 sign-off blocking 3, 2026-05-25) | **(i)** R3-§5.5 verbatim *"Our final submission (`exp\_0220`) populates all six slots with RoBERTa-large systems."* → final sentence of §3 Methodology paragraph 5 (the LLM detour paragraph). Natural close to the LLM-detour narrative and a §14-carve-out case (b): `exp_0220` IS the submission record being referenced. **(ii)** Inventory §N closing reflection *"Taken together, these limitations do not invalidate the headline numbers, but the enhanced-track results in particular should be read as upper bounds on what a fair pipeline would extract from the same inputs."* → final sentence of §5.4 Limitations. | §3.5 (LLM detour) closing + §5.4 closing. |
| **CO-7** | Tables consolidated (Phase 1 sign-off blocking 4, 2026-05-25) | **One Table 1** (six-cell headline, inventory §A) placed in §4.1; **one Table 2** (full ablation, inventory §C, all 17 rows) placed at the §4/§5 boundary; **one Table 3** (backbone comparison, inventory §D, all 11 rows) placed in §4.4. Per-subtask split rejected: it would force breaking the inventory §C row order, which is R1's standing integrity anchor. Consolidated layout keeps the audit trail intact and produces fewer floats. CO-2's per-subtask §4 subsection structure is unchanged; tables sit where indicated above, with each subsection prose pointing back to the relevant rows by subtask × track. | Table 1 in §4.1; Table 2 at §4/§5 boundary; Table 3 in §4.4. Captions per CO-3. |

---

## Part 2 — Section-by-section mapping

Format: BIT.UA element (left) → Touché content (right) + source pointer + style notes.

### Title block

| BIT.UA element | Touché content | Source / decision |
|---|---|---|
| `\title{BIT.UA at BioASQ 13B: Revisiting Evaluation,}` `\title{DPRF-Enhanced Retrieval and Fine-Tuned LLMs}` | `\title{Psychias at Touché 26: Encoder Fine-Tuning, Synthetic Augmentation, and an LLM Detour}` | dossier §1 (revised). Comma-separated theme list confirmed. |
| Sub-title `Notebook for the BioASQ Lab at CLEF 2025` | `\title[mode=sub]{Notebook for the Touché Lab at CLEF 2026}` | dossier §1. |
| Single corresponding author with ORCID + email | `\author[1]{Stelios Psychias}[orcid=…, email=…]` with placeholders pending camera-ready | inventory §T item 2. |
| Affiliation footnote `1IEETA/DETI, LASI, University of Aveiro, Aveiro, Portugal` | `\address[1]{Independent Researcher, <Street>, <City>, <Post code>, <Country>}` placeholders | inventory §T item 2. |

### Abstract — 9-sentence template

| BIT.UA sentence function | Touché sentence (target) |
|---|---|
| S1 domain importance | "Online discussions on platforms like Reddit are a productive testbed for fallacy detection: short texts, broad topical range, and the full inventory of informal fallacies catalogued by classical argumentation theory long before social media." |
| S2 benchmark introduction | "The Touché 2026 lab at CLEF formalises this as a shared task on a Reddit-derived corpus and provides a structured benchmark for systems that combine fallacy detection with argument-scheme classification." |
| S3 participation statement | "This paper describes our participation in the Touché 2026 Fallacy Detection lab, covering all three subtasks (ST1 binary fallacy detection, ST2 eight-way fallacy classification, ST3 four-way argument-scheme classification) and both evaluation tracks." |
| S4 method summary 1 | "Our submitted systems are fine-tuned RoBERTa-large encoders with task-specific heads, augmented with a small batch of LLM-generated synthetic contrastive pairs." |
| S5 method summary 2 | "For ST2 and ST3 on the enhanced track we additionally extended training with a pseudo-labelled top-up of the published unlabelled pool, gated by a leak-audit step; we ran a wider backbone grid covering DeBERTa-v3, ModernBERT, and four Qwen variants (3B zero-shot/few-shot, 7B and 32B CoT-LoRA, and Qwen3-4B-Thinking) for comparison." |
| S6 key insight (narrative) | "A key consideration for this year's submission was the structural status of the enhanced-track fields, which were rewritten by the lab organisers with knowledge of the gold labels and which therefore shaped both our system design and our interpretation of results." |
| S7 high-level outcome (no per-cell numbers) | "On our held-out development partitions our systems perform competitively across all six (subtask, track) cells, with the strongest results on the enhanced track of ST2." |
| S8 discussion forward-pointer | "We discuss these outcomes in light of the enhanced-track leakage question and the small-corpus limits of the encoder-vs-LLM comparison, and outline directions for future work." |
| S9 code/data availability | TODO — analogue of BIT.UA's `https://github.com/bioinformatics-ua/BioASQ13B`. Awaiting senior-author confirmation per inventory §T item 4. Default in Phase 2 draft: omit S9 entirely; Phase 5 `revision-questions.md` surfaces the choice. |

**Ban list (dossier §14, applied):** no F1 numbers, no `(i)/(ii)/(iii)` enumeration, no caveat preceding the key insight.

### Keywords

| BIT.UA pattern | Touché content |
|---|---|
| 6 noun phrases, `\sep`-separated: *Information Retrieval, Dense Retrieval, Semantic Search, Large Language Model, Answer Generation, Pseudo Relevance Feedback* | *Fallacy detection \sep Argument scheme classification \sep Touché 2026 \sep RoBERTa \sep Synthetic data augmentation \sep Pseudo-labelling \sep CLEF Working Notes* (unchanged from current draft; already 7 phrases, `\sep`-separated) |

### §1 Introduction — 5-paragraph rhythm

| BIT.UA paragraph | Touché paragraph |
|---|---|
| P1 domain + benchmark (cites lab overview + previous notebook) | Domain importance of fallacy detection in online discussions + Touché 2026 as the benchmark + cites: `sahai2021invisiblewall`, `walton2008schemes`, `heinrich2026touche`. Source: current `sections/01_introduction.tex` paragraph 1; preserve content, retemplate first sentence to mirror BIT.UA's "are critical for navigating…" cadence. |
| P2 high-level system (cites prior team submission) | High-level pipeline + backbone grid + LLM detour. Source: current `sections/01_introduction.tex` paragraph 2. Locked: R3 sentence "We submit one run per subtask and track, totalling six TIRA runs." (inventory §Q). Preserve all 13 citations (frobe2023tira, liu2019roberta, he2021debertav3, warner2024modernbert, qwen25, qwen3, hu2022lora, dettmers2023qlora, wei2022cot, plus the four implicit through context). |
| P3 central tension (BIT.UA: metric misalignment; Touché: enhanced-track leakage) | Rewrite the current "The central tension of this notebook is not which backbone won" sentence — too op-ed (dossier §14 anti-pattern). Target register: measured, parallel to BIT.UA's *"However, a key finding from our previous work was the significant misalignment between automatic evaluation metrics… and human evaluation of answer quality."* Touché analogue (draft): *"A central consideration for this year's submission was the structural status of the enhanced-track fields. The lab organisers produced the enhanced fields with knowledge of the gold labels, which means a classifier trained on them has access to features that an honest pipeline would have to recover from raw text. This shaped both how we read our highest numbers and how the rest of the paper is organised."* |
| P4 how that tension reshaped methodology | Source: current `sections/01_introduction.tex` paragraph 3 (post-BIT.UA-refresh version). Preserve: separate base/enhanced reporting, anchor on base-track + 5-fold CV, keep negative results in body. Inventory §R (the four-part §6.2 scaffold) is referenced from this paragraph. |
| P5 theme statement + roadmap | Theme: "honest reading of small-corpus comparisons" or similar; roadmap matches BIT.UA's *"Section 2 describes our submissions from the previous year… Section 6 summarizes our conclusions and outlines directions for future work."* Touché roadmap: Section 2 (Previous Work / lineage), Section 3 (Methodology), Section 4 (Results), Section 5 (Discussion and Future Work), Section 6 (Conclusion). |

**Mandatory cite in §1:** the Touché overview sentence (locked, inventory §Q): *"Our submission and this notebook contribute to the lab's overall record as documented in the Touché 2026 overview paper~\cite{heinrich2026touche}."* Place at end of P1.

### §2 Previous Work (renamed from current "Background and Related Work")

**Function:** BIT.UA §2 is a self-recap; Touché analogue is a lineage section.

| BIT.UA element | Touché content |
|---|---|
| §2 opener (lineage-anchor variant per CO-1) | First sentence: lineage opener per CO-1 above. |
| Recap of prior pipeline | Source dataset (Sahai 2021) + argument-scheme NLP lineage (Walton-Reed-Macagno + Macagno 2022 + jo2021schemes). Source: current `sections/02_background.tex` §2.1 + §2.2. |
| Prior performance and lessons learned | Adjacent benchmarks lineage: ARC (`habernal2018arc`), Propaganda (`dasanmartino2019propaganda`), Political fallacy (`goffredo2022political`), Multitask (`alhindi2022multitask`), MAFALDA (`helwe2024mafalda`), Pan zero-shot (`pan2024zeroshotfallacy`). Source: current §2.1. |
| Tables 1–2 (rankings + lessons) | NO equivalent for Touché (no prior submission). Skip. |
| Final lesson sentences with hedges | Closing sentence of §2: encoder-vs-LLM trade-off as open question, motivating the §4 comparison. Source: current `sections/02_background.tex` last sentence (post-length-pass merge). |

**Note:** the merged §2.3 (encoder-vs-LLM) and §2.4 (Touché lab history → moved to §3.1) from the length-pass round STAY merged; do not re-expand.

### §3 Methodology — multi-paragraph changes-this-year

| BIT.UA element | Touché content |
|---|---|
| Opening: name the year's theme | First sentence: *"Our pipeline this year is a single RoBERTa-large encoder fine-tuned per (subtask, track), with three input streams (real, synthetic, pseudo-labelled) mixed at fixed loss weights into one training loop."* (or analogous). |
| Paragraph 1: inference engine + model lineup | Touché analogue: pipeline overview + Figure 1. Source: current `sections/04_system.tex` §4.1 + Fig 1 TikZ. |
| Paragraph 2: ensemble/summarisation strategy | Touché analogue: backbone-selection narrative (RoBERTa-large chosen; DeBERTa-v3 abandoned; ModernBERT underperformed). Source: current §4.2. Numeric anchors: ST1 base 0.689→0.716 etc. (inventory §E); DeBERTa failure log (inventory §F: 4 NaN + 7 collapse). |
| Paragraph 3: fine-tuning + LoRA | Touché analogue: synthetic data augmentation. Source: current §4.4 + R2 verbatim sentence (inventory §Q): *"We generated synthetic contrastive pairs using a frontier LLM via API."* Inventory §J (synth batch inventory) + ablation deltas from inventory §C. |
| Paragraph 4: prompts forward-pointer | Touché analogue: pseudo-labelling under the leak-audit gate + threshold calibration. Source: current §4.5 + §4.7. Inventory §H (pe-class evidence chain) + §I (leakage chain `exp_0062`/`exp_0081`). Closing sentence forward-points to Appendix A: *"The exact synthetic-pair generation prompts can be seen in Appendix A."* |
| (BIT.UA does not have a separate "LLM detour" paragraph; we add one) | Touché adds: §3.5 LLM detour. Source: current §4.6. Numeric anchors: Qwen2.5-3B 0.467, 0.529; Qwen2.5-7B 0.851±0.114; Qwen2.5-32B 0.841 (CV stale); Qwen3-4B-Thinking 0.963/0.448 (inventory §D). BIT.UA's "Despite our belief in the quality of these summaries, their performance did not surpass that of the single-model system" is the register target. **Closing sentence (CO-6 locked):** *"…the CoT-LoRA configurations we evaluated did not show a validated benefit over the encoder within our compute envelope; our final submission (`exp\_0220`) populates all six slots with RoBERTa-large systems."* — folds the R3-§5.5 locked verbatim into the LLM-detour close, where `exp_0220` is structurally load-bearing per §14 carve-out (b). |

### §4 Results — per-subtask layout (CO-2)

| BIT.UA element | Touché content |
|---|---|
| §4 opening paragraph (no table on first line) | *"In this section, we report the dev-partition and cross-validation results for our submitted systems and the wider grid of backbones and recipes we evaluated. All numbers are macro-averaged F1 on the held-out development partition unless otherwise noted; the test-set leaderboard was not released at the time of writing."* |
| §4.1 Validation Results (BIT.UA: hyperparam-sweep table, prose walk-through) | §4.1 Validation Results — **opens with R1-locked sentence per CO-4(iii):** *"Following the official Touché 2026 release, we hold out a stratified development partition from the published training data; the ST3 partition contains 473 examples."* Plus the immediately-following sentence on ST1/ST2 dev sizes deferred to camera-ready (TODO retained inline + listed in `revision-questions.md` at Phase 5). Then Table 1 (the headline 6-cell table from current §5.1). Caption per CO-3. Anchor: inventory §A. |
| §4.2 Official Results — Phase A (BIT.UA: per-batch system summary + performance table) | **§4.2 Official Results — ST1.** Tables: ST1-row excerpt from ablation Table 2 (inventory §C rows for ST1) + ST1-related backbone rows (inventory §E). |
| §4.3 Official Results — Phase A+ | **§4.3 Official Results — ST2.** Tables: ST2-row excerpt from ablation Table 2 + ST2 enhanced/base comparison. Anchor: 0.730 base, 0.970 enhanced. |
| §4.4 Official Results — Phase B | **§4.4 Official Results — ST3.** Tables: ST3-row excerpt from ablation Table 2 + the full backbone-comparison table (inventory §D). Anchor: 0.523 base, 0.735 enhanced, 0.875 CV mean. Plus the schematic confusion figure (current Fig 2; inventory §M Fig 2). |
| BIT.UA closing of §4.4 (hedge about not over-interpreting) | Touché closing of §4.4: forward-point to §5.2 leakage discussion + §5.3 pe-ceiling. *"We turn to the interpretation of the enhanced-track numbers and the rare-class behaviour we observed on ST3 in §5."* |

**Table-introduction pattern (dossier §8):** each subsection opens with the BIT.UA 3-sentence frame (name table + describe column grouping + acknowledge the cross-cell variability), then walks the rows by setup descriptor (NOT by exp_id, per dossier §14 carve-out).

**Tables in §4 (consolidated per CO-7, decision locked):**
- **Table 1** (Validation, §4.1): the 6-row headline. Cells from inventory §A.
- **Table 2** (Ablation, at §4/§5 boundary): all 17 rows in inventory §C order preserved. Captioned "Per-subtask augmentation ablation".
- **Table 3** (Backbone comparison, §4.4): all 11 rows from inventory §D. Caption per CO-3.

The per-subtask split layout is **rejected** per CO-7: it would force breaking inventory §C row order, which is R1's standing integrity anchor.

### §5 Discussion and Future Work — mirror BIT.UA §5 (four-subsection layout per CO-5)

| BIT.UA element | Touché content |
|---|---|
| §5 opener: reflective frame about competition + evaluation | First sentence (target register, parallel to BIT.UA *"This year, we would like to reflect more broadly on the competition itself, its evaluation process, and critically assess our own submissions."*): *"This year we would like to reflect on the structural shape of the Touché 2026 evaluation — particularly the enhanced-track design — and on what our submissions can and cannot say about it."* |
| §5 mid-section: per-phase reflection | **§5.1 Encoder vs LLM at this scale** (current §6.1). Anchor: 0.875 vs 0.851±0.114 vs 0.841; std-overlap framing; operational vs statistical separation (R2 round-2 fix locked). |
| BIT.UA's metric-misalignment as the central reflective sub-argument | **§5.2 The enhanced-track leakage question** (current §6.2). **Locked structure (inventory §R): four-part (a)/(b)/(c)/(d) skeleton.** Locked verbatim (inventory §Q): "we cannot rule out softer leakage from the rewriter's semantic choices." Locked anchor data: regex-mask F1 retained 0.937 vs unmasked 0.937 (inventory §I). Register adjustment: soften from the current "single biggest credibility risk in this paper" framing to BIT.UA's measured-share-an-observation register, while preserving the locked sentences and the structural scaffold. |
| BIT.UA's per-batch consistency reflection | **§5.3 The pe rare-class ceiling** (current §6.3). **Locked anchor (inventory §Q):** the pe = 0.573 sentence with the 0.19–0.24 gap framing. Source: current §6.3 (post-round-3 trim to 3-sentence form). |
| (no direct BIT.UA analogue — limitations folded in per CO-5) | **§5.4 Limitations** (current §7, six items as prose paragraphs in BIT.UA register, NOT a numbered list). The six prose paragraphs cover, in order: (1) dev/CV-only numbers (test set not released); (2) seed-variance — **P1-locked sentence** verbatim per inventory §Q; (3) enhanced-track leakage cross-reference to §5.2; (4) Qwen-32B CV stale at deadline; (5) per-class decoding-regime asymmetry (Appendix C); (6) TSV reconciliation caveat. **Closes with the inventory §N reflection** (locked per CO-6): *"Taken together, these limitations do not invalidate the headline numbers, but the enhanced-track results in particular should be read as upper bounds on what a fair pipeline would extract from the same inputs."* |
| §5 closing: concrete future-work bullets | Inline future-work paragraph (NOT bullets — BIT.UA uses prose). Four follow-ups: (i) leakage-controlled re-evaluation with separate decoder, (ii) larger CV budget on Qwen-32B fold split, (iii) per-class threshold sweep for ST3 base (currently only ST3 enh under thresholds), (iv) targeted pe-class data collection. Two of these are already in current §6.1 and §6.2 closing sentences (post-BIT.UA refresh); pull and consolidate. Placed AFTER §5.4 Limitations, as the natural closing of §5. |

### §6 Conclusion — single paragraph, BIT.UA register

| BIT.UA element | Touché content |
|---|---|
| Opening phase-by-phase recap | *"On the encoder side, RoBERTa-large with synthetic contrastive augmentation reached competitive results across all six (subtask, track) cells, with pseudo-labelling unlocking the enhanced track for ST2 and ST3. On the LLM side, the CoT-LoRA configurations we evaluated did not show a validated benefit over the encoder within our compute envelope, though the 7B 5-fold std and the deadline-stale 32B CV leave that comparison underpowered rather than closed."* |
| Hedge sentence | *"We remain cautious about the enhanced-track numbers in particular: the rewriter's label-aware design makes them easier to obtain than to defend, and we report them only alongside the base-track and 5-fold CV anchors."* |
| Forward-looking closing | *"Touché continues to be a productive platform for benchmarking argument-aware NLP, and we look forward to further contributions in future editions."* |

Source: current `sections/08_conclusion.tex` (156 words; BIT.UA §6 is ~180 words — already a close fit; mostly a register pass).

### Back matter

| BIT.UA element | Touché content | Source |
|---|---|---|
| Acknowledgments (one paragraph, funding only) | **DEFAULT: omit acks section entirely** per inventory §T.5. Phase-5 `revision-questions.md` surfaces the (a) omit vs (b) generic-thanks-the-lab choice. | inventory §T.5. |
| Declaration on Generative AI (one short sentence) | Keep current multi-sentence form unchanged. Inventory §S notes it is longer than BIT.UA's but matches CEUR-WS policy template. | inventory §S; locked. |
| References (numbered `[n]`, ceurart-style) | 33 entries in `references.bib`, 29 cited, 4 uncited-but-retained. Inventory §P. **Frozen.** | inventory §P. |
| Appendix A — Prompts (verbatim text in body font, labelled "Prompt 1", "Prompt 2", …) | Current Appendix A with TODO placeholders for synth-generation prompt + pe-targeted variant + pseudo-label selection note. Inventory §O. Format: tighten to BIT.UA's body-font-not-code-block convention. | inventory §O App A. |
| Appendix B — Phase A trained models table | Current Appendix B Hyperparameter Grid. Format target: BIT.UA Table 10 wide tabular with hyphens for carried-over cells. Inventory §O App B. | inventory §O App B. |
| (no BIT.UA analogue) | **Appendix C — Per-class F1** (current draft Appendix B). Retain. Inventory §G + §O App C. | inventory §G. |
| (no BIT.UA analogue) | **Appendix D — Citation graph** (current draft Appendix C). Retain. Inventory §M Fig 3 + §O App D. | inventory §M Fig 3. |

---

## Part 3 — BIT.UA elements with NO Touché analogue (and justification)

| BIT.UA element | Why no analogue |
|---|---|
| §2's self-recap of prior team submission | Sole-author, first submission to this lab — there is no prior submission to recap. Replaced by lineage section per CO-1. |
| §4's "Best Competitor" and "Median" rows in Tables 5/7/9 | Test set unreleased; no leaderboard reference available. Per CO-3. |
| §3's "we switched our inference engine" paragraph | We did not switch infrastructure during the project; the inference stack was HuggingFace `transformers` throughout. The Methodology section narrates configuration changes (backbones tried, augmentation versions, threshold calibration) instead. |
| Acknowledgments naming an FCT grant with DOI | Sole-author submission, no grant funding. Default to omit; see inventory §T.5. |

---

## Part 4 — Touché content with NO direct BIT.UA position (and proposed placement)

| Touché content | Why no direct BIT.UA position | Proposed placement |
|---|---|---|
| The §6.2 four-part (a)/(b)/(c)/(d) leakage scaffold | BIT.UA's metric-misalignment subargument is structurally similar but not labelled; ours is explicitly labelled per inventory §R lock. | Placed as the labelled mid-section of §5 (the new Discussion and Future Work), structurally parallel to BIT.UA's metric-misalignment paragraph. |
| The §6.3 pe rare-class anchor sentence (locked verbatim) | BIT.UA does not have an equivalent class-level deep-dive — its per-batch reflection is closest in function. | Placed in §5.3 as the dedicated pe-class subsection, after §5.2 leakage. |
| The DeBERTa-v3-large NaN narrative (inventory §F) | BIT.UA does not have a single equivalent abandoned-route narrative at that scale. | Folded into §3 Methodology, paragraph 2 (backbone selection), as a sub-argument explaining the RoBERTa-large choice. |
| Appendices C (per-class F1) and D (citation graph) | BIT.UA has only Appendices A and B. | Added as App C and App D; rationale: per-class F1 is the load-bearing evidence for the §5.3 pe-ceiling claim; citation graph is the audit-trail figure that ties references to pipeline components. Both are retained from current draft per inventory §O. |
| The Locked verbatim sentences (inventory §Q, 8 items) | BIT.UA has no analogue concept — those locks are internal to our review-loop discipline. | Each sits in its sourced section; the Phase 2 rewriter preserves byte-identical content. |
| The current draft's §3 Task and Data (three subsections) (CO-4) | BIT.UA has no Task-and-Data section — equivalent material distributed across §1, §3, and §4.1. | (i) §3.1 subtasks → §1 P2 high-level mention + §3 Methodology paragraph 1 (full scaffolding before streams-into-mini-batch sentence). (ii) §3.2 base/enhanced tracks → §3 Methodology paragraph 1, immediately after subtask description; the §5.2 leakage scaffold depends on this setup being established here. (iii) §3.3 dataset statistics with R1 lock → §4.1 Validation Results opening paragraph (structurally parallel to BIT.UA's *"Validation was conducted on the first and second batches of the 2024 dataset"* sentence). Current `sections/03_task_data.tex` content absorbed; file retired from active build. |
| The current draft's §7 Limitations (six numbered items) (CO-5) | BIT.UA has no §7 — limitations folded into §5. | §5.4 Limitations as a labelled mid-section of §5 (six prose paragraphs in BIT.UA register, NOT a numbered list). Closes with the §N reflection (CO-6). Six-section TOC preserved (no §7 in rewritten paper). |

---

## Phase 1 stop-condition checklist

Before Phase 1 closes and Phase 2 launches:

- [x] CO-1, CO-2, CO-3 surfaced as standalone visible rows in Part 1.
- [x] **CO-4, CO-5, CO-6, CO-7** added in response to senior-author Phase 1 sign-off (2026-05-25).
- [x] Section-by-section mapping covers every part of the BIT.UA TOC (Title / Abstract / Keywords / §1 / §2 / §3 / §4 / §5 / §6 / Acks / GenAI / References / App A / App B).
- [x] BIT.UA elements with no Touché analogue enumerated and justified.
- [x] Touché content with no direct BIT.UA position enumerated and placed — including the full §3 Task and Data fold-in detail.
- [x] All locked items from inventory §Q and §R cross-referenced to their target Phase-2 location.
- [x] Open factual questions (inventory §T) retained, not silently resolved.
- [x] CO-7 lock supersedes the prior "Decision deferred for Phase 2 review" line on table consolidation.

**Procedural notes for Phase 2 (senior-author Phase 1 sign-off, 2026-05-25):**
1. The Phase 2 rewriter MUST read `mapping.md` Part 1 (CO-1 through CO-7) before any prose work. The seven carry-overs are non-negotiable architecture, not stylistic preferences.
2. The `% from exp_NNNN` audit comments on every numeric claim survive into Phase 2 output regardless of whether the exp_id is rendered in prose, per the §14 carve-out. This is the only thing the Stage 4.5 integrity check resolves against, so the rewriter must not strip the audit comments during paragraph rewrites.

Phase 2 entry conditions met. Ready to launch.
