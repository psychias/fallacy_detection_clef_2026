# Style dossier — BIT.UA at BioASQ 13B (CEUR-WS Vol-4038, paper_22)

Source: extracted text at `/tmp/bitua.txt` (pdftotext -layout, 810 lines). Cross-referenced to PDF section/table numbering. Use this dossier as the structural mould for the Touché rewrite.

---

## 1. Front matter

**Title pattern.** Two lines, comma-separated theme list after the venue tag:

> *BIT.UA at BioASQ 13B: Revisiting Evaluation,*
> *DPRF-Enhanced Retrieval and Fine-Tuned LLMs*

**Touché title (Phase-0 sign-off, option (a) per senior brief 2026-05-25):**

> *Psychias at Touché 26: Encoder Fine-Tuning,*
> *Synthetic Augmentation, and an LLM Detour*

with sub-title `\title[mode=sub]{Notebook for the Touché Lab at CLEF 2026}`.

The colon-and-comma-separated theme-list sub-title structure (`Title: Comma, Separated, Theme List`) is **confirmed** as the template. `Touché 26` carries the edition number in the same position `BioASQ 13B` does in BIT.UA; team-tag-as-surname is the right fit for a single-author submission.

**Author block.** ORCID + email per author, single affiliation footnote. Corresponding-author asterisk on first author.

---

## 2. Abstract — sentence-by-sentence template

The BIT.UA abstract is a 9-sentence narrative. Each sentence does a specific job:

| # | Function | BIT.UA exemplar |
|---|---|---|
| S1 | Domain-importance opener (no system, no numbers) | "Biomedical information retrieval and question answering are critical for navigating the vast and continually expanding body of biomedical literature." |
| S2 | Benchmark introduction | "The BioASQ Task B Challenge provides a valuable benchmark for developing and evaluating systems capable of retrieving relevant documents and generating high-quality answers to biomedical questions." |
| S3 | Participation statement | "This paper describes our participation in the thirteenth edition of the BioASQ challenge, focusing on Task B…" |
| S4 | Phase/subtask 1 — method summary | "For Phase A, we employed a hybrid two-stage retrieval pipeline combining BM25-based retrieval with transformer-based rerankers such as BioLinkBERT and PubMedBERT." |
| S5 | Phase/subtask 2 — method summary | "In Phase B, we used a range of large language models (LLMs) for answer generation, including OpenBioLLM, LLaMA Nemotron, and a custom fine-tuned Gemma-3 27B model." |
| S6 | Key insight (narratively framed, not numerically) | "A key insight from this year's participation was the persistent misalignment between automatic evaluation metrics and human-judged answer quality…" |
| S7 | High-level outcome (no per-cell numbers) | "For phase A our systems consistently achieved top rankings." |
| S8 | Discussion forward-pointer | "We discuss these outcomes in light of evaluation challenges and outline promising directions for future work." |
| S9 | Code availability | "All code is publicly available at https://github.com/bioinformatics-ua/BioASQ13B." |

**Bans for the Touché abstract:** no F1 numbers; no `(i)/(ii)/(iii)` contribution enumeration; no defensive caveat ahead of the key insight; no per-cell breakdown.

---

## 3. Keywords

5–7 keywords separated by `\sep`, all short noun phrases, no acronym expansion. BIT.UA used: *Information Retrieval, Dense Retrieval, Semantic Search, Large Language Model, Answer Generation, Pseudo Relevance Feedback*.

---

## 4. Introduction — 5-paragraph rhythm

P1 (domain + benchmark): same hedge as the abstract's S1+S2, slightly expanded. Last sentence: "The BioASQ Task B Challenge continues to provide a critical platform for benchmarking and advancing these technologies in the biomedical domain."

P2 (high-level system, citing prior work): "Our participation in the 13th BioASQ challenge [1] employed a hybrid two-stage retrieval pipeline…" Cites the lab overview (`[1]`) and the team's previous notebook (`[2]`).

P3 (the central tension — narrated as a finding from prior work, not as a defensive caveat): "However, a key finding from our previous work was the significant misalignment between automatic evaluation metrics, such as ROUGE-based F1 scores, and human evaluation of answer quality." Sets up the *why* behind methodology choices, in measured register. Touché analogue: the enhanced-track leakage question, as the structural reason for separate base/enhanced reporting and 5-fold CV anchoring.

P4 (how that tension reshaped this year's methodology): "These misleading conclusions motivated a fundamental rethink of our methodology for the 13th BioASQ challenge. We prioritized reproducibility and robustness…" Outcomes-first, not method-name-first.

P5 (theme statement + roadmap): one paragraph naming the year's theme ("A central theme of our work this year is evaluative uncertainty"), then a roadmap paragraph closing with "The rest of the paper is organized as follows. Section 2 describes our submissions from the previous year… Section 6 summarizes our conclusions and outlines directions for future work."

---

## 5. Section structure (template)

| # | Title | What it does |
|---|---|---|
| 1 | Introduction | 5-paragraph rhythm above |
| 2 | **Previous Work** | Recap of authors' own prior submission(s). Touché analogue: lineage of fallacy-detection NLP we situate ourselves inside (Sahai → Habernal → Da San Martino → Goffredo → Alhindi → MAFALDA → Pan → Jo). |
| 3 | Methodology | Narrative changes-this-year. Multiple short paragraphs, each focused on one change (inference engine, model lineup, ensembling, fine-tuning, prompts). Last sentence: forward-pointer to Appendix A for prompts. |
| 4 | Results | 4 subsections: §4.1 Validation, §4.2 Official Results — Phase A, §4.3 Phase A+, §4.4 Phase B. Touché analogue: §4.1 Validation, §4.2 Official — ST1, §4.3 Official — ST2, §4.4 Official — ST3 (or one combined per-subtask table per ST). |
| 5 | Discussion and Future Work | Opens with a reflective paragraph about the competition itself and evaluation. Mid-section: per-phase reflection. Closes with concrete future-work bullets (DPRF further work, ensemble strategies, prompt adaptation). |
| 6 | Conclusion | Single paragraph. Phase-by-phase recap + hedge + forward-pointer. |
|  | Acknowledgments | One paragraph; funding only. |
|  | Declaration on Generative AI | One short sentence: "During the preparation of this work, the authors used [tool] as a writing assistant." |
|  | References | Numbered [n], standard CEURART style. |
|  | Appendix A — Prompts | Verbatim prompt text in monospace blocks, labelled "Prompt 1", "Prompt 2", … |
|  | Appendix B — Training-config table | One wide table, models × hyperparameters (size, seed, epochs, sampler, trainer, ExPOS, warmup). |

---

## 6. Section openings — first-sentence template

- §2 opens with a flashback sentence: *"In our participation in BioASQ 12 Task B [2], our approach comprised…"*
- §3 opens by naming the year's theme: *"Our main changes this year focused on generation, which had been our weakest component in previous years…"*
- §4 opens with one introductory paragraph framing the section (no table on the first line): *"In this section, we describe the different models and configurations we evaluated as part of our participation…"*
- §4.1 opens with a validation-context sentence + result anchoring to prior year: *"Validation was conducted on the first and second batches of the 2024 dataset. For reference, our best scores from these batches were 0.4142 for Batch 1 (B1) and 0.4412 for Batch 2 (B2)…"*
- §4.2 opens with table-pointer: *"A summary of the systems submitted across each Phase A batch is presented in Table 4, with the performance results shown in Table 5."*
- §5 opens with a reflective frame: *"This year, we would like to reflect more broadly on the competition itself, its evaluation process, and critically assess our own submissions."*
- §6 opens with a phase-by-phase recap clause: *"In Phase A, our DPRF-based systems consistently performed better than our larger ensembles…"*

---

## 7. Section closings — last-sentence template

- §1 closes with the roadmap sentence pattern above.
- §3 closes with a forward-pointer: *"The prompts can be seen in Appendix A."*
- §4.x closes with a hedge: *"…suggesting limited significance in their comparative performance at this stage."*
- §5 closes with explicit hedging about not over-interpreting: *"We prefer to avoid drawing premature conclusions that may not align with qualitative assessments once they become available."*
- §6 closes with a forward-looking sentence: *"…BioASQ continues to be a valuable platform for rigorous benchmarking and reflective system development, and we look forward to further contributing in future editions."*

---

## 8. Table introduction pattern

1. **Sentence 1:** name the table + its purpose. *"A summary of the systems submitted across each Phase A batch is presented in Table 4, with the performance results shown in Table 5."*
2. **Sentence 2 (often):** describe the column grouping and what varies. *"Each system varies in the combination of models used, including different training epochs, samplers, and ensemble strategies."*
3. **Sentence 3 (often):** acknowledge the cross-batch instability honestly. *"Some techniques were developed between batches, which explains the evolving configurations across submissions, however we tried to keep most things stable between batches…"*
4. **Walk-through paragraphs:** one paragraph per batch column, opening with *"In Batch 1, we began by establishing baseline systems."* and naming systems by setup descriptor (System-1 = OpenBioLLM Prompt 3), not by exp/run id.

---

## 9. Tables — layout conventions

- **Table 1 / Table 5 (rankings + performance):** one column per batch; rows per system. Bold = best of own. Median + Best Competitor rows beneath.
- **Table 4 (systems summary):** one column per batch, rows per system slot, cells contain compact recipe strings ("Summ. N P5 (3)", "OpenBioLLM Prompt 5"). Numbers in parentheses = number of abstracts/snippets used.
- **Table 10 (training config, Appendix B):** wide; rows per (model, size, seed, epochs, sampler, trainer, ExPOS flag, warmup flag). Hyphen `-` for "carried over from row above". Compact, no horizontal rules between groups.
- **Captions:** 2–3 sentences. First sentence states what; second states what to look for ("Bold values represent our best submission"); occasionally a third hedges.

---

## 10. Voice and tense conventions

- **First-person plural ("we") throughout.** Solo-author papers using BIT.UA's template still use "we" as conventional academic register.
- **Past tense for completed experiments** ("We switched our inference engine"). **Present tense for findings + conclusions** ("DPRF appears conclusively better this year"). **Future / conditional for proposed work** ("would let us decide more cleanly").
- **Hedging is structural, not apologetic.** "We caution against overinterpreting…", "We refrain from in-depth analysis at this stage", "We remain cautious in interpreting the results before final gold standards…". The hedge follows the substantive claim — never precedes it.
- **No defensive lead-ins.** Substantive claim first; caveat second. *Not*: "The result carries a caveat: our score is 0.97." *But*: "Our score is 0.97; we caution against reading this as a straight system-quality statement without human evaluation."

---

## 11. Footnotes and citations

- Numeric citations `[n]`. References list in CEURART order-of-appearance.
- Footnotes (`1`, `2`, `3`…) used for URL-style endnotes: HuggingFace model URLs, dataset hosting, Unsloth links. Footnote text is a single URL or one short clause.
- Section labels mentioned in body text use *Section N* (not §N), occasionally *§* in deep cross-references.

---

## 12. Acknowledgments / GenAI

- **Acknowledgments:** one paragraph, funding only. Names individual grants with DOI where applicable. No "we thank X" name-checks of organizers.
- **Declaration on Generative AI:** one short sentence naming the tool and the task ("During the preparation of this work, the authors used ChatGPT as a writing assistant."). Touché paper keeps current wording, which is longer than BIT.UA's but matches the policy template — keep as-is unless cutting for length.

---

## 13. Appendix style

- **Appendix A — Prompts:** Each prompt is labelled "Prompt 1", "Prompt 2", …, followed by the verbatim prompt text in body font (not in a code box). Variable placeholders use `{var}` notation. Multi-line prompts run inline.
- **Appendix B — training config table:** wide tabular, no caption beyond a one-sentence header. Hyphens to indicate "carried-over" cells, reducing visual noise.

---

## 14. Anti-patterns identified in current Touché draft (to avoid)

- Front-loaded F1 numbers in the abstract.
- `(i)/(ii)/(iii)` contribution enumeration.
- "The central tension of this notebook is not which backbone won. It is that…" — too op-ed; BIT.UA never lectures the reader. Rephrase as a measured statement of the finding.
- Inline `\texttt{exp\_NNNN}` parentheticals chained through prose — BIT.UA never names runs by id; it names them by setup ("System-1, the OpenBioLLM-Prompt-3 configuration").
- "Headline result" framing that puts a single number before its context.
- Defensive lead-ins where the caveat precedes the substantive claim.

### §14 carve-out — exp_id-in-prose policy (Phase-0 sign-off, 2026-05-25)

**The §14 anti-pattern targets *gratuitous* run-id chains, not every in-prose mention. §Q (locked verbatim sentences) wins where the two rules conflict.**

The rewriter SHOULD strip `\texttt{exp\_NNNN}` from prose by default, replacing each parenthetical with a setup descriptor ("the RoBERTa-large + synth\_v002 recipe at sw=0.5", "the deadline-stale 32B run"). The rewriter MAY leave an exp_id in prose only where one of these two conditions holds:

- **(a)** the sentence is on the §Q (`content-inventory.md`) locked-verbatim list — including the §6.3 pe-anchor sentence with "`exp_0073` under the per-class thresholds from `exp_0145`" and the §5.5 final-submission sentence with "Our final submission (`exp_0220`)"; OR
- **(b)** the run id is structurally load-bearing for that sentence — e.g., the deadline-stale Qwen-32B caveat in §7 where `exp_0207` IS the entity being discussed, or the leak-audit gate `exp_0081` in a figure node label where the run id IS the figure's subject.

**Per-paragraph ceiling:** at most ONE in-prose exp_id mention per paragraph, even under (a) or (b). The §6.3 pe-anchor sentence is the one exception (it carries two — `exp_0073` and `exp_0145` — because the locked verbatim phrasing requires both); the rewriter treats this as a §Q-locked exception, not as a template for other paragraphs.

**Audit trail unchanged:** every numeric claim in body prose still carries its `% from exp_NNNN` end-of-line LaTeX comment regardless of whether the exp_id appears in rendered text. The audit comments are how the Stage 4.5 integrity check resolves claims to source rows.

This carve-out resolves the Phase-0 blocking clarification on the exp_id-in-prose tension.
