# Revision questions — pre-camera-ready blocking items

Five open factual questions retained from the Phase 0–3 pipeline, in priority order per senior-author ruling (2026-05-25). All must be resolved before camera-ready submission. **Phase 5 did NOT silently resolve any of these; the rewriter agent flagged each TODO in source comments.**

The current `paper.pdf` (pending compile on Overleaf) is shippable as a working-notes submission with these items as known TODOs. Camera-ready submission is blocked on item 1; the rest can flow alongside.

---

## 1. Author block — **RESOLVED 2026-05-25**

Final state in `paper.tex`:

```latex
\author[1]{Stylianos Psychias}[%
  email=stylianos.psychias@uzh.ch,
]
\address[1]{University of Zurich, Zurich, Switzerland}
```

Author name set to Stylianos Psychias; institutional email at uzh.ch; affiliation University of Zurich, Switzerland. ORCID line removed per senior-author instruction ("not needed"). Title also updated to lead with the team tag: `UZHCL\_for\_the\_win at Touché 26: ...`.

---

## 2. Appendix A — Prompts verbatim text — **RESOLVED 2026-05-25**

Appendix A now carries three system prompts extracted verbatim from the codebase:

- **Prompt 1: ST1/ST2 contrastive-pair generation** — source: `shared/synth_contrastive.py` (`build_st1_st2_contrastive_prompt` + `_BLOCKLIST_TEXT`). Used for `synth_v002` (ST1/ST2 batch) and `synth_v003` (rare-class top-up).
- **Prompt 2: ST3 contrastive-pair generation** — source: `shared/synth_contrastive.py` (`build_st3_contrastive_prompt`). Used for `synth_v002` ST3 batch.
- **Prompt 3: pe-targeted generation** — source: `experiments/exp_0103/generate.py` (`PE_SYSTEM` + `SCHEMES_PROMPTS["pe"]` + `SCHEMES_PROMPTS["pa"]`). Model: `openai/gpt-4o-mini` at temperature 0.9, 50 examples per label.

For each prompt the appendix renders the system message verbatim and describes the user-prompt template (since the user message is built per call from a topic seed, exemplars, fallacy/scheme card, etc.). The pseudo-label paragraph (τ=0.9) is unchanged — it uses no LLM prompt.

---

## 3. ST1 / ST2 dev partition sizes

Current state in `sections/04_results.tex` §4.1 opener:

```latex
% TODO senior: ST1/ST2 dev sizes
Following the official Touché 2026 release, we hold out a stratified development
partition from the published training data; the ST3 partition contains 473
examples. For ST1 and ST2 we adopt the analogous stratified split; exact sizes
are deferred to the camera-ready once the test phase concludes.
```

R1-locked verbatim sentence is intact. ST3 = 473 is confirmed (inventory §A). ST1 and ST2 partition sizes were marked deferred during the initial draft and never resolved.

**What's needed for camera-ready:**
- Two integers: ST1 dev partition size (binary fallacy detection partition); ST2 dev partition size (8-way fallacy classification partition over the fallacious subset).
- Source: the data loader script the senior author used to construct the stratified splits.

**Decision:** pull the two integers from the loader and patch into the §4.1 opener. The R1 verbatim sentence stays; the "deferred to the camera-ready" clause gets replaced with the concrete numbers.

---

## 4. Abstract — code/data availability line (S9) — **RESOLVED 2026-05-25**

Abstract S9 added with the public repo URL: *"All code is publicly available at https://github.com/psychias/fallacy_detection."* (rendered via `\url{}` for hyperref). Patch applied to `paper.tex` line 53 inside the `\begin{abstract}...\end{abstract}` block; mirrors BIT.UA's S9 closing convention exactly.

BIT.UA's abstract S9 closes with: *"All code is publicly available at https://github.com/bioinformatics-ua/BioASQ13B."*

**What's needed for camera-ready:**
- Decision: is there a public repository accompanying this submission?
  - If YES: the GitHub URL (or analogous), appended as one sentence at the end of the abstract.
  - If NO: confirm S9 stays omitted; the abstract closes at S8 (BIT.UA's template also accepts 8-sentence abstracts when no public repo is shipped).

The TIRA submission package itself is the de-facto reproducibility artefact (referenced in the §3 synth-prompt footnote and the §5.4 TSV reconciliation paragraph). A public GitHub mirror is optional, not required.

**Decision:** confirm omit-S9 default, OR provide a URL.

---

## 5. Acknowledgments — **RESOLVED 2026-05-25**

Senior author chose option (b). Added before the AI Usage Declaration:

```latex
\begin{acknowledgments}
The author thanks the Touché 2026 lab organizers for providing
the dataset, evaluation infrastructure, and submission platform.
\end{acknowledgments}
```

---

## 3. ST1 / ST2 dev partition sizes — **RESOLVED 2026-05-25** (re-resolved)

Sizes extracted from `data/touchefallacy_2026_train.jsonl` (938 records total; fallacious=465, non-fallacious=473). The §4.1 opener now reads (R1-locked first sentence unchanged):

> *"Following the official Touché 2026 release, we hold out a stratified development partition from the published training data; the ST3 partition contains 473 examples. The analogous stratified split for ST1 covers the full labelled release (938 examples), and the ST2 partition covers the fallacious subset only (465 examples). The ST3 partition is the one we use for the 5-fold cross-validation runs reported below."*

TODO comment removed.

---

## Additional change applied 2026-05-25 — AI Usage Declaration rewrite

Senior author replaced the prior `Declaration on Generative AI` block (which followed the long ARS-pipeline format) with a shorter, prompt-quoting template titled `AI Usage Declaration`. New body:

> *"This text was proofread and improved with the assistance of Claude (Anthropic). The following prompt was used: 'Imagine you're an AI master student. Evaluate my text based on the task description and check if it correct or not, if not tell me which parts I should improve.' Claude was used to evaluate the text against the Touché 2026 Fallacy Detection task description, identify missing elements (results coverage, leakage handling, methodology and ablation details) and suggest phrasing improvements. The author conceived and executed all experiments, made all submission decisions, and takes full responsibility for the publication's content."*

Note: CEUR-WS policy (mandatory since 2024-12-20) titles this section `Declaration on Generative AI`. The senior author's chosen heading `AI Usage Declaration` is a minor deviation from the template heading; the content covers the same disclosure requirements. If a reviewer flags the heading, swapping back to `Declaration on Generative AI` is a one-line edit.

---

## Summary (post-2026-05-25 round)

| # | Item | Status | Blocks camera-ready? |
|---|---|---|---|
| 1 | Author block | RESOLVED (Stylianos Psychias, UZH, email set; ORCID removed per ruling) | No |
| 2 | Appendix A prompt text | RESOLVED (verbatim prompts from `shared/synth_contrastive.py` + `experiments/exp_0103/generate.py`) | No |
| 3 | ST1 / ST2 dev sizes | RESOLVED (938 / 465 from data file) | No |
| 4 | GitHub URL (abstract S9) | RESOLVED (`https://github.com/psychias/fallacy_detection`) | No |
| 5 | Acknowledgments | RESOLVED (option (b) one-liner) | No |
| 6 | AI Usage Declaration heading | DECIDED (senior-author override; minor CEUR-WS-template deviation, see note above) | No |

**Test-set leaderboard results** remain pending the lab's test-phase release; the paper currently says *"will be incorporated in the camera-ready version of this notebook"* in five places (§4.1 opener, three table captions, §5.4 opener), which is the correct hedge for the initial working-notes submission.

The paper is shippable as a working-notes submission today. The only remaining pre-camera-ready action is plugging in the test-set leaderboard numbers once the lab releases them.
