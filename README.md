# Fallacy Detection — Touché 2026 @ CLEF

**UZHCL_for_the_win** submission to the Touché 2026 Fallacy Detection lab at CLEF.

## Submission summary

- **Team:** UZHCL_for_the_win
- **Contact:** Stylianos Psychias — `stylianos.psychias@uzh.ch`
- **Affiliation:** University of Zurich, Switzerland
- **Subtasks covered:** ST1 (binary fallacy detection), ST2 (eight-way fallacy classification), ST3 (four-way argument-scheme classification)
- **Tracks:** base and enhanced
- **Submitted runs:** 6 (one per subtask × track), all fine-tuned RoBERTa-large with synthetic contrastive augmentation; pseudo-labelling additionally applied on ST2 and ST3 enhanced tracks

## Repository layout

```
fallacy_detection/
├── README.md                     — this file
├── .gitignore
├── requirements.txt              — Python dependencies (Python 3.10, CUDA 12.1)
├── paper/                        — LaTeX sources for the notebook paper
│   ├── paper.tex
│   ├── ceurart.cls
│   ├── references.bib
│   ├── per_class_f1_st2_st3.csv
│   └── sections/
├── data/
│   ├── experiments.tsv           — canonical experiment-tracking artefact;
│   │                               every numeric claim in the paper is
│   │                               traceable to a row here via the
│   │                               `% from exp_NNNN` LaTeX comments
│   └── touchefallacy_2026_train.jsonl — published Touché 2026 training data
├── data_synth/                   — generated synthetic / pseudo-label batches
│   ├── synth_v002_st1st2/        — ST1/ST2 contrastive pairs (130 pairs)
│   ├── synth_v002_st3/           — ST3 cross-scheme pairs (110 pairs)
│   ├── synth_v003_st3/           — ST3 rare-class top-up
│   ├── pseudo_v001_st{1,2,3}/    — pseudo-labelled test pool (τ=0.9)
│   └── pe_synth_v001_st3/        — pe-targeted batch (exp_0103)
├── shared/                       — library: data loading, training utilities,
│                                   eval, synth generation, calibration,
│                                   pseudo-labelling, k-fold splits
├── tools/                        — cot_audit, cv_aggregator, gen_kfold_splits
├── experiments/                  — one dir per cited experiment (57 total),
│                                   each shipping config.json + train.py /
│                                   generate.py + metrics.json
├── scripts/                      — reproduction orchestrators (see below)
│   ├── _lib.py
│   ├── run_experiment.py
│   ├── reproduce_submitted.py
│   ├── reproduce_synth.py
│   └── manifests/
│       ├── submitted_runs.json
│       └── synth_pipeline.json
└── docs/                         — process artefacts from the writing pipeline
```

## Building the PDF

```
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Reproducing the paper's experiments

### 1. Install

```
python -m venv .venv
source .venv/bin/activate              # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` and `bitsandbytes` are platform-specific. Install a `torch` wheel
that matches your CUDA version *before* the `pip install -r` step if the
default wheel does not suit (`https://pytorch.org/get-started/locally/`).

### 2. Verify data is in place

```
ls data/touchefallacy_2026_train.jsonl       # the published Touché train set
ls data_synth/                                # 7 batch dirs
```

The held-out **test set** is not redistributed — the lab keeps it private
until the test phase concludes. Reproduction therefore targets the
held-out development partition reported in §4.1 of the paper.

### 3. Reproduce the six submitted runs

```
python scripts/reproduce_submitted.py
```

This runs `exp_0033`, `exp_0035`, `exp_0032`, `exp_0072`, `exp_0069`,
`exp_0073` in that order, then the threshold sweep `exp_0145`. Each run
takes ≈12–30 min on a single A100-40GB; full reproduction is ≈2 h.

Run a single cell instead:

```
python scripts/reproduce_submitted.py --only exp_0073
python scripts/run_experiment.py exp_0073        # equivalent
```

Reproduction targets, copied verbatim from §4.1 Table 1:

| Cell           | exp\_id     | F1 (single-fold dev) | F1 (5-fold CV) |
| -------------- | ----------- | -------------------- | -------------- |
| ST1 base       | `exp_0033`  | 0.734                | —              |
| ST1 enhanced   | `exp_0035`  | 0.914                | —              |
| ST2 base       | `exp_0032`  | 0.730                | —              |
| ST2 enhanced   | `exp_0072`  | 0.970                | —              |
| ST3 base       | `exp_0069`  | 0.523                | —              |
| ST3 enhanced   | `exp_0073`  | 0.735                | 0.875          |

Re-runs should land within seed noise; large divergences signal an
environment mismatch (most often a `transformers` or `torch` version
gap — see `requirements.txt` for the tested range).

### 4. (Optional) Regenerate synthetic data from scratch

The committed `data_synth/` artefacts back the reported numbers. Only
rerun generation if you want to study prompt or model variation; results
will differ run-to-run because the LLM is non-deterministic.

```
export OPENROUTER_API_KEY=sk-or-...
python scripts/reproduce_synth.py                  # all four batches
python scripts/reproduce_synth.py --only exp_0103  # pe-targeted only
```

Pseudo-labels are not LLM-driven and use the leak-audited checkpoint
from `exp_0081`:

```
python shared/pseudo_label.py \
    --exp_dir experiments/exp_0081 \
    --top_k 30 \
    --output data_synth/pseudo_v001_st3/data.jsonl
```

### 5. Add a new experiment

The orchestrators are manifest-driven. To add a run:

1. Add `experiments/exp_NNNN/{config.json,train.py}`.
2. Append an entry to `scripts/manifests/submitted_runs.json` (or to a
   new manifest of your own).
3. Re-run `python scripts/reproduce_submitted.py`.

No orchestrator code needs to change — the scripts iterate whatever the
manifest declares.

## Pipeline note

The paper sources in `paper/` were produced via an iterative
writing/review pipeline modelled on the BIT.UA at BioASQ 13B (CEUR-WS
Vol-4038, paper_22) notebook template. The pipeline artefacts in `docs/`
document the architectural decisions (CO-1 through CO-7), the locked
verbatim sentences (`docs/content-inventory.md` §Q), and the §5.2
four-part leakage scaffold that the rewrite preserved.

## Status

Initial working-notes submission. Test-set leaderboard results will be
incorporated in the camera-ready version of the notebook.
