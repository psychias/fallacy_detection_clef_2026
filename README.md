# Fallacy Detection — Touché 2026 @ CLEF

**UZHCL_for_the_win** submission to the Touché 2026 Fallacy Detection lab at CLEF.

## Submission summary

- **Team:** UZHCL_for_the_win
- **Contact:** Stylianos Psychias — `stylianos.psychias@uzh.ch`
- **Affiliation:** University of Zurich, Switzerland
- **Subtasks covered:** ST1 (binary fallacy detection), ST2 (eight-way fallacy classification), ST3 (four-way argument-scheme classification)
- **Tracks:** base and enhanced
- **Submitted runs:** 6 (one per subtask × track), all fine-tuned RoBERTa-large with synthetic contrastive augmentation; pseudo-labelling additionally applied on ST2 and ST3 enhanced tracks

## The six submitted runs

Six TIRA slots = 3 subtasks × {base, enhanced}. Each was trained by the `train.py`
in its `exp_id` folder under `experiments/`, driven by that folder's `config.json`.
All six prediction files were produced by **`scripts/generate_submissions.py`** and
are committed under `submissions/`, organised one folder per slot
(`submissions/st1_base/` … `submissions/st3_enhanced/`), each holding that run's
`config.json` + `train.py` + `metrics.json` + uploaded `.jsonl`.

| TIRA run | Subtask / Track | exp_id | Train script | Model | Input fields | Training data | Track |
|---|---|---|---|---|---|---|---|
| `st1_base`      | ST1 / base     | `exp_0161` | `experiments/exp_0161/train.py` | RoBERTa-large | base + `text_raw` (raw_concat, max_len 512) | real (938) + `synth_v002_st1st2` | **base** |
| `st1_first_run` | ST1 / enhanced | `exp_0035` | `experiments/exp_0035/train.py` | RoBERTa-large | `text_enhanced`, `argument_enhanced` (+title/parent) | real + `synth_v002_st1st2` | **enhanced** |
| `st2_base`      | ST2 / base     | `exp_0032` | `experiments/exp_0032/train.py` | RoBERTa-large | `text_base`, `argument_base` | real (465 fallacious) + `synth_v002_st1st2` | **base** |
| `st2`           | ST2 / enhanced | `exp_0072` | `experiments/exp_0072/train.py` | RoBERTa-large | `text_enhanced`, `argument_enhanced` | real + `synth_v002_st1st2` + `pseudo_v001_st2` | **enhanced** |
| `st3_base`      | ST3 / base     | `exp_0069` | `experiments/exp_0069/train.py` | RoBERTa-large | `text_base`, `argument_base` | real (473 non-fallacious) + `synth_v002_st3` | **base** |
| `st3`           | ST3 / enhanced | `exp_0073` | `experiments/exp_0073/train.py` | RoBERTa-large | `text_enhanced`, `argument_enhanced` | real + `synth_v002_st3` + `pseudo_v001_st3` ⚠ | **enhanced** |

### How to tell base from enhanced in the code

The track is set by `"track"` in each `config.json` and resolved in
`shared/data_utils.py:build_input_text`:

- **base** (lines 100–113): only `text_raw_title`, `text_raw_parent`, `text_base`,
  `argument_base` (claim + supports). The enhanced fields are never read.
- **enhanced** (lines 114–128): additionally uses `text_enhanced` and `argument_enhanced`.

`st1_base` (exp_0161) uses a base-legal variant, `build_raw_concat_text`
(`experiments/exp_0161/train.py:52–60`): the base fields **plus** `text_raw`. Both
`text_base` and `text_raw` are original Reddit fields, so this is a base-track run.


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
│   ├── synth_v002_st3/           — ST3 cross-scheme pairs (non-fallacious only)
│   ├── synth_v003_st3/           — ST3 rare-class top-up (NOT used by a submitted run)
│   ├── pseudo_v001_st{1,2,3}/    — pseudo-labelled predictions on the unlabelled pool
│   └── pe_synth_v001_st3/        — pe-targeted batch (exp_0103; NOT used by a submitted run)
├── shared/                       — library: data loading, training utilities,
│                                   eval, synth generation, calibration,
│                                   pseudo-labelling, k-fold splits
├── scripts/
│   ├── generate_submissions.py   — as-shipped inference for all 6 TIRA slots
│   ├── reproduce_submitted.py    — manifest-driven retraining of the runs
│   ├── reproduce_synth.py
│   ├── run_experiment.py
│   ├── _lib.py
│   └── manifests/{submitted_runs,synth_pipeline}.json
├── submissions/                  — one folder per TIRA slot (st1_base/ … st3_enhanced/),
│                                   each with that run's config.json + train.py +
│                                   metrics.json + uploaded prediction .jsonl
├── tools/                        — cot_audit, cv_aggregator, gen_kfold_splits
├── experiments/                  — one dir per cited experiment, each shipping
│                                   config.json + train.py (+ metrics.json)
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

## Reproducing

### 1. Install

```
python -m venv .venv
source .venv/bin/activate              # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` and `bitsandbytes` are platform-specific. Install a `torch` wheel that
matches your CUDA version *before* the `pip install -r` step if the default wheel
does not suit (`https://pytorch.org/get-started/locally/`).

### 2. Verify data is in place

```
ls data/touchefallacy_2026_train.jsonl       # the published Touché train set
ls data_synth/                                # batch dirs
```

The held-out **test set** is not redistributed — the lab keeps it private.
Retraining therefore targets the held-out development partition reported in §4.1.

### 3. Regenerate the submitted prediction files (as shipped)

```
python scripts/generate_submissions.py
```

Loads each `experiments/<exp_id>/ckpt/best`, runs non-routed inference on the test
pool, and writes `submissions/submission_<subtask>_<track>.jsonl`. Requires the lab
test file `data/touchefallacy_2026_test_task.jsonl` (held by the organizers; not
bundled here) and the trained checkpoints (not committed — retrain to regenerate).
ST1/ST2/ST3 decode with argmax (the ST3 per-class thresholds from exp_0145 are a
dev-partition analysis only, not applied to the uploaded files).

### 4. Retrain the runs

```
python scripts/reproduce_submitted.py
```

Runs the cells in `scripts/manifests/submitted_runs.json` in order, then the ST3
threshold sweep `exp_0145`. Each run takes ≈12–30 min on a single A100-40GB.

```
python scripts/reproduce_submitted.py --only exp_0073
python scripts/run_experiment.py exp_0073        # equivalent
```

Reproduction targets (dev partition, §4.1 Table 1):

| Cell           | exp\_id     | F1 (single-fold dev) | F1 (5-fold CV) |
| -------------- | ----------- | -------------------- | -------------- |
| ST1 base       | `exp_0161`  | 0.761                | —              |
| ST1 enhanced   | `exp_0035`  | 0.914                | —              |
| ST2 base       | `exp_0032`  | 0.730                | —              |
| ST2 enhanced   | `exp_0072`  | 0.970                | —              |
| ST3 base       | `exp_0069`  | 0.523                | —              |
| ST3 enhanced   | `exp_0073`  | 0.735                | 0.875          |

Re-runs should land within seed noise; large divergences usually signal a
`transformers`/`torch` version gap (see `requirements.txt`).

**ST3-enhanced seed variance.** `exp_0073` is not bit-reproducible across GPUs or
library versions even with a fixed seed. A five-seed re-run on the *same* fixed dev
split (seeds 42–46) gives dev macro-F1 **0.720 ± 0.015** (range 0.700–0.741), so the
submitted **0.735** is a favourable draw — see §4.4 / §5.3 of the paper. The per-seed
scores and reconstructed confusion matrices are in
`experiments/exp_0073/seed_results.jsonl`, and `scripts/colab_rerun_exp0073.ipynb`
reproduces the recovery on a Colab GPU. `shared/train_utils.set_seed` now enables
deterministic algorithms (`use_deterministic_algorithms(warn_only=True)` + cuDNN),
which makes training bit-reproducible on fixed hardware; set `EXP_SEED` to vary only
the training seed (the dev split is held fixed by `make_splits`).

### 5. (Optional) Regenerate synthetic / pseudo-label data

The committed `data_synth/` artefacts back the reported numbers. Synthetic
generation is non-deterministic (LLM via OpenRouter):

```
export OPENROUTER_API_KEY=sk-or-...
python scripts/reproduce_synth.py                  # synth batches
```

Pseudo-labels are not LLM-driven; each batch is the top-K confident self-predictions
of the matching real-only enhanced checkpoint on the unlabelled pool
(ST3 ← `exp_0066`, ST2 ← `exp_0065`):

```
python shared/pseudo_label.py \
    --exp_dir experiments/exp_0066 \
    --top_k 30 \
    --output data_synth/pseudo_v001_st3/data.jsonl
```

## Synthetic / pseudo-label data and the ST3 training set

- `synth_v002_st3` contains only **non-fallacious** scheme examples (`fallacy_exists=0`).
- `synth_v003_st3` and `pe_synth_v001_st3` were generated for experiments but are
  **not referenced by any of the six submitted configs**.
- `pseudo_v001_st{2,3}` are used only by the ST2/ST3 **enhanced** runs.

> ⚠ **ST3-enhanced pseudo caveat.** `pseudo_v001_st3` is drawn from the *full*
> unlabelled pool and stamped `fallacy_exists=0` by `shared/pseudo_label.py:208–214`
> (the candidate set is not pre-filtered to non-fallacious). A check against the gold
> test annotations shows 66 of its 120 records are actually fallacious. So `st3`
> (exp_0073) mixes a small set of fallacious comments — relabeled as scheme examples —
> into otherwise non-fallacious ST3 training. `st3_base` (exp_0069) trains on
> non-fallacious arguments only (real filtered by `fallacy_exists==0` in
> `shared/data_utils.py:159–160` + `:83–86`; synth is non-fallacious by construction).

## Status

Working-notes submission, updated with the official Touché 2026 test-set results
(§4.2, Table 2): **first place on ST2-base and ST3-base**, second on both ST1 tracks.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The
notebook paper itself is licensed CC BY 4.0 per the CEUR-WS proceedings. The Reddit
fallacy data under `data/` is redistributed by the Touché 2026 lab (originally Sahai
et al., 2021) and remains subject to the lab's and the original dataset's terms.
