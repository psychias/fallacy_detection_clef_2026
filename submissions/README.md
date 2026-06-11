# Submitted runs


| Folder | Subtask / Track | exp_id | Model | Dev F1 |
|---|---|---|---|---|
| `st1_base/`      | ST1 / base     | exp_0161 | RoBERTa-large | 0.761 |
| `st1_enhanced/`  | ST1 / enhanced | exp_0035 | RoBERTa-large | 0.914 |
| `st2_base/`      | ST2 / base     | exp_0032 | RoBERTa-large | 0.730 |
| `st2_enhanced/`  | ST2 / enhanced | exp_0072 | RoBERTa-large | 0.970 |
| `st3_base/`      | ST3 / base     | exp_0069 | RoBERTa-large | 0.523 |
| `st3_enhanced/`  | ST3 / enhanced | exp_0073 | RoBERTa-large | 0.735 |

The same `config.json`/`train.py` also live under `experiments/<exp_id>/` (the
canonical location the reproduction tooling iterates).

`base` reads only original Reddit fields; `enhanced` additionally uses
`text_enhanced` / `argument_enhanced` (see `shared/data_utils.py:build_input_text`).
`st1_base` (exp_0161) uses a base-legal raw_concat input (base fields + `text_raw`).

To regenerate the prediction files, run `scripts/generate_submissions.py`; by
default it writes `submission_<subtask>_<track>.jsonl` to this directory's root
(the copies here were placed into the per-slot folders for clarity).

> ⚠ `st3_enhanced` (exp_0073) mixes a pseudo-labelled stream that includes
> fallacious comments relabelled as scheme examples; `st3_base` (exp_0069) trains
> on non-fallacious arguments only. See the repo-root README, "Synthetic /
> pseudo-label data and the ST3 training set".
