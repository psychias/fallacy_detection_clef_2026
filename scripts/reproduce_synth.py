"""
scripts/reproduce_synth.py — regenerate the synthetic-data batches.

Default reproduction does NOT need this script: the committed
data_synth/ artifacts are byte-stable across runs and are what the
paper's reported numbers were trained on. Run this only when you want
to vary the generation prompt, model, or seed and study the effect.

Requires OPENROUTER_API_KEY in the environment.

    python scripts/reproduce_synth.py                 # all four batches
    python scripts/reproduce_synth.py --only exp_0103 # pe-targeted only

Exit code is 0 iff every requested batch returned 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so `scripts._lib` resolves when this file is run
# directly (python scripts/reproduce_synth.py ...), not only as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._lib import RunResult, load_manifest, require_env, run_experiment


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated exp_ids; only these are run")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = load_manifest("synth_pipeline")
    require_env(*manifest.get("requires_env", []))

    only_set = set(args.only.split(",")) if args.only else None
    steps = [s for s in manifest["steps"]
             if only_set is None or s["exp_id"] in only_set]
    if not steps:
        print("no steps selected after applying --only", file=sys.stderr)
        return 2

    results: list[RunResult] = []
    for step in steps:
        print(f"[synth] {step['exp_id']} → {step['produces']}")
        results.append(run_experiment(step["exp_id"]))

    failures = [r for r in results if not r.ok]
    if failures:
        print(f"\n{len(failures)} step(s) failed: "
              f"{', '.join(r.exp_id for r in failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
