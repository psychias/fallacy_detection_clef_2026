"""
scripts/reproduce_submitted.py — run the six submitted training runs.

Iterates scripts/manifests/submitted_runs.json in declared order. Each
run is a separate subprocess; failures are reported but do not stop the
remaining runs (so one broken slot does not block the other five).

After the six runs complete, post_run entries (currently: the ST3
threshold sweep exp_0145) execute if their dependencies succeeded.

    python scripts/reproduce_submitted.py
    python scripts/reproduce_submitted.py --only exp_0073
    python scripts/reproduce_submitted.py --skip exp_0072,exp_0073

Exit code is 0 iff every requested run returned 0.
"""

from __future__ import annotations

import argparse
import sys

from scripts._lib import RunResult, load_manifest, run_experiment


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated exp_ids; only these are run")
    p.add_argument("--skip", type=str, default=None,
                   help="comma-separated exp_ids; these are skipped")
    return p.parse_args(argv)


def filter_runs(runs: list[dict], only: str | None, skip: str | None) -> list[dict]:
    only_set = set(only.split(",")) if only else None
    skip_set = set(skip.split(",")) if skip else set()
    out = []
    for r in runs:
        if only_set is not None and r["exp_id"] not in only_set:
            continue
        if r["exp_id"] in skip_set:
            continue
        out.append(r)
    return out


def summarize(results: list[RunResult]) -> int:
    print()
    print("=" * 60)
    print("Reproduction summary")
    print("=" * 60)
    ok = sum(1 for r in results if r.ok)
    print(f"  Runs attempted: {len(results)}")
    print(f"  Succeeded:      {ok}")
    print(f"  Failed:         {len(results) - ok}")
    for r in results:
        flag = "ok" if r.ok else "FAIL"
        print(f"  [{flag}] {r.exp_id}  ({r.elapsed_s:.1f}s)")
    return 0 if ok == len(results) else 1


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = load_manifest("submitted_runs")

    runs = filter_runs(manifest["runs"], args.only, args.skip)
    if not runs:
        print("no runs selected after applying --only/--skip", file=sys.stderr)
        return 2

    results: list[RunResult] = []
    for entry in runs:
        results.append(run_experiment(entry["exp_id"]))

    succeeded_ids = {r.exp_id for r in results if r.ok}
    for post in manifest.get("post_run", []):
        deps = post.get("depends_on_exp", [])
        if any(d not in succeeded_ids for d in deps):
            print(f"[skip] {post['exp_id']}: prerequisite(s) failed ({deps})")
            continue
        results.append(run_experiment(post["exp_id"]))

    return summarize(results)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
