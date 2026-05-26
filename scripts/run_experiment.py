"""
scripts/run_experiment.py — run one experiment dir by id.

    python scripts/run_experiment.py exp_0033

Exits with the subprocess's return code so it composes cleanly in shells
and in the other reproduce_*.py orchestrators.
"""

from __future__ import annotations

import sys

from scripts._lib import run_experiment


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python scripts/run_experiment.py <exp_id>", file=sys.stderr)
        return 2
    return run_experiment(argv[1]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
