"""
scripts/_lib.py — shared helpers for the reproduction scripts.

Single responsibility:
  - resolve the repo root,
  - load a JSON manifest,
  - run one experiment dir as a subprocess.

Everything else (which runs, in what order, with what prerequisites)
lives in manifests/ — adding a run does not require editing this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


@dataclass(frozen=True)
class RunResult:
    exp_id: str
    returncode: int
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def load_manifest(name: str) -> dict:
    """Read scripts/manifests/<name>.json."""
    path = MANIFESTS_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def experiment_script(exp_id: str) -> Path:
    """
    Locate the runnable script in experiments/<exp_id>/.
    Each exp dir ships exactly one of: train.py, generate.py, predict.py,
    sweep.py. Raises FileNotFoundError if the dir or script is missing.
    """
    exp_dir = EXPERIMENTS_DIR / exp_id
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"experiment dir not found: {exp_dir}")
    for candidate in ("train.py", "generate.py", "predict.py", "sweep.py"):
        script = exp_dir / candidate
        if script.is_file():
            return script
    raise FileNotFoundError(
        f"no runnable script found in {exp_dir} "
        f"(expected one of train.py/generate.py/predict.py/sweep.py)"
    )


def run_experiment(exp_id: str, extra_env: dict | None = None) -> RunResult:
    """
    Execute experiments/<exp_id>/<script>.py as a subprocess.

    FALLACY_WORKSPACE is set to the repo root so shared/data_utils.py can
    locate data/ and data_synth/ without a manual export. Returns a
    RunResult; does not raise on non-zero exit.
    """
    script = experiment_script(exp_id)
    env = os.environ.copy()
    env["FALLACY_WORKSPACE"] = str(REPO_ROOT)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)

    print(f"[run] {exp_id}: {script.relative_to(REPO_ROOT)}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        env=env,
    )
    elapsed = time.monotonic() - started
    result = RunResult(exp_id=exp_id, returncode=completed.returncode, elapsed_s=elapsed)
    status = "OK" if result.ok else f"FAILED (exit {result.returncode})"
    print(f"[run] {exp_id}: {status} in {elapsed:.1f}s", flush=True)
    return result


def require_env(*names: str) -> None:
    """Fail fast if a required env var is missing."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(
            f"error: required environment variable(s) not set: {', '.join(missing)}"
        )
