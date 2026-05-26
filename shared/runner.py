"""
shared/runner.py — Queue runner: drain_forever() polls experiments/*/status.json,
picks queued work with satisfied dependencies, runs it as a subprocess.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Constants ─────────────────────────────────────────────────────────
POLL_INTERVAL = 5  # seconds between scans
DEFAULT_TIME_BUDGET = 540  # 9 min
GRACE_PERIOD = 30  # extra seconds for subprocess timeout
STALE_HEARTBEAT_S = 900  # 15 min — mark running experiments as stale

def get_gpu_vram_gb() -> float:
    """Return total GPU VRAM in GB, or 0 if no GPU."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_experiments_dir() -> Path:
    ws = Path(os.environ.get(
        "FALLACY_WORKSPACE",
        "/content/drive/MyDrive/fallacy_detection"
    ))
    return ws / "experiments"


def scan_experiments(exp_dir: Path) -> dict:
    """
    Scan all experiment directories and return a dict:
    {exp_id: {"status": {...}, "config": {...}, "path": Path}}
    """
    results = {}
    if not exp_dir.exists():
        return results
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("exp_"):
            continue
        status_path = d / "status.json"
        config_path = d / "config.json"
        if not status_path.exists():
            continue
        try:
            status = load_json(status_path)
            config = load_json(config_path) if config_path.exists() else {}
            results[d.name] = {
                "status": status,
                "config": config,
                "path": d,
            }
        except (json.JSONDecodeError, OSError) as e:
            print(f"[runner] Warning: could not read {d.name}: {e}")
    return results


def deps_satisfied(exp_info: dict, all_exps: dict) -> tuple[bool, str]:
    """
    Check if all depends_on for an experiment are done.
    Returns (satisfied: bool, reason: str).
    """
    deps = exp_info["config"].get("depends_on", [])
    for dep_id in deps:
        dep = all_exps.get(dep_id)
        if dep is None:
            return False, f"dep_missing: {dep_id}"
        dep_state = dep["status"].get("state", "")
        if dep_state == "done":
            continue
        if dep_state in ("crashed", "stale", "skipped"):
            return False, f"dep_failed: {dep_id}"
        # Still running or queued
        return False, f"dep_pending: {dep_id}"
    return True, ""


def pick_next(all_exps: dict) -> str | None:
    """
    Pick the oldest queued experiment with satisfied dependencies
    and sufficient GPU VRAM.  Returns exp_id or None.
    """
    gpu_vram = get_gpu_vram_gb()
    candidates = []
    for exp_id, info in all_exps.items():
        if info["status"].get("state") != "queued":
            continue
        # Skip experiments that need more VRAM than available
        min_vram = info["config"].get("min_vram_gb", 0)
        if min_vram > 0 and gpu_vram > 0 and gpu_vram < min_vram:
            continue
        satisfied, reason = deps_satisfied(info, all_exps)
        if satisfied:
            candidates.append(exp_id)
        elif reason.startswith("dep_failed"):
            # Mark as skipped
            info["status"]["state"] = "skipped"
            info["status"]["reason"] = reason
            info["status"]["ended_at"] = now_iso()
            write_json(info["path"] / "status.json", info["status"])
            print(f"[runner] Skipped {exp_id}: {reason}")

    if not candidates:
        return None
    # Sort by (priority, exp_id). Lower priority value runs first.
    # Default priority = exp number, so natural ordering is preserved.
    def sort_key(eid):
        pri = all_exps[eid]["config"].get("priority", int(eid.split("_")[1]))
        return (pri, eid)
    candidates.sort(key=sort_key)
    return candidates[0]


def run_experiment(exp_id: str, info: dict) -> str:
    """
    Run a single experiment. Returns final state: 'done', 'crashed', or 'stale'.
    """
    exp_path = info["path"]
    config = info["config"]
    kind = config.get("kind", "train")
    time_budget = config.get("time_budget_s", DEFAULT_TIME_BUDGET)
    timeout = time_budget + GRACE_PERIOD

    script = config.get("script", "train.py") if kind != "generate" else config.get("script", "generate.py")
    script_path = exp_path / script

    if not script_path.exists():
        print(f"[runner] ERROR: {script_path} not found!")
        info["status"]["state"] = "crashed"
        info["status"]["reason"] = f"{script} not found"
        info["status"]["ended_at"] = now_iso()
        write_json(exp_path / "status.json", info["status"])
        return "crashed"

    # Mark as running
    info["status"]["state"] = "running"
    info["status"]["started_at"] = now_iso()
    info["status"]["last_heartbeat"] = now_iso()
    write_json(exp_path / "status.json", info["status"])

    print(f"[runner] Starting {exp_id} ({kind}, budget={time_budget}s) ...")

    stderr_path = exp_path / "stderr.log"
    try:
        with open(stderr_path, "w") as stderr_file:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(exp_path),
                timeout=timeout,
                stdout=None,
                stderr=stderr_file,
            )

        # Print stderr if non-empty
        if stderr_path.exists() and stderr_path.stat().st_size > 0:
            stderr_text = stderr_path.read_text(errors="replace")[-2000:]
            print(f"[runner] stderr for {exp_id}:\n{stderr_text}")

        # Re-read status (the script may have updated it)
        status = load_json(exp_path / "status.json")
        if status.get("state") in ("done", "crashed"):
            print(f"[runner] {exp_id} finished: {status['state']}")
            return status["state"]

        # Script exited but didn't update status
        if result.returncode == 0:
            status["state"] = "done"
            status["ended_at"] = now_iso()
            write_json(exp_path / "status.json", status)
            print(f"[runner] {exp_id} done (exit 0)")
            return "done"
        else:
            status["state"] = "crashed"
            status["reason"] = f"exit_code={result.returncode}"
            status["ended_at"] = now_iso()
            write_json(exp_path / "status.json", status)
            print(f"[runner] {exp_id} crashed (exit {result.returncode})")
            return "crashed"

    except subprocess.TimeoutExpired:
        status = load_json(exp_path / "status.json")
        if status.get("state") == "done":
            return "done"
        status["state"] = "crashed"
        status["reason"] = f"timeout after {timeout}s"
        status["ended_at"] = now_iso()
        write_json(exp_path / "status.json", status)
        print(f"[runner] {exp_id} timed out after {timeout}s")
        return "crashed"

    except Exception as e:
        status = load_json(exp_path / "status.json")
        status["state"] = "crashed"
        status["reason"] = str(e)
        status["ended_at"] = now_iso()
        write_json(exp_path / "status.json", status)
        print(f"[runner] {exp_id} exception: {e}")
        return "crashed"


def drain_forever():
    """
    Main loop: scan for queued experiments, run the next one, repeat.
    On SIGINT, marks currently-running experiment as stale and exits.
    """
    exp_dir = get_experiments_dir()
    current_exp_id = None

    def handle_sigint(signum, frame):
        print(f"\n[runner] SIGINT received.")
        if current_exp_id:
            try:
                p = exp_dir / current_exp_id / "status.json"
                if p.exists():
                    s = load_json(p)
                    if s.get("state") == "running":
                        s["state"] = "stale"
                        s["reason"] = "SIGINT"
                        s["ended_at"] = now_iso()
                        write_json(p, s)
                        print(f"[runner] Marked {current_exp_id} as stale.")
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    print("[runner] === Touché 2026 Queue Runner ===")
    print(f"[runner] Experiments dir: {exp_dir}")
    print(f"[runner] Polling every {POLL_INTERVAL}s")
    print()

    cycle = 0
    while True:
        cycle += 1
        try:
            all_exps = scan_experiments(exp_dir)
        except OSError as e:
            print(f"[runner] Drive error during scan (retrying): {e}")
            time.sleep(POLL_INTERVAL * 2)
            continue

        queued = [eid for eid, info in all_exps.items()
                  if info["status"].get("state") == "queued"]
        running = [eid for eid, info in all_exps.items()
                   if info["status"].get("state") == "running"]
        done = [eid for eid, info in all_exps.items()
                if info["status"].get("state") == "done"]

        # Auto-mark stale experiments (heartbeat > 15 min old)
        for eid in running:
            info = all_exps[eid]
            hb = info["status"].get("last_heartbeat", "")
            if hb:
                try:
                    hb_dt = datetime.fromisoformat(hb)
                    age_s = (datetime.now(timezone.utc) - hb_dt).total_seconds()
                    if age_s > STALE_HEARTBEAT_S:
                        info["status"]["state"] = "stale"
                        info["status"]["reason"] = f"heartbeat {age_s:.0f}s old"
                        info["status"]["ended_at"] = now_iso()
                        write_json(info["path"] / "status.json", info["status"])
                        print(f"[runner] Marked {eid} as stale (heartbeat {age_s:.0f}s old)")
                except (ValueError, TypeError):
                    pass

        if cycle % 12 == 1:  # Print status every ~60s
            print(f"[runner] Cycle {cycle}: "
                  f"{len(queued)} queued, {len(running)} running, "
                  f"{len(done)} done, {len(all_exps)} total")

        exp_id = pick_next(all_exps)
        if exp_id:
            current_exp_id = exp_id
            info = all_exps[exp_id]
            try:
                final_state = run_experiment(exp_id, info)
            except OSError as e:
                print(f"[runner] Drive error running {exp_id}: {e}")
                time.sleep(POLL_INTERVAL * 2)
                current_exp_id = None
                continue

            # Clean up GPU memory between experiments
            if HAS_TORCH:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            current_exp_id = None
            time.sleep(1)  # Brief pause between experiments
        else:
            # No queued or running experiments — exit
            if not running:
                print("[runner] All experiments finished. Exiting.")
                break
            time.sleep(POLL_INTERVAL)
