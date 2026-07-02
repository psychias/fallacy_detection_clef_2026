"""
shared/train_utils.py — Training helpers: heartbeat, wall-clock guard,
checkpoint saving, model factory.
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil


def get_exp_dir(script_path: str = None) -> Path:
    """Get the experiment directory from the script location."""
    if script_path:
        return Path(script_path).parent.resolve()
    return Path.cwd()


def load_config(exp_dir: Path) -> dict:
    """Load config.json from experiment directory."""
    with open(exp_dir / "config.json", "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_status(exp_dir: Path, state: str, **extra):
    """Write/update status.json."""
    path = exp_dir / "status.json"
    now = datetime.now(timezone.utc).isoformat()
    if path.exists():
        with open(path, "r", encoding="utf-8-sig") as f:
            status = json.load(f)
    else:
        status = {}
    status["state"] = state
    if state == "running" and "started_at" not in status:
        status["started_at"] = now
    if state in ("done", "crashed", "stale"):
        status["ended_at"] = now
    status["last_heartbeat"] = now
    status.update(extra)
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


class Heartbeat:
    """Heartbeat helper — updates status.json periodically."""

    def __init__(self, exp_dir: Path, interval_s: float = 30.0):
        self.exp_dir = exp_dir
        self.interval_s = interval_s
        self.last_beat = 0.0

    def beat(self, step: int = None, extra: dict = None):
        now = time.time()
        if now - self.last_beat < self.interval_s:
            return
        self.last_beat = now
        info = {}
        if step is not None:
            info["step"] = step
        if extra:
            info.update(extra)
        write_status(self.exp_dir, "running", **info)


class WallClockGuard:
    """
    Guard that fires when time_budget_s - margin is exceeded.
    Scripts should check guard.exceeded() in the training loop.
    """

    def __init__(self, time_budget_s: int, margin_s: int = 60):
        self.start = time.time()
        self.deadline = self.start + time_budget_s - margin_s

    def elapsed(self) -> float:
        return time.time() - self.start

    def exceeded(self) -> bool:
        return time.time() >= self.deadline


def get_peak_vram_mb() -> float:
    """Get peak VRAM usage in MB (returns 0 if no GPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def count_params(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_traceback(exp_dir: Path, exc: Exception):
    """Write traceback.txt for a crashed experiment."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    with open(exp_dir / "traceback.txt", "w") as f:
        f.write("".join(tb))


def save_checkpoint(model, tokenizer, exp_dir: Path, label: str = "best"):
    """Save model checkpoint."""
    import torch
    ckpt_dir = exp_dir / "ckpt" / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))


def load_checkpoint(exp_dir: Path, label: str = "best"):
    """Return the path to a checkpoint directory."""
    return exp_dir / "ckpt" / label


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set all random seeds and request deterministic algorithms for
    reproducibility.

    cuDNN is pinned to its deterministic kernels and CUBLAS is configured for
    reproducible matmuls. ``use_deterministic_algorithms`` is enabled with
    ``warn_only=True`` so a run that hits an op without a deterministic CUDA
    kernel degrades to a warning rather than crashing -- this keeps the setting
    safe for every experiment in the repo. Bit-exact reproducibility still
    requires the same GPU and library versions; across hardware, expect the
    small run-to-run spread reported in the paper.
    """
    import random
    import numpy as np
    import torch
    # cuBLAS determinism must be requested before the CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older torch without warn_only, or an env that refuses the switch;
        # the cuDNN/CUBLAS settings above still apply.
        pass


def disk_free_gb(path: str = "/") -> float:
    """Return free disk space in GB for the filesystem containing *path*."""
    import shutil
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def check_disk(min_gb: float = 15.0, path: str = "/"):
    """Print disk usage and raise if free space is below *min_gb*."""
    free = disk_free_gb(path)
    import shutil
    total = shutil.disk_usage(path).total / (1024 ** 3)
    used = total - free
    print(f"[disk] {used:.1f}/{total:.1f} GB used, {free:.1f} GB free")
    if free < min_gb:
        raise RuntimeError(
            f"Disk space critically low: {free:.1f} GB free < {min_gb} GB minimum. "
            f"Clear HF cache or old checkpoints before continuing."
        )


def clear_hf_cache_for_model(model_name: str):
    """Remove cached files for a specific model from HF_HOME."""
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"
    if not hub_dir.exists():
        return
    # HF cache uses models--org--name format
    safe_name = "models--" + model_name.replace("/", "--")
    model_cache = hub_dir / safe_name
    if model_cache.exists():
        size_gb = sum(f.stat().st_size for f in model_cache.rglob("*") if f.is_file()) / (1024**3)
        print(f"[disk] Removing cached {model_name} ({size_gb:.1f} GB)")
        shutil.rmtree(model_cache, ignore_errors=True)
