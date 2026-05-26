"""Thin wrapper - delegates to shared/train_qwen_cv.py."""
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

from shared.train_qwen_cv import run_cv
from shared.train_utils import write_traceback, write_status

if __name__ == "__main__":
    try:
        run_cv(script_dir)
    except Exception as e:
        write_traceback(script_dir, e)
        write_status(script_dir, "crashed", reason=str(e))
        raise
