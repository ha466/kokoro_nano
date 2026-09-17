import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

kokoro_patched_path = str(REPO_ROOT / "kokoro_patched")
if kokoro_patched_path not in sys.path:
    sys.path.insert(0, kokoro_patched_path)

training_path = str(HERE)
if training_path not in sys.path:
    sys.path.insert(0, training_path)
