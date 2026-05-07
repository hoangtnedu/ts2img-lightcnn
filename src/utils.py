from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import yaml


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """
    Return project storage root.

    On Colab, experiment artifacts are stored in Google Drive so that checkpoints
    survive GPU/runtime interruption. On a laptop, artifacts are stored in the
    current repository folder.
    """
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        root = drive_root / "ts2img-lightcnn"
        root.mkdir(parents=True, exist_ok=True)
        return root
    return Path.cwd()


def prepare_run_dirs(
    project_root: Path,
    dataset: str,
    representation: str,
    model_type: str,
    seed: int,
) -> Dict[str, Path]:
    run_name = f"{dataset}_{representation}_{model_type}_seed{seed}"
    run_dir = project_root / "runs" / run_name
    dirs = {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "backup": run_dir / "backup",
        "logs": run_dir / "logs",
        "results": project_root / "results",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_environment() -> None:
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Current working directory:", os.getcwd())
