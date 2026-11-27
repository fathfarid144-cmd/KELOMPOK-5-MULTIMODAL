"""Shared, environment-aware paths and defaults for the project.

Override paths with MM_* environment variables. Dataset files are supplied
separately and are intentionally not included in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else default
    return path.resolve()


# Colab users can set MM_DATA_DIR to the mounted Drive bundle. Local runs
# default to the ignored repository data/ directory.
DATA_DIR = _path_from_env("MM_DATA_DIR", PROJECT_ROOT / "data")
CACHE_DIR = _path_from_env("MM_CACHE_DIR", PROJECT_ROOT / "cache")
RESULTS_DIR = _path_from_env("MM_RESULTS_DIR", PROJECT_ROOT / "results")

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
IMG_DIR = DATA_DIR / "images" if (DATA_DIR / "images").is_dir() else DATA_DIR

SEED = int(os.environ.get("MM_SEED", "0"))
DEVICE = os.environ.get("MM_DEVICE", "cpu")

IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL_EN = IMAGE_MODEL
TEXT_MODEL_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def prepare_directories(milestone: str | None = None) -> Path:
    """Create cache and result directories; return the selected result path."""
    output_dir = RESULTS_DIR / milestone if milestone else RESULTS_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_dataset(require_images: bool = True) -> list[Path]:
    """Raise a clear error for missing bundle files and return checked paths."""
    required = [TRAIN_CSV, VAL_CSV]
    if require_images:
        required.append(IMG_DIR)
    missing = [path for path in required if not path.exists()]
    if missing:
        expected = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Dataset bundle is incomplete: {expected}. Set MM_DATA_DIR to the "
            "directory containing train.csv, val.csv, and images/."
        )
    return required
