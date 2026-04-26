"""Shared helpers for the experiment scripts.

Each experiment writes a Parquet of raw rows to ``results/`` and a small
manifest JSON capturing the seed, sample size and key statistics so that the
acceptance-criterion checker can verify reproducibility offline.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def write_manifest(name: str, payload: dict[str, Any]) -> None:
    """Atomic write of ``results/<name>.manifest.json``."""
    target = RESULTS_DIR / f"{name}.manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    full = {
        "name": name,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(full, indent=2, sort_keys=True, default=str))
    os.replace(tmp, target)


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    """Atomic Parquet write under ``results/``."""
    target = RESULTS_DIR / f"{name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, target)
    return target
