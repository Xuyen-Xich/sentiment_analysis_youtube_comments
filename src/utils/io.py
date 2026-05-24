"""Reusable I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
    """Read a CSV file with robust defaults for social-media text."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    return df


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as pretty JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
