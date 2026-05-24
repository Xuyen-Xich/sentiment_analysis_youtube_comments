"""Load and normalize project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "src/config/default.yaml") -> dict[str, Any]:
    """Load YAML configuration and resolve project-relative paths."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    root = Path.cwd()
    for key, value in config.get("paths", {}).items():
        config["paths"][key] = str((root / value).resolve())

    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create configured data and output directories if they do not exist."""
    for key in ("processed_dir", "figures_dir", "metrics_dir", "models_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
