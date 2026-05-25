"""Load and normalize project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import yaml


def load_slang_map(slang_map_path: str | Path) -> dict[str, str]:
    """Load slang mappings from CSV file."""
    slang_map = {}
    path = Path(slang_map_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Slang map file not found: {path}")
    
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row and "slang" in row and "meaning" in row:
                slang_map[row["slang"]] = row["meaning"]
    
    return slang_map


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

    # Load slang_map from CSV file
    slang_map_path = config["paths"].get("slang_map_file")
    if slang_map_path:
        slang_map = load_slang_map(slang_map_path)
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        config["preprocessing"]["slang_map"] = slang_map

    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create configured data and output directories if they do not exist."""
    for key in ("processed_dir", "figures_dir", "metrics_dir", "models_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
