"""Dataset validation checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_csv, save_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def validate_data(config: dict[str, Any]) -> dict[str, Any]:
    """Validate schema, missingness, duplicates, and abnormal values."""
    raw_path = Path(config["paths"]["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    LOGGER.info("Loading data for validation: %s", raw_path)
    df = read_csv(raw_path, sample_size=config["data"].get("sample_size"))
    required = config["data"]["required_columns"]
    optional = config["data"].get("optional_columns", [])
    expected_sentiments = set(config["data"].get("sentiment_labels", []))

    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    duplicate_count = int(df.duplicated().sum())
    duplicate_comment_ids = None
    if "CommentID" in df.columns:
        duplicate_comment_ids = int(df["CommentID"].duplicated().sum())

    report: dict[str, Any] = {
        "path": str(raw_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "required_columns_present": {column: column in df.columns for column in required},
        "optional_columns_present": {column: column in df.columns for column in optional},
        "missing_values": df.isna().sum().astype(int).to_dict(),
        "missing_percent": (df.isna().mean() * 100).round(4).to_dict(),
        "duplicate_rows": duplicate_count,
        "duplicate_comment_ids": duplicate_comment_ids,
        "sentiment_distribution": df["Sentiment"].value_counts(dropna=False).to_dict(),
        "category_distribution_top20": df["CategoryID"]
        .value_counts(dropna=False)
        .head(20)
        .to_dict(),
    }

    invalid_sentiments = sorted(set(df["Sentiment"].dropna().astype(str)) - expected_sentiments)
    report["invalid_sentiment_labels"] = invalid_sentiments
    report["empty_comment_count"] = int(
        df["CommentText"].fillna("").astype(str).str.strip().eq("").sum()
    )

    for column in ("Likes", "Replies"):
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            report[f"{column.lower()}_negative_count"] = int((numeric < 0).sum())
            report[f"{column.lower()}_non_numeric_count"] = int(numeric.isna().sum())

    if "PublishedAt" in df.columns:
        parsed_dates = pd.to_datetime(df["PublishedAt"], errors="coerce")
        report["published_at_invalid_count"] = int(parsed_dates.isna().sum())
        report["published_at_min"] = parsed_dates.min()
        report["published_at_max"] = parsed_dates.max()

    output_path = Path(config["paths"]["metrics_dir"]) / "validation_report.json"
    save_json(report, output_path)
    LOGGER.info("Validation report saved to %s", output_path)
    return report
