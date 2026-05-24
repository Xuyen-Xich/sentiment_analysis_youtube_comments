"""Preprocessing pipeline for cleaned train/validation/test files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.text_cleaning import clean_text
from src.utils.io import read_csv
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _choose_stratify_column(df: pd.DataFrame) -> pd.Series | None:
    """Choose a split column that keeps target classes represented."""
    category_counts = df["CategoryID"].astype(str).value_counts()
    if len(category_counts) > 1 and category_counts.min() >= 3:
        return df["CategoryID"].astype(str)

    sentiment_counts = df["Sentiment"].astype(str).value_counts()
    if len(sentiment_counts) > 1 and sentiment_counts.min() >= 3:
        return df["Sentiment"].astype(str)
    return None


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust calendar features from PublishedAt."""
    if "PublishedAt" not in df.columns:
        df["published_hour"] = 0
        df["published_dayofweek"] = 0
        df["published_month"] = 0
        return df

    dates = pd.to_datetime(df["PublishedAt"], errors="coerce")
    df["published_hour"] = dates.dt.hour.fillna(0).astype(int)
    df["published_dayofweek"] = dates.dt.dayofweek.fillna(0).astype(int)
    df["published_month"] = dates.dt.month.fillna(0).astype(int)
    return df


def preprocess_data(config: dict[str, Any]) -> dict[str, str]:
    """Clean text, add features, split data, and save processed CSV files."""
    raw_path = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading raw data from %s", raw_path)
    df = read_csv(raw_path, sample_size=config["data"].get("sample_size"))

    required = config["data"]["required_columns"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot preprocess because columns are missing: {missing}")

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["CommentText", "Sentiment", "CategoryID"])
    df["CommentText"] = df["CommentText"].astype(str)
    df = df[df["CommentText"].str.strip().str.len() >= config["data"]["min_comment_length"]]

    for column in ("VideoTitle", "CountryCode"):
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    for column in ("Likes", "Replies"):
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)

    slang_map = config["preprocessing"].get("slang_map", {})
    max_repeats = int(config["preprocessing"].get("max_repeated_chars", 2))
    LOGGER.info("Cleaning comment and title text")
    df["clean_comment"] = df["CommentText"].map(
        lambda text: clean_text(text, slang_map=slang_map, max_repeated_chars=max_repeats)
    )
    df["clean_title"] = df["VideoTitle"].map(
        lambda text: clean_text(text, slang_map=slang_map, max_repeated_chars=max_repeats)
    )
    df["model_text"] = (df["clean_comment"] + " title " + df["clean_title"]).str.strip()
    df["comment_length"] = df["clean_comment"].str.len()
    df["word_count"] = df["clean_comment"].str.split().map(len)
    df["title_length"] = df["clean_title"].str.len()
    df = _add_time_features(df)

    df["Sentiment"] = df["Sentiment"].astype(str)
    df["CategoryID"] = df["CategoryID"].astype(str)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(
        {"model_text": "", "clean_comment": "", "clean_title": "", "CountryCode": ""}
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=config["data"]["test_size"] + config["data"]["validation_size"],
        random_state=config["project"]["random_state"],
        stratify=_choose_stratify_column(df),
    )
    relative_val_size = config["data"]["validation_size"] / (
        config["data"]["test_size"] + config["data"]["validation_size"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_size,
        random_state=config["project"]["random_state"],
        stratify=_choose_stratify_column(temp_df),
    )

    paths = {
        "train": str(processed_dir / "train.csv"),
        "validation": str(processed_dir / "validation.csv"),
        "test": str(processed_dir / "test.csv"),
        "processed_full": str(processed_dir / "processed_full.csv"),
    }
    df.to_csv(paths["processed_full"], index=False, encoding="utf-8")
    train_df.to_csv(paths["train"], index=False, encoding="utf-8")
    val_df.to_csv(paths["validation"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8")

    LOGGER.info(
        "Saved processed splits: train=%s validation=%s test=%s",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return paths
