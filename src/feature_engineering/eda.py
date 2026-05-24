"""Exploratory data analysis and learning-oriented figures."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional plotting dependency
    sns = None

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _save_barplot(series: pd.Series, title: str, path: Path, top_n: int = 30) -> None:
    """Save a horizontal bar chart for top values."""
    values = series.value_counts().head(top_n)
    plt.figure(figsize=(10, max(4, 0.3 * len(values))))
    if sns:
        sns.barplot(x=values.values, y=values.index.astype(str), color="#2f80ed")
    else:
        plt.barh(values.index.astype(str), values.values, color="#2f80ed")
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _word_frequency(texts: pd.Series, top_n: int = 50) -> pd.DataFrame:
    """Build a simple word frequency table from cleaned text."""
    counter: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        counter.update(token for token in text.split() if len(token) > 1)
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def run_eda(config: dict[str, Any]) -> None:
    """Generate EDA charts, word frequencies, and optional word cloud."""
    processed_path = Path(config["paths"]["processed_dir"]) / "processed_full.csv"
    if not processed_path.exists():
        raise FileNotFoundError("Run preprocess before eda.")

    figures_dir = Path(config["paths"]["figures_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading processed data for EDA: %s", processed_path)
    df = pd.read_csv(processed_path, low_memory=False)

    _save_barplot(df["Sentiment"], "Sentiment distribution", figures_dir / "sentiment_distribution.png")
    _save_barplot(df["CategoryID"].astype(str), "Top CategoryID distribution", figures_dir / "category_distribution_top30.png")

    if "CountryCode" in df.columns:
        _save_barplot(df["CountryCode"].fillna("unknown"), "Top countries", figures_dir / "country_distribution_top30.png")

    plt.figure(figsize=(10, 5))
    if sns:
        sns.histplot(df["word_count"].fillna(0), bins=60, color="#27ae60")
    else:
        plt.hist(df["word_count"].fillna(0), bins=60, color="#27ae60")
    plt.title("Comment word count distribution")
    plt.xlabel("Word count")
    plt.tight_layout()
    plt.savefig(figures_dir / "comment_word_count_distribution.png", dpi=160)
    plt.close()

    if "published_hour" in df.columns:
        plt.figure(figsize=(10, 5))
        if sns:
            sns.countplot(x=df["published_hour"], color="#f2994a")
        else:
            hour_counts = df["published_hour"].value_counts().sort_index()
            plt.bar(hour_counts.index.astype(str), hour_counts.values, color="#f2994a")
        plt.title("Comments by published hour")
        plt.xlabel("Hour")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figures_dir / "published_hour_distribution.png", dpi=160)
        plt.close()

    freq = _word_frequency(df["clean_comment"], top_n=100)
    freq.to_csv(metrics_dir / "word_frequency.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(10, 12))
    if sns:
        sns.barplot(data=freq.head(40), x="count", y="word", color="#9b51e0")
    else:
        top_freq = freq.head(40)
        plt.barh(top_freq["word"], top_freq["count"], color="#9b51e0")
        plt.gca().invert_yaxis()
    plt.title("Top words in comments")
    plt.tight_layout()
    plt.savefig(figures_dir / "word_frequency_top40.png", dpi=160)
    plt.close()

    try:
        from wordcloud import WordCloud

        text = " ".join(df["clean_comment"].fillna("").astype(str).sample(
            n=min(len(df), 50000),
            random_state=config["project"]["random_state"],
        ))
        cloud = WordCloud(width=1400, height=800, background_color="white").generate(text)
        plt.figure(figsize=(14, 8))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(figures_dir / "wordcloud.png", dpi=160)
        plt.close()
    except ImportError:
        LOGGER.warning("wordcloud is not installed; skipping word cloud.")

    LOGGER.info("EDA outputs saved under %s and %s", figures_dir, metrics_dir)
