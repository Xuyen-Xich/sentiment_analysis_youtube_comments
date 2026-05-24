"""TF-IDF baseline models for separate sentiment and category tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional plotting dependency
    sns = None
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.evaluation.metrics import classification_metrics, save_confusion_matrix
from src.utils.io import save_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _load_splits(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(config["paths"]["processed_dir"])
    train_path = processed_dir / "train.csv"
    validation_path = processed_dir / "validation.csv"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run preprocess before train_baseline.")
    return (
        pd.read_csv(train_path, low_memory=False),
        pd.read_csv(validation_path, low_memory=False),
    )


def _make_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    baseline_cfg = config["baseline"]
    text_feature = config["features"]["text_feature"]
    numeric_features = config["features"]["numeric_features"]
    categorical_features = config["features"]["categorical_features"]

    text_pipeline = TfidfVectorizer(
        max_features=baseline_cfg["max_features"],
        ngram_range=tuple(baseline_cfg["ngram_range"]),
        min_df=baseline_cfg["min_df"],
        max_df=baseline_cfg["max_df"],
        sublinear_tf=True,
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, text_feature),
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _make_model(config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", _make_preprocessor(config)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config["baseline"]["max_iter"],
                    class_weight=config["baseline"]["class_weight"],
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _ensure_feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for column in config["features"]["numeric_features"]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    for column in config["features"]["categorical_features"]:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = df[column].fillna("unknown").astype(str)
    df[config["features"]["text_feature"]] = df[config["features"]["text_feature"]].fillna("")
    return df


def _save_text_feature_importance(
    pipeline: Pipeline,
    labels: list[str],
    task_name: str,
    config: dict[str, Any],
    top_n: int = 25,
) -> None:
    """Save coefficient-based TF-IDF feature importance."""
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    vectorizer = pipeline.named_steps["features"].named_transformers_["text"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    rows: list[dict[str, Any]] = []

    for class_index, label in enumerate(labels):
        coefficients = classifier.coef_[class_index][: len(feature_names)]
        top_indices = coefficients.argsort()[-top_n:][::-1]
        for rank, index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "task": task_name,
                    "label": label,
                    "rank": rank,
                    "feature": feature_names[index],
                    "coefficient": float(coefficients[index]),
                }
            )

    importance = pd.DataFrame(rows)
    csv_path = metrics_dir / f"{task_name}_tfidf_feature_importance.csv"
    importance.to_csv(csv_path, index=False, encoding="utf-8")

    for label in labels[: min(len(labels), 8)]:
        label_df = importance[importance["label"] == label].head(15)
        plt.figure(figsize=(9, 5))
        if sns:
            sns.barplot(data=label_df, x="coefficient", y="feature", color="#2d9cdb")
        else:
            plt.barh(label_df["feature"], label_df["coefficient"], color="#2d9cdb")
            plt.gca().invert_yaxis()
        plt.title(f"{task_name} important words: {label}")
        plt.tight_layout()
        safe_label = str(label).replace("/", "_").replace("\\", "_")
        plt.savefig(figures_dir / f"{task_name}_feature_importance_{safe_label}.png", dpi=160)
        plt.close()


def train_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Train separate baseline models for sentiment and category."""
    train_df, val_df = _load_splits(config)
    train_df = _ensure_feature_columns(train_df, config)
    val_df = _ensure_feature_columns(val_df, config)

    artifacts: dict[str, Any] = {"models": {}, "label_encoders": {}}
    metrics: dict[str, Any] = {}

    for task_name, target_column in {
        "sentiment": "Sentiment",
        "category": "CategoryID",
    }.items():
        LOGGER.info("Training baseline %s model", task_name)
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_df[target_column].astype(str))
        labels = list(label_encoder.classes_)
        model = _make_model(config)
        model.fit(train_df, y_train)

        y_pred_encoded = model.predict(val_df)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_true = val_df[target_column].astype(str).to_numpy()
        y_score = model.predict_proba(val_df)

        task_metrics = classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            y_score=y_score,
            top_k=config["baseline"]["top_k"],
            average="weighted" if task_name == "sentiment" else "macro",
        )
        metrics[task_name] = task_metrics
        save_confusion_matrix(
            y_true,
            y_pred,
            labels,
            f"Baseline {task_name} confusion matrix",
            Path(config["paths"]["figures_dir"]) / f"baseline_{task_name}_confusion_matrix.png",
        )
        _save_text_feature_importance(model, labels, task_name, config)
        artifacts["models"][task_name] = model
        artifacts["label_encoders"][task_name] = label_encoder

    model_path = Path(config["paths"]["models_dir"]) / "baseline_models.joblib"
    joblib.dump(artifacts, model_path)
    save_json(metrics, Path(config["paths"]["metrics_dir"]) / "baseline_validation_metrics.json")
    LOGGER.info("Baseline artifacts saved to %s", model_path)
    return metrics
