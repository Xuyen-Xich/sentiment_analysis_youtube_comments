"""Evaluate saved models on the held-out test split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.evaluation.metrics import classification_metrics, save_confusion_matrix
from src.training.baseline import _ensure_feature_columns
from src.utils.io import save_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def evaluate_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate baseline models on the test set."""
    model_path = Path(config["paths"]["models_dir"]) / "baseline_models.joblib"
    test_path = Path(config["paths"]["processed_dir"]) / "test.csv"
    if not model_path.exists():
        raise FileNotFoundError("Run train_baseline before evaluate.")
    if not test_path.exists():
        raise FileNotFoundError("Run preprocess before evaluate.")

    artifacts = joblib.load(model_path)
    test_df = _ensure_feature_columns(pd.read_csv(test_path, low_memory=False), config)
    metrics: dict[str, Any] = {}

    for task_name, target_column in {"sentiment": "Sentiment", "category": "CategoryID"}.items():
        model = artifacts["models"][task_name]
        encoder = artifacts["label_encoders"][task_name]
        labels = list(encoder.classes_)
        y_true = test_df[target_column].astype(str).to_numpy()
        y_pred = encoder.inverse_transform(model.predict(test_df))
        y_score = model.predict_proba(test_df)
        metrics[task_name] = classification_metrics(
            y_true,
            y_pred,
            labels,
            y_score=y_score,
            top_k=config["baseline"]["top_k"],
            average="weighted" if task_name == "sentiment" else "macro",
        )
        save_confusion_matrix(
            y_true,
            y_pred,
            labels,
            f"Test baseline {task_name} confusion matrix",
            Path(config["paths"]["figures_dir"]) / f"test_baseline_{task_name}_confusion_matrix.png",
        )

        errors = test_df.assign(prediction=y_pred)
        errors = errors[errors[target_column].astype(str) != errors["prediction"].astype(str)]
        columns = ["CommentText", "VideoTitle", target_column, "prediction", "Sentiment", "CategoryID"]
        existing_columns = [column for column in columns if column in errors.columns]
        errors[existing_columns].head(100).to_csv(
            Path(config["paths"]["metrics_dir"]) / f"{task_name}_error_examples.csv",
            index=False,
            encoding="utf-8",
        )

    save_json(metrics, Path(config["paths"]["metrics_dir"]) / "baseline_test_metrics.json")
    LOGGER.info("Baseline test metrics saved.")
    return metrics


def evaluate_all(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate available trained models and create a comparison table."""
    results: dict[str, Any] = {}
    baseline_path = Path(config["paths"]["models_dir"]) / "baseline_models.joblib"
    if baseline_path.exists():
        results["baseline"] = evaluate_baseline(config)
    else:
        LOGGER.warning("Skipping baseline evaluation; model file not found.")

    transformer_metrics = Path(config["paths"]["metrics_dir"]) / "transformer_validation_metrics.json"
    if transformer_metrics.exists():
        LOGGER.info("Transformer validation metrics are available at %s", transformer_metrics)

    comparison_rows = []
    for model_name, model_metrics in results.items():
        for task_name, task_metrics in model_metrics.items():
            comparison_rows.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "accuracy": task_metrics.get("accuracy"),
                    "f1": task_metrics.get("f1"),
                    "macro_f1": task_metrics.get("macro_f1"),
                }
            )
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(
            Path(config["paths"]["metrics_dir"]) / "model_comparison.csv",
            index=False,
            encoding="utf-8",
        )
    return results
