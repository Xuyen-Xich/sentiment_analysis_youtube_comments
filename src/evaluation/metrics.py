"""Model evaluation metrics and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


def classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str],
    y_score: np.ndarray | None = None,
    top_k: int = 3,
    average: str = "weighted",
) -> dict[str, Any]:
    """Return common classification metrics and optional top-k accuracy."""
    y_true_array = np.asarray(y_true).astype(str)
    y_pred_array = np.asarray(y_pred).astype(str)
    trained_labels = [str(label) for label in labels]
    extra_labels = sorted((set(y_true_array) | set(y_pred_array)) - set(trained_labels))
    report_labels = trained_labels + extra_labels
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(
                y_true_array,
                y_pred_array,
                labels=report_labels,
                average=average,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_array,
                y_pred_array,
                labels=report_labels,
                average=average,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=report_labels,
                average=average,
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=report_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true_array,
            y_pred_array,
            labels=report_labels,
            zero_division=0,
            output_dict=True,
        ),
    }
    if y_score is not None and len(trained_labels) > 2 and set(y_true_array) <= set(trained_labels):
        safe_k = min(top_k, len(trained_labels) - 1)
        metrics[f"top_{safe_k}_accuracy"] = float(
            top_k_accuracy_score(y_true_array, y_score, k=safe_k, labels=trained_labels)
        )
    return metrics


def save_confusion_matrix(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    """Save a confusion matrix image."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(7, min(20, 0.45 * len(labels)))
    height = max(6, min(20, 0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, height))
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
