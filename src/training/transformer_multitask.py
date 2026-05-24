"""Transformer shared-encoder multi-task training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.evaluation.metrics import classification_metrics, save_confusion_matrix
from src.utils.io import save_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _import_torch_stack() -> tuple[Any, Any, Any, Any, Any]:
    """Import heavy transformer dependencies only when needed."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer, get_scheduler
    except ImportError as error:
        raise ImportError(
            "Transformer training requires torch and transformers. "
            "Install requirements.txt or run train_baseline only."
        ) from error
    return torch, nn, Dataset, DataLoader, (AutoModel, AutoTokenizer, get_scheduler)


def _build_classes(nn: Any, auto_model: Any) -> tuple[type, type]:
    """Create dataset and model classes after torch imports exist."""

    class CommentDataset(nn.Module):
        """Tokenized comments with both targets."""

        def __init__(
            self,
            texts: list[str],
            sentiment_labels: np.ndarray,
            category_labels: np.ndarray,
            tokenizer: Any,
            max_length: int,
        ) -> None:
            super().__init__()
            self.texts = texts
            self.sentiment_labels = sentiment_labels
            self.category_labels = category_labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> dict[str, Any]:
            encoded = self.tokenizer(
                self.texts[index],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "sentiment_label": self.sentiment_labels[index],
                "category_label": self.category_labels[index],
            }

    class MultiTaskTransformer(nn.Module):
        """Shared encoder with sentiment and category classification heads."""

        def __init__(self, model_name: str, sentiment_classes: int, category_classes: int) -> None:
            super().__init__()
            self.encoder = auto_model.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.2)
            self.sentiment_head = nn.Linear(hidden_size, sentiment_classes)
            self.category_head = nn.Linear(hidden_size, category_classes)

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            return self.sentiment_head(pooled), self.category_head(pooled)

    return CommentDataset, MultiTaskTransformer


def _sample_frame(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows and len(df) > max_rows:
        return df.sample(max_rows, random_state=random_state).reset_index(drop=True)
    return df.reset_index(drop=True)


def train_transformer(config: dict[str, Any]) -> dict[str, Any]:
    """Train a multi-task transformer model with a shared encoder."""
    torch, nn, _, DataLoader, transformer_stack = _import_torch_stack()
    auto_model, auto_tokenizer, get_scheduler = transformer_stack
    CommentDataset, MultiTaskTransformer = _build_classes(nn, auto_model)

    processed_dir = Path(config["paths"]["processed_dir"])
    train_path = processed_dir / "train.csv"
    validation_path = processed_dir / "validation.csv"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run preprocess before train_transformer.")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(validation_path, low_memory=False)
    transformer_cfg = config["transformer"]
    random_state = config["project"]["random_state"]
    train_df = _sample_frame(train_df, transformer_cfg.get("max_train_samples"), random_state)
    val_df = _sample_frame(val_df, transformer_cfg.get("max_eval_samples"), random_state)

    sentiment_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    train_sentiment = sentiment_encoder.fit_transform(train_df["Sentiment"].astype(str))
    train_category = category_encoder.fit_transform(train_df["CategoryID"].astype(str))
    val_sentiment = sentiment_encoder.transform(val_df["Sentiment"].astype(str))
    val_category = category_encoder.transform(val_df["CategoryID"].astype(str))

    tokenizer = auto_tokenizer.from_pretrained(transformer_cfg["model_name"])
    train_dataset = CommentDataset(
        train_df["model_text"].fillna("").astype(str).tolist(),
        train_sentiment,
        train_category,
        tokenizer,
        transformer_cfg["max_length"],
    )
    val_dataset = CommentDataset(
        val_df["model_text"].fillna("").astype(str).tolist(),
        val_sentiment,
        val_category,
        tokenizer,
        transformer_cfg["max_length"],
    )
    train_loader = DataLoader(train_dataset, batch_size=transformer_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=transformer_cfg["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTransformer(
        transformer_cfg["model_name"],
        len(sentiment_encoder.classes_),
        len(category_encoder.classes_),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=transformer_cfg["learning_rate"],
        weight_decay=transformer_cfg["weight_decay"],
    )
    total_steps = len(train_loader) * transformer_cfg["epochs"]
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(transformer_cfg["epochs"]):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_label"].long().to(device)
            category_labels = batch["category_label"].long().to(device)

            sentiment_logits, category_logits = model(input_ids, attention_mask)
            loss = criterion(sentiment_logits, sentiment_labels) + criterion(
                category_logits,
                category_labels,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))

        LOGGER.info("Transformer epoch %s loss %.4f", epoch + 1, float(np.mean(losses)))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})

    model.eval()
    sentiment_preds: list[int] = []
    category_preds: list[int] = []
    sentiment_scores: list[np.ndarray] = []
    category_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            sentiment_logits, category_logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            sentiment_prob = torch.softmax(sentiment_logits, dim=1).cpu().numpy()
            category_prob = torch.softmax(category_logits, dim=1).cpu().numpy()
            sentiment_scores.append(sentiment_prob)
            category_scores.append(category_prob)
            sentiment_preds.extend(sentiment_prob.argmax(axis=1).tolist())
            category_preds.extend(category_prob.argmax(axis=1).tolist())

    sentiment_labels = list(sentiment_encoder.classes_)
    category_labels = list(category_encoder.classes_)
    sentiment_pred_labels = sentiment_encoder.inverse_transform(sentiment_preds)
    category_pred_labels = category_encoder.inverse_transform(category_preds)
    sentiment_score_array = np.vstack(sentiment_scores)
    category_score_array = np.vstack(category_scores)

    metrics = {
        "sentiment": classification_metrics(
            val_df["Sentiment"].astype(str).to_numpy(),
            sentiment_pred_labels,
            sentiment_labels,
            y_score=sentiment_score_array,
            top_k=transformer_cfg["top_k"],
            average="weighted",
        ),
        "category": classification_metrics(
            val_df["CategoryID"].astype(str).to_numpy(),
            category_pred_labels,
            category_labels,
            y_score=category_score_array,
            top_k=transformer_cfg["top_k"],
            average="macro",
        ),
    }

    figures_dir = Path(config["paths"]["figures_dir"])
    save_confusion_matrix(
        val_df["Sentiment"].astype(str).to_numpy(),
        sentiment_pred_labels,
        sentiment_labels,
        "Transformer sentiment confusion matrix",
        figures_dir / "transformer_sentiment_confusion_matrix.png",
    )
    save_confusion_matrix(
        val_df["CategoryID"].astype(str).to_numpy(),
        category_pred_labels,
        category_labels,
        "Transformer category confusion matrix",
        figures_dir / "transformer_category_confusion_matrix.png",
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(config["paths"]["metrics_dir"]) / "transformer_training_history.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o")
    plt.title("Transformer training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(figures_dir / "transformer_training_curve.png", dpi=160)
    plt.close()

    model_dir = Path(config["paths"]["models_dir"]) / "transformer_multitask"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
    joblib.dump(
        {
            "sentiment_encoder": sentiment_encoder,
            "category_encoder": category_encoder,
            "model_name": transformer_cfg["model_name"],
            "max_length": transformer_cfg["max_length"],
        },
        model_dir / "metadata.joblib",
    )
    save_json(metrics, Path(config["paths"]["metrics_dir"]) / "transformer_validation_metrics.json")
    LOGGER.info("Transformer multi-task model saved to %s", model_dir)
    return metrics
