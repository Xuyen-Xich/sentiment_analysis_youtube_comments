"""Inference utilities for new YouTube comments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing.text_cleaning import clean_text
from src.training.baseline import _ensure_feature_columns
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def prepare_inference_frame(records: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    """Convert raw inference records into model-ready features."""
    df = pd.DataFrame(records)
    if "CommentText" not in df.columns:
        raise ValueError("Inference data must include CommentText.")

    for column in ("VideoTitle", "CountryCode"):
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    for column in ("Likes", "Replies"):
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)

    if "PublishedAt" in df.columns:
        dates = pd.to_datetime(df["PublishedAt"], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=df.index)

    slang_map = config["preprocessing"].get("slang_map", {})
    max_repeats = config["preprocessing"].get("max_repeated_chars", 2)
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
    df["published_hour"] = dates.dt.hour.fillna(0).astype(int)
    df["published_dayofweek"] = dates.dt.dayofweek.fillna(0).astype(int)
    df["published_month"] = dates.dt.month.fillna(0).astype(int)
    return _ensure_feature_columns(df, config)


def predict_baseline(
    config: dict[str, Any],
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    records: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Predict sentiment and category with the saved baseline models."""
    model_path = Path(config["paths"]["models_dir"]) / "baseline_models.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run train_baseline before inference.")

    if records is None:
        if input_path is None:
            records = [
                {
                    "CommentText": "This video is very helpful, thank you!",
                    "VideoTitle": "Sample YouTube video",
                    "Likes": 0,
                    "Replies": 0,
                    "CountryCode": "US",
                }
            ]
        else:
            records = pd.read_csv(input_path, low_memory=False).to_dict("records")

    df = prepare_inference_frame(records, config)
    artifacts = joblib.load(model_path)

    for task_name in ("sentiment", "category"):
        model = artifacts["models"][task_name]
        encoder = artifacts["label_encoders"][task_name]
        prediction = encoder.inverse_transform(model.predict(df))
        df[f"predicted_{task_name}"] = prediction

        probabilities = model.predict_proba(df)
        classes = encoder.classes_
        top_indices = probabilities.argsort(axis=1)[:, ::-1][:, : min(3, len(classes))]
        df[f"{task_name}_top_predictions"] = [
            [
                {"label": str(classes[index]), "probability": float(probabilities[row_idx, index])}
                for index in row
            ]
            for row_idx, row in enumerate(top_indices)
        ]

    result_columns = [
        "CommentText",
        "VideoTitle",
        "predicted_sentiment",
        "sentiment_top_predictions",
        "predicted_category",
        "category_top_predictions",
    ]
    existing = [column for column in result_columns if column in df.columns]
    result = df[existing]

    output = Path(output_path) if output_path else Path(config["paths"]["metrics_dir"]) / "inference_predictions.csv"
    result.to_csv(output, index=False, encoding="utf-8")
    LOGGER.info("Inference predictions saved to %s", output)
    return result


def predict_transformer(
    config: dict[str, Any],
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    records: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Predict sentiment and category with the saved multi-task transformer."""
    from src.training.transformer_multitask import _build_classes, _import_torch_stack

    torch, nn, _, _, transformer_stack = _import_torch_stack()
    auto_model, auto_tokenizer, _ = transformer_stack

    model_dir = Path(config["paths"]["models_dir"]) / "transformer_multitask"
    metadata_path = model_dir / "metadata.joblib"
    weights_path = model_dir / "pytorch_model.bin"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError("Run train_transformer before transformer inference.")

    if records is None:
        if input_path is None:
            records = [{"CommentText": "This video is very helpful, thank you!"}]
        else:
            records = pd.read_csv(input_path, low_memory=False).to_dict("records")

    df = prepare_inference_frame(records, config)
    metadata = joblib.load(metadata_path)
    sentiment_encoder = metadata["sentiment_encoder"]
    category_encoder = metadata["category_encoder"]
    _, MultiTaskTransformer = _build_classes(nn, auto_model)

    tokenizer = auto_tokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTransformer(
        metadata["model_name"],
        len(sentiment_encoder.classes_),
        len(category_encoder.classes_),
    ).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    sentiment_predictions: list[str] = []
    category_predictions: list[str] = []
    batch_size = config["transformer"]["batch_size"]
    texts = df["model_text"].fillna("").astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=metadata["max_length"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            sentiment_logits, category_logits = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
            sentiment_indices = sentiment_logits.argmax(dim=1).cpu().numpy()
            category_indices = category_logits.argmax(dim=1).cpu().numpy()
            sentiment_predictions.extend(sentiment_encoder.inverse_transform(sentiment_indices))
            category_predictions.extend(category_encoder.inverse_transform(category_indices))

    df["predicted_sentiment"] = sentiment_predictions
    df["predicted_category"] = category_predictions
    result_columns = ["CommentText", "VideoTitle", "predicted_sentiment", "predicted_category"]
    existing = [column for column in result_columns if column in df.columns]
    result = df[existing]

    output = Path(output_path) if output_path else Path(config["paths"]["metrics_dir"]) / "transformer_inference_predictions.csv"
    result.to_csv(output, index=False, encoding="utf-8")
    LOGGER.info("Transformer inference predictions saved to %s", output)
    return result
