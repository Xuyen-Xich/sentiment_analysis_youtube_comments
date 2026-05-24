"""Command-line entry point for the NLP project."""

from __future__ import annotations

import argparse

from src.config.settings import ensure_output_dirs, load_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="YouTube comments NLP pipeline")
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "validate",
            "preprocess",
            "eda",
            "train_baseline",
            "train_transformer",
            "evaluate",
            "inference",
            "all",
        ],
        help="Pipeline step to run.",
    )
    parser.add_argument("--config", default="src/config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--input", default=None, help="Optional CSV for inference.")
    parser.add_argument("--output", default=None, help="Optional output CSV for inference.")
    parser.add_argument(
        "--model",
        default="baseline",
        choices=["baseline", "transformer"],
        help="Model type for inference.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one pipeline step."""
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)

    try:
        if args.step == "validate":
            from src.preprocessing.validation import validate_data

            validate_data(config)
        elif args.step == "preprocess":
            from src.preprocessing.pipeline import preprocess_data

            preprocess_data(config)
        elif args.step == "eda":
            from src.feature_engineering.eda import run_eda

            run_eda(config)
        elif args.step == "train_baseline":
            from src.training.baseline import train_baseline

            train_baseline(config)
        elif args.step == "train_transformer":
            from src.training.transformer_multitask import train_transformer

            train_transformer(config)
        elif args.step == "evaluate":
            from src.evaluation.evaluate import evaluate_all

            evaluate_all(config)
        elif args.step == "inference":
            from src.inference.predict import predict_baseline, predict_transformer

            if args.model == "baseline":
                predict_baseline(config, input_path=args.input, output_path=args.output)
            else:
                predict_transformer(config, input_path=args.input, output_path=args.output)
        elif args.step == "all":
            from pathlib import Path

            from src.evaluation.evaluate import evaluate_all
            from src.feature_engineering.eda import run_eda
            from src.inference.predict import predict_baseline
            from src.preprocessing.pipeline import preprocess_data
            from src.preprocessing.validation import validate_data
            from src.training.baseline import train_baseline
            from src.training.transformer_multitask import train_transformer

            validate_data(config)
            preprocess_data(config)
            run_eda(config)
            train_baseline(config)
            try:
                train_transformer(config)
            except ImportError as error:
                LOGGER.warning("Skipping transformer step: %s", error)
            evaluate_all(config)
            predict_baseline(config, output_path=Path(config["paths"]["metrics_dir"]) / "inference_predictions.csv")
    except Exception:
        LOGGER.exception("Pipeline step failed: %s", args.step)
        raise


if __name__ == "__main__":
    main()
