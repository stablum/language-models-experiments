"""Shared helpers for importable ClearML pipeline stage functions."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import click

from src.corpora.splits import partitioned_metric_names
from src.tracking.clearml import download_task_artifact


STAGED_TOKENIZER_MODEL_NAME = "input-tokenizer.model"


def stage_tokenizer_model(
    *,
    tokenizer_task_id: str | None,
    tokenizer_model: Path | None,
    staging_dir: Path,
) -> Path:
    validate_tokenizer_source(
        tokenizer_task_id=tokenizer_task_id,
        tokenizer_model=tokenizer_model,
    )

    if tokenizer_task_id is not None:
        return download_task_artifact(
            task_id=tokenizer_task_id,
            artifact_name="sentencepiece-model",
            destination_dir=staging_dir,
            filename=STAGED_TOKENIZER_MODEL_NAME,
        )

    if tokenizer_model is None:
        raise click.ClickException(
            "Language model training requires a tokenizer artifact source."
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    destination = staging_dir / STAGED_TOKENIZER_MODEL_NAME
    if tokenizer_model.resolve() != destination.resolve():
        shutil.copy2(tokenizer_model, destination)
    return destination


def validate_tokenizer_source(
    *,
    tokenizer_task_id: str | None,
    tokenizer_model: Path | None,
) -> None:
    if tokenizer_task_id is not None and tokenizer_model is not None:
        raise click.ClickException(
            "Pass either --tokenizer-task-id or --tokenizer-model, not both."
        )

    if tokenizer_task_id is None and tokenizer_model is None:
        raise click.ClickException(
            "Language model training now resolves tokenizer artifacts from the tokenizer pipeline. "
            "Set --tokenizer-model-name on the training pipeline."
        )


def stage_model_artifacts(
    *,
    model_task_id: str | None,
    model_path: Path | None,
    staging_dir: Path,
) -> Path:
    validate_model_source(model_task_id=model_task_id, model_path=model_path)

    if model_task_id is not None:
        staged_model_path = download_task_artifact(
            task_id=model_task_id,
            artifact_name="trained-model-json",
            destination_dir=staging_dir,
        )
        download_task_artifact(
            task_id=model_task_id,
            artifact_name="input-tokenizer-model",
            destination_dir=staging_dir,
            filename=stored_tokenizer_filename(staged_model_path),
        )
        return staged_model_path

    if model_path is None:
        raise click.ClickException("Model artifact source is required.")
    return model_path


def validate_model_source(
    *,
    model_task_id: str | None,
    model_path: Path | None,
) -> None:
    if model_task_id is not None and model_path is not None:
        raise click.ClickException("Pass either --model-task-id or --model-path, not both.")

    if model_task_id is None and model_path is None:
        raise click.ClickException("Pass --model-task-id or --model-path.")


def training_summary_metrics(summary: object) -> dict[str, object]:
    return {
        "vocab_size": getattr(summary, "vocab_size", None),
        "sequence_count": getattr(summary, "sequence_count", None),
        "token_count": getattr(summary, "token_count", None),
        "transition_count": getattr(summary, "transition_count", None),
        "unigram_count": getattr(summary, "unigram_count", None),
        "bigram_transition_count": getattr(summary, "bigram_transition_count", None),
        "trigram_transition_count": getattr(summary, "trigram_transition_count", None),
        "continuation_unigram_count": getattr(summary, "continuation_unigram_count", None),
        "continuation_bigram_type_count": getattr(
            summary,
            "continuation_bigram_type_count",
            None,
        ),
        "smoothing": getattr(summary, "smoothing", None),
        "discount": getattr(summary, "discount", None),
        "unigram_weight": getattr(summary, "unigram_weight", None),
        "bigram_weight": getattr(summary, "bigram_weight", None),
        "trigram_weight": getattr(summary, "trigram_weight", None),
    }


def evaluation_metrics(summary: object) -> dict[str, object]:
    return {
        "sequence_count": getattr(summary, "sequence_count", None),
        "token_count": getattr(summary, "token_count", None),
        "transition_count": getattr(summary, "transition_count", None),
        "correct_next_token_count": getattr(summary, "correct_next_token_count", None),
        "top_k_correct_next_token_count": getattr(
            summary,
            "top_k_correct_next_token_count",
            None,
        ),
        "next_token_accuracy": getattr(summary, "next_token_accuracy", None),
        "top_k_accuracy": getattr(summary, "top_k_accuracy", None),
        "average_negative_log_likelihood": getattr(
            summary,
            "average_negative_log_likelihood",
            None,
        ),
        "cross_entropy_bits": getattr(summary, "cross_entropy_bits", None),
        "perplexity": getattr(summary, "perplexity", None),
        "zero_probability_count": getattr(summary, "zero_probability_count", None),
        "top_k": getattr(summary, "top_k", None),
        "discount": getattr(summary, "discount", None),
        "unigram_weight": getattr(summary, "unigram_weight", None),
        "bigram_weight": getattr(summary, "bigram_weight", None),
        "trigram_weight": getattr(summary, "trigram_weight", None),
    }


def evaluation_metrics_for_partition(
    summary: object,
    *,
    partition: str,
) -> dict[str, object]:
    return partitioned_metric_names(
        evaluation_metrics(summary),
        partition=partition,
    )


def evaluation_payload(summary: object) -> dict[str, object]:
    return {
        "model_artifact_file": artifact_file(getattr(summary, "model_path", None)),
        "tokenizer_artifact_file": artifact_file(getattr(summary, "tokenizer_model", None)),
        "text_normalization": getattr(summary, "text_normalization", None),
        **evaluation_metrics(summary),
    }


def query_metrics(result: object) -> dict[str, object]:
    next_predictions = getattr(result, "next_token_predictions", [])
    top_probability = next_predictions[0].probability if next_predictions else None
    return {
        "prompt_token_count": len(getattr(result, "prompt_token_ids", [])),
        "generated_token_count": len(getattr(result, "generated_token_ids", [])),
        "total_token_count": len(getattr(result, "token_ids", [])),
        "next_token_candidate_count": len(next_predictions),
        "top_next_token_probability": top_probability,
    }


def query_payload(result: object) -> dict[str, object]:
    return {
        "model_artifact_file": artifact_file(getattr(result, "model_path", None)),
        "tokenizer_artifact_file": artifact_file(getattr(result, "tokenizer_model", None)),
        "decoding": getattr(result, "decoding", None),
        "text_normalization": getattr(result, "text_normalization", None),
        "prompt": getattr(result, "prompt", None),
        "prompt_token_ids": getattr(result, "prompt_token_ids", None),
        "generated_token_ids": getattr(result, "generated_token_ids", None),
        "token_ids": getattr(result, "token_ids", None),
        "continuation_text": getattr(result, "continuation_text", None),
        "generated_text": getattr(result, "generated_text", None),
        "next_token_predictions": [
            {
                "token_id": prediction.token_id,
                "piece": prediction.piece,
                "count": prediction.count,
                "probability": prediction.probability,
            }
            for prediction in getattr(result, "next_token_predictions", [])
        ],
    }


def artifact_file(path: object) -> str | None:
    if path is None:
        return None
    return Path(path).name


def stored_tokenizer_filename(model_path: Path) -> str | None:
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None

    tokenizer_model = payload.get("tokenizer_model")
    if tokenizer_model is None:
        return None
    return Path(str(tokenizer_model)).name
