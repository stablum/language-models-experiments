"""Shared model staging and payload helpers for ClearML pipeline stages."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import click

from src.ml_core.data.splits import partitioned_metric_names
from src.ml_core.tracking import (
    clearml_task_parameter,
    download_task_output_model,
    maybe_download_task_input_model,
)


STAGED_TOKENIZER_MODEL_NAME = "input-tokenizer.model"
TRAINING_METRIC_ATTRS = (
    "vocab_size",
    "sequence_count",
    "token_count",
    "transition_count",
    "unigram_count",
    "bigram_transition_count",
    "trigram_transition_count",
    "continuation_unigram_count",
    "continuation_bigram_type_count",
    "smoothing",
    "discount",
    "unigram_weight",
    "bigram_weight",
    "trigram_weight",
)
EVALUATION_METRIC_ATTRS = (
    "sequence_count",
    "token_count",
    "transition_count",
    "correct_next_token_count",
    "top_k_correct_next_token_count",
    "next_token_accuracy",
    "top_k_accuracy",
    "average_negative_log_likelihood",
    "cross_entropy_bits",
    "perplexity",
    "zero_probability_count",
    "top_k",
    "discount",
    "unigram_weight",
    "bigram_weight",
    "trigram_weight",
)


def stage_tokenizer_model(
    *,
    tokenizer_task_id: str | None,
    tokenizer_model_name: str | None = None,
    tokenizer_model: Path | None = None,
    staging_dir: Path,
    clearml_run: object | None = None,
) -> Path:
    validate_tokenizer_source(
        tokenizer_task_id=tokenizer_task_id,
        tokenizer_model=tokenizer_model,
    )

    if tokenizer_task_id is not None:
        return download_task_output_model(
            task_id=tokenizer_task_id,
            destination_dir=staging_dir,
            filename=STAGED_TOKENIZER_MODEL_NAME,
            model_name=tokenizer_model_name,
            connect_to_task=getattr(clearml_run, "task", None),
        )

    if tokenizer_model is None:
        raise click.ClickException(
            "Language model training requires a tokenizer model source."
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
            "Language model training now resolves tokenizer models from tokenizer-training runs. "
            "Set --tokenizer-model-name on model training."
        )


def stage_model_files(
    *,
    model_task_id: str | None,
    model_path: Path | None,
    staging_dir: Path,
    clearml_run: object | None = None,
    output_model_name: str | None = None,
) -> Path:
    validate_model_source(model_task_id=model_task_id, model_path=model_path)

    if model_task_id is not None:
        staged_model_path = download_task_output_model(
            task_id=model_task_id,
            destination_dir=staging_dir,
            filename=stored_model_filename(output_model_name),
            model_name=output_model_name,
            connect_to_task=getattr(clearml_run, "task", None),
        )
        tokenizer_filename = stored_tokenizer_filename(staged_model_path)
        staged_tokenizer_path = maybe_download_task_input_model(
            task_id=model_task_id,
            destination_dir=staging_dir,
            filename=tokenizer_filename,
            connect_to_task=getattr(clearml_run, "task", None),
        )
        if staged_tokenizer_path is None:
            tokenizer_task_id = clearml_task_parameter(
                model_task_id,
                "Pipeline/tokenizer_task_id",
            )
            if tokenizer_task_id is None:
                raise click.ClickException(
                    f"ClearML model task {model_task_id} has no linked input tokenizer model "
                    "and no Pipeline/tokenizer_task_id parameter."
                )
            download_task_output_model(
                task_id=tokenizer_task_id,
                destination_dir=staging_dir,
                filename=tokenizer_filename,
                connect_to_task=getattr(clearml_run, "task", None),
            )
        return staged_model_path

    if model_path is None:
        raise click.ClickException("Model source is required.")
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
    return _attrs(summary, TRAINING_METRIC_ATTRS)


def evaluation_metrics(summary: object) -> dict[str, object]:
    return _attrs(summary, EVALUATION_METRIC_ATTRS)


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
        "model_file": artifact_file(getattr(summary, "model_path", None)),
        "tokenizer_model_file": artifact_file(getattr(summary, "tokenizer_model", None)),
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
        "model_file": artifact_file(getattr(result, "model_path", None)),
        "tokenizer_model_file": artifact_file(getattr(result, "tokenizer_model", None)),
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


def query_debug_sample(result: object) -> str:
    prompt = getattr(result, "prompt", "") or "(empty)"
    continuation = getattr(result, "continuation_text", "") or "(empty)"
    generated_text = getattr(result, "generated_text", "") or "(empty)"
    return (
        "Prompt:\n"
        f"{prompt}\n\n"
        "Continuation:\n"
        f"{continuation}\n\n"
        "Generated text:\n"
        f"{generated_text}\n"
    )


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


def stored_model_filename(output_model_name: str | None) -> str | None:
    if output_model_name is None:
        return None
    output_path = Path(output_model_name)
    return output_path.name if output_path.suffix else f"{output_path.name}.json"


def _attrs(obj: object, names: tuple[str, ...]) -> dict[str, object]:
    return {name: getattr(obj, name, None) for name in names}
