"""Definition for the tokenizer-training ClearML pipeline."""

from __future__ import annotations

from pathlib import Path

from src.ml_core.data.splits import VALIDATION_PARTITION
from src.pipelines.language_model.definition import (
    DEFAULT_TOKENIZER_TRAINING_NAME,
    TOKENIZER_STAGE,
    TOKENIZER_TRAINING_STAGE_DEPENDENCIES,
    TOKENIZER_TRAINING_STAGES,
    PipelineDefinition,
    output_uri_value,
    stage_gate_callback,
)
from src.pipelines.language_model.stage_entries import train_tokenizer_stage_entry
from src.pipelines.language_model.stages import (
    pipeline_artifact_monitors,
    pipeline_metric_monitors,
)


TOKENIZER_TRAINING_PIPELINE = PipelineDefinition(
    default_name=DEFAULT_TOKENIZER_TRAINING_NAME,
    stages=TOKENIZER_TRAINING_STAGES,
    stage_dependencies=TOKENIZER_TRAINING_STAGE_DEPENDENCIES,
)


def add_pipeline_steps(
    pipeline: object,
    *,
    clearml_project: str,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    clearml_config_file: Path | None,
    execution_queue: str | None,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    vocab_size: int,
    artifact_name: str,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
    text_normalization: str,
) -> None:
    artifact_monitors = pipeline_artifact_monitors()
    metric_monitors = pipeline_metric_monitors(VALIDATION_PARTITION)
    common_step_kwargs = {
        "clearml_output_uri": clearml_output_uri,
        "clearml_tags": "\n".join(clearml_tags),
        "clearml_config_file": str(clearml_config_file) if clearml_config_file else None,
    }
    step_options = {
        "project_name": clearml_project,
        "execution_queue": execution_queue,
        "output_uri": output_uri_value(clearml_output_uri),
        "auto_connect_frameworks": False,
        "auto_connect_arg_parser": False,
        "pre_execute_callback": stage_gate_callback,
        "tags": list(clearml_tags) if clearml_tags else None,
    }

    pipeline.add_function_step(
        name=TOKENIZER_STAGE,
        function=train_tokenizer_stage_entry,
        function_kwargs={
            "corpus": corpus,
            "dataset_id": dataset_id,
            "source_split": source_split,
            "text_column": text_column,
            "streaming": streaming,
            "limit": limit,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "vocab_size": vocab_size,
            "artifact_name": artifact_name,
            "model_type": model_type,
            "character_coverage": character_coverage,
            "hard_vocab_limit": hard_vocab_limit,
            "max_sentence_length": max_sentence_length,
            "text_normalization": text_normalization,
            **common_step_kwargs,
        },
        task_name=TOKENIZER_STAGE,
        task_type="training",
        monitor_artifacts=artifact_monitors[TOKENIZER_STAGE],
        monitor_metrics=metric_monitors[TOKENIZER_STAGE],
        stage=TOKENIZER_STAGE,
        **step_options,
    )


add_tokenizer_training_step = add_pipeline_steps


__all__ = (
    "DEFAULT_TOKENIZER_TRAINING_NAME",
    "TOKENIZER_STAGE",
    "TOKENIZER_TRAINING_PIPELINE",
    "TOKENIZER_TRAINING_STAGE_DEPENDENCIES",
    "TOKENIZER_TRAINING_STAGES",
    "add_pipeline_steps",
    "add_tokenizer_training_step",
)
