"""Shared helpers for stage-resume command-line interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import click

from src.ml_core import pipeline as core_pipeline
from src.ml_core.cli.config import load_defaults_from_sections
from src.pipelines.language_model import model_training as model_pipeline


def load_stage_command_defaults(stage_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = load_defaults_from_sections(("train",))
    for key in ("model_name", "tokenizer_model_name"):
        if key in train_defaults:
            defaults[key] = train_defaults[key]
    defaults.update(load_defaults_from_sections((stage_section,)))
    return defaults


def require_tokenizer_model_name(tokenizer_model_name: str | None, *, action: str) -> str:
    resolved = str(tokenizer_model_name or "").strip()
    if not resolved:
        raise click.ClickException(
            f"{action} requires --tokenizer-model-name, or tokenizer_model_name in [train]."
        )
    return resolved


def reject_pipeline_local(pipeline_local: bool) -> None:
    if pipeline_local:
        raise click.ClickException(
            "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
            "Use --pipeline-queued for stage CLIs."
        )


def reject_deprecated_model_dependency(
    model_task_id: str | None,
    model_path: Path | None,
    *,
    action: str,
) -> None:
    if model_task_id is not None or model_path is not None:
        raise click.ClickException(
            f"{action} now resumes the canonical ClearML pipeline DAG. "
            "Run train first in the same pipeline instead of passing --model-task-id or --model-path."
        )


def resume_model_training_stage(
    *,
    stage_name: str,
    pipeline_name: str,
    pipeline_version: str,
    controller_queue: str,
    wait: bool,
    pipeline_controller_id: str | None,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    parameter_filters: Mapping[str, object],
) -> None:
    core_pipeline.resume_pipeline_controller_stage(
        stage_name=stage_name,
        pipeline_controller_id=pipeline_controller_id,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        controller_queue=controller_queue,
        wait=wait,
        clearml_project=clearml_project,
        clearml_task_name=clearml_task_name,
        clearml_config_file=clearml_config_file,
        clearml_connectivity_check=clearml_connectivity_check,
        clearml_output_uri=clearml_output_uri,
        clearml_tags=clearml_tags,
        parameter_filters=parameter_filters,
        stage_dependencies=(
            model_pipeline.MODEL_TRAINING_PIPELINE.stage_dependencies
        ),
        stage_names=model_pipeline.MODEL_TRAINING_PIPELINE.stages,
    )
