"""Runtime helpers for importable ClearML language-model steps."""

from __future__ import annotations

from pathlib import Path

import click

from src.ml_core import tracking
from src.ml_core.cli import output as cli_out
from src.pipelines.language_model import definition as lm_def


def configure_step_clearml(clearml_config_file: str | None) -> None:
    if clearml_config_file is not None:
        tracking.configure_clearml_config_file(Path(clearml_config_file))


def emit_pipeline_stage_title(
    stage: str,
    *,
    index: int | None,
    total: int | None,
    title: str | None,
) -> None:
    if index is None or total is None or title is None:
        index, total, title = lm_def.standalone_stage_title(stage)
    cli_out.emit_stage_title(index, total, title)


def current_step_run(
    *,
    clearml_output_uri: str | None,
    clearml_tags: str | list[str] | tuple[str, ...] | None,
    stage: str,
) -> tracking.ClearMLRun:
    try:
        from clearml import OutputModel, Task
    except ImportError as error:
        raise click.ClickException(
            "ClearML pipeline steps require the clearml Python package. "
            "Run `uv sync` before using the pipeline CLI."
        ) from error

    task = Task.current_task()
    if task is None:
        raise click.ClickException(
            "This pipeline step must run inside a ClearML task created by PipelineController."
        )

    tags = tuple(dict.fromkeys(normalize_tags(clearml_tags)))
    if tags:
        task.add_tags(list(tags))
    return tracking.ClearMLRun(
        task=task,
        output_model_type=OutputModel,
        output_uri=clearml_output_uri,
        task_tags=tags,
    )


def require_task_id(clearml_run: tracking.ClearMLRun) -> str:
    task_id = clearml_run.task_id
    if task_id is None:
        raise click.ClickException("ClearML step task ID is not available.")
    return task_id


def normalize_tags(clearml_tags: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if clearml_tags is None:
        return ()
    if isinstance(clearml_tags, str):
        return tuple(tag for tag in clearml_tags.splitlines() if tag)
    return tuple(clearml_tags)
