"""Runtime helpers for importable ClearML language-model steps."""

from __future__ import annotations

from pathlib import Path

import click

from src.ml_core import cfg as core_cfg
from src.ml_core import tracking
from src.ml_core.cli import output as cli_out
from src.pipelines.language_model import definition as lm_def


ClearmlTags = str | list[str] | tuple[str, ...] | None


class StepRuntimeCfg(core_cfg.FrozenBaseCfg):
    """Cfg (configuration) shared by ClearML function-step entry points."""

    stage: str
    clearml_output_uri: str | None = None
    clearml_tags: ClearmlTags = None
    clearml_config_file: str | None = None
    pipeline_stage_index: int | None = None
    pipeline_stage_total: int | None = None
    pipeline_stage_title: str | None = None


def start_step(cfg: StepRuntimeCfg) -> tracking.ClearMLRun:
    """Configure ClearML, print the stage title, and return the current step run."""

    configure_step_clearml(cfg.clearml_config_file)
    emit_pipeline_stage_title(
        cfg.stage,
        index=cfg.pipeline_stage_index,
        total=cfg.pipeline_stage_total,
        title=cfg.pipeline_stage_title,
    )
    return current_step_run(
        clearml_output_uri=cfg.clearml_output_uri,
        clearml_tags=cfg.clearml_tags,
        stage=cfg.stage,
    )


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
    clearml_tags: ClearmlTags,
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


def normalize_tags(clearml_tags: ClearmlTags) -> tuple[str, ...]:
    if clearml_tags is None:
        return ()
    if isinstance(clearml_tags, str):
        return tuple(tag for tag in clearml_tags.splitlines() if tag)
    return tuple(clearml_tags)
