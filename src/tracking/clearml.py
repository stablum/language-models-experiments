"""Optional ClearML experiment tracking integration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click


DEFAULT_CLEARML_PROJECT = "language-models-experiments"


@dataclass(frozen=True)
class ClearMLSettings:
    enabled: bool
    project_name: str = DEFAULT_CLEARML_PROJECT
    task_name: str | None = None
    output_uri: str | None = None
    tags: tuple[str, ...] = ()


def clearml_options(command: Any) -> Any:
    command = click.option(
        "--clearml-tag",
        "clearml_tags",
        multiple=True,
        help="Tag to attach to the ClearML task. Can be passed multiple times.",
    )(command)
    command = click.option(
        "--clearml-output-uri",
        default=None,
        envvar="CLEARML_OUTPUT_URI",
        help="Optional ClearML output URI for uploaded models and artifacts.",
    )(command)
    command = click.option(
        "--clearml-task-name",
        default=None,
        help="Override the ClearML task name.",
    )(command)
    command = click.option(
        "--clearml-project",
        default=DEFAULT_CLEARML_PROJECT,
        envvar="CLEARML_PROJECT",
        show_default=True,
        help="ClearML project name.",
    )(command)
    command = click.option(
        "--clearml/--no-clearml",
        default=False,
        envvar="LME_CLEARML",
        show_default=True,
        help="Register this CLI run in ClearML.",
    )(command)
    return command


def clearml_settings(
    *,
    enabled: bool,
    project_name: str,
    task_name: str | None,
    output_uri: str | None,
    tags: tuple[str, ...],
) -> ClearMLSettings:
    return ClearMLSettings(
        enabled=enabled,
        project_name=project_name,
        task_name=task_name,
        output_uri=output_uri,
        tags=tuple(tags),
    )


class ClearMLRun:
    def __init__(
        self,
        *,
        task: Any | None = None,
        output_model_type: Any | None = None,
        output_uri: str | None = None,
    ) -> None:
        self.task = task
        self.output_model_type = output_model_type
        self.output_uri = output_uri

    @property
    def enabled(self) -> bool:
        return self.task is not None

    def __enter__(self) -> ClearMLRun:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def connect_parameters(self, parameters: Mapping[str, Any], *, name: str = "CLI") -> None:
        if self.task is None:
            return
        self.task.connect(sanitize_value(parameters), name=name)

    def log_metrics(
        self,
        title: str,
        metrics: Mapping[str, Any],
        *,
        iteration: int = 0,
    ) -> None:
        if self.task is None:
            return

        logger = self.task.get_logger()
        for series, value in metrics.items():
            numeric_value = as_finite_float(value)
            if numeric_value is None:
                continue

            logger.report_scalar(
                title=title,
                series=series,
                value=numeric_value,
                iteration=iteration,
            )
            logger.report_single_value(
                name=f"{title}/{series}",
                value=numeric_value,
            )

    def upload_artifact(
        self,
        name: str,
        artifact: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.task is None:
            return

        self.task.upload_artifact(
            name=name,
            artifact_object=sanitize_value(artifact),
            metadata=sanitize_value(metadata) if metadata is not None else None,
            wait_on_upload=True,
        )

    def register_model(
        self,
        *,
        name: str,
        model_path: Path,
        framework: str = "custom",
        tags: tuple[str, ...] = (),
        comment: str | None = None,
    ) -> None:
        if self.task is None or self.output_model_type is None:
            return
        if not model_path.exists():
            return

        output_model = self.output_model_type(
            task=self.task,
            name=name,
            tags=list(tags),
            comment=comment,
            framework=framework,
        )
        output_model.update_weights(
            weights_filename=str(model_path),
            upload_uri=self.output_uri,
            auto_delete_file=False,
            async_enable=False,
        )

    def close(self) -> None:
        if self.task is None:
            return
        if self.output_model_type is not None:
            self.output_model_type.wait_for_uploads()
        self.task.close()


def start_clearml_run(
    settings: ClearMLSettings,
    *,
    default_task_name: str,
    task_type: str,
) -> ClearMLRun:
    if not settings.enabled:
        return ClearMLRun()

    try:
        from clearml import OutputModel, Task
    except ImportError as error:
        raise click.ClickException(
            "ClearML tracking requires the clearml Python package. "
            "Run `uv sync` before using --clearml."
        ) from error

    output_uri: str | bool = settings.output_uri if settings.output_uri is not None else True
    task_type_value = getattr(Task.TaskTypes, task_type, task_type)
    task = Task.init(
        project_name=settings.project_name,
        task_name=settings.task_name or default_task_name,
        task_type=task_type_value,
        tags=list(settings.tags),
        output_uri=output_uri,
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
    )
    return ClearMLRun(
        task=task,
        output_model_type=OutputModel,
        output_uri=settings.output_uri,
    )


def sanitize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value]
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    return None
