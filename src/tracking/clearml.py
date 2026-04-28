"""Optional ClearML experiment tracking integration."""

from __future__ import annotations

import math
import os
import re
import shutil
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click


DEFAULT_CLEARML_PROJECT = "language-models-experiments"
CLEARML_CONNECT_TIMEOUT_SECONDS = 2.0
CLEARML_CONFIG_ENDPOINTS = ("api_server", "files_server")


@dataclass(frozen=True)
class ClearMLSettings:
    project_name: str = DEFAULT_CLEARML_PROJECT
    task_name: str | None = None
    config_file: Path | None = None
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
        "--clearml-config-file",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        envvar="CLEARML_CONFIG_FILE",
        help="ClearML SDK config file.",
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
    return command


def clearml_settings(
    *,
    project_name: str,
    task_name: str | None,
    config_file: Path | None,
    output_uri: str | None,
    tags: tuple[str, ...],
) -> ClearMLSettings:
    return ClearMLSettings(
        project_name=project_name,
        task_name=task_name,
        config_file=config_file,
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
        task_tags: tuple[str, ...] = (),
    ) -> None:
        self.task = task
        self.output_model_type = output_model_type
        self.output_uri = output_uri
        self.task_tags = task_tags

    @property
    def enabled(self) -> bool:
        return self.task is not None

    @property
    def task_id(self) -> str | None:
        if self.task is None:
            return None
        return str(self.task.id)

    @property
    def task_url(self) -> str | None:
        if self.task is None:
            return None
        url_getter = getattr(self.task, "get_output_log_web_page", None)
        if not callable(url_getter):
            return None
        url = url_getter()
        return str(url) if url else None

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

        model_tags = tuple(dict.fromkeys((*tags, *self.task_tags)))
        output_model = self.output_model_type(
            task=self.task,
            name=name,
            tags=list(model_tags),
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
    resolved_config_file = configure_clearml_config_file(settings.config_file)
    assert_clearml_endpoints_reachable(resolved_config_file, settings.output_uri)

    try:
        from clearml import OutputModel, Task
    except ImportError as error:
        raise click.ClickException(
            "ClearML tracking requires the clearml Python package. "
            "Run `uv sync` before using the experiment CLIs."
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
        auto_connect_streams=False,
    )
    return ClearMLRun(
        task=task,
        output_model_type=OutputModel,
        output_uri=settings.output_uri,
        task_tags=settings.tags,
    )


def configure_clearml_config_file(config_file: Path | None) -> Path | None:
    if config_file is None:
        return None

    resolved_config_file = config_file.expanduser()
    if not resolved_config_file.is_absolute():
        resolved_config_file = Path.cwd() / resolved_config_file
    if not resolved_config_file.exists():
        raise click.ClickException(
            f"ClearML config file does not exist: {resolved_config_file}. "
            "Create it with `Copy-Item clearml.local.conf.example clearml.conf` "
            "or pass --clearml-config-file."
        )

    os.environ["CLEARML_CONFIG_FILE"] = str(resolved_config_file)
    return resolved_config_file


def assert_clearml_endpoints_reachable(
    config_file: Path | None,
    output_uri: str | None,
) -> None:
    endpoints = clearml_endpoints(config_file, output_uri)
    failures = [
        f"{label} {url}: {error}"
        for label, url in endpoints
        if (error := endpoint_connection_error(url)) is not None
    ]
    if not failures:
        return

    checked = ", ".join(f"{label} {url}" for label, url in endpoints)
    details = "; ".join(failures)
    raise click.ClickException(
        "ClearML server is not reachable before task initialization. "
        f"Checked: {checked}. "
        "Start the repo-local server with `docker compose -f docker-compose.clearml.yml up -d`, "
        "or pass a working --clearml-config-file / --clearml-output-uri. "
        f"Details: {details}"
    )


def clearml_endpoints(
    config_file: Path | None,
    output_uri: str | None,
) -> list[tuple[str, str]]:
    endpoints = clearml_config_endpoints(config_file)
    if output_uri is not None:
        endpoints.append(("output_uri", output_uri))

    unique_endpoints: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for label, url in endpoints:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        unique_endpoints.append((label, url))
    return unique_endpoints


def clearml_config_endpoints(config_file: Path | None) -> list[tuple[str, str]]:
    if config_file is None:
        return []

    endpoint_pattern = re.compile(r"^\s*(api_server|files_server)\s*:\s*\"([^\"]+)\"")
    endpoints: list[tuple[str, str]] = []
    for line in config_file.read_text(encoding="utf-8").splitlines():
        match = endpoint_pattern.match(line)
        if match is None:
            continue
        label, url = match.groups()
        if label in CLEARML_CONFIG_ENDPOINTS:
            endpoints.append((label, url))
    return endpoints


def endpoint_connection_error(url: str) -> str | None:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        return None

    hostname = parsed_url.hostname
    if hostname is None:
        return "missing hostname"

    port = parsed_url.port
    if port is None:
        port = 443 if parsed_url.scheme == "https" else 80

    try:
        with socket.create_connection(
            (hostname, port),
            timeout=CLEARML_CONNECT_TIMEOUT_SECONDS,
        ):
            return None
    except OSError as error:
        return str(error)


def download_task_artifact(
    *,
    task_id: str,
    artifact_name: str,
    destination_dir: Path,
    filename: str | None = None,
) -> Path:
    try:
        from clearml import Task
    except ImportError as error:
        raise click.ClickException(
            "ClearML artifact download requires the clearml Python package. "
            "Run `uv sync` before using this command."
        ) from error

    task = Task.get_task(task_id=task_id)
    artifact = task.artifacts.get(artifact_name)
    if artifact is None:
        available = ", ".join(sorted(task.artifacts)) or "none"
        raise click.ClickException(
            f"ClearML task {task_id} has no artifact named {artifact_name!r}. "
            f"Available artifacts: {available}."
        )

    local_copy = artifact.get_local_copy()
    if local_copy is None:
        raise click.ClickException(
            f"Could not download ClearML artifact {artifact_name!r} from task {task_id}."
        )

    source = Path(local_copy)
    if not source.exists():
        raise click.ClickException(
            f"Downloaded ClearML artifact path does not exist: {source}"
        )
    if source.is_dir():
        raise click.ClickException(
            f"ClearML artifact {artifact_name!r} from task {task_id} is a directory; "
            "this CLI expects a single file artifact."
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (filename or source.name)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


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
