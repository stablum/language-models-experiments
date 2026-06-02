"""ClearML task, model, and artifact helpers."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import click

from src.ml_core.cli import staging as cli_staging


CLEARML_ARTIFACT_LOCK_TIMEOUT_SECONDS = 1800.0
CLEARML_ARTIFACT_LOCK_POLL_SECONDS = 0.1
CLEARML_ARTIFACT_LOCK_DIR = cli_staging.STAGING_ROOT / ".clearml-artifact-locks"


def download_task_artifact(
    *,
    task_id: str,
    artifact_name: str,
    destination_dir: Path,
    filename: str | None = None,
) -> Path:
    artifact_path = maybe_download_task_artifact(
        task_id=task_id,
        artifact_name=artifact_name,
        destination_dir=destination_dir,
        filename=filename,
    )
    if artifact_path is None:
        task = clearml_task(task_id)
        available = ", ".join(sorted(task.artifacts)) or "none"
        raise click.ClickException(
            f"ClearML task {task_id} has no artifact named {artifact_name!r}. "
            f"Available artifacts: {available}."
        )
    return artifact_path


def maybe_download_task_artifact(
    *,
    task_id: str,
    artifact_name: str,
    destination_dir: Path,
    filename: str | None = None,
) -> Path | None:
    task = clearml_task(task_id)
    artifact = task.artifacts.get(artifact_name)
    if artifact is None:
        return None

    return download_clearml_artifact(
        task_id=task_id,
        artifact_name=artifact_name,
        artifact=artifact,
        destination_dir=destination_dir,
        filename=filename,
    )


def task_has_output_model(task_id: str, model_name: str | None = None) -> bool:
    return task_model(
        task_id=task_id,
        model_role="output",
        model_name=model_name,
        required=False,
    ) is not None


def download_task_output_model(
    *,
    task_id: str,
    destination_dir: Path,
    filename: str | None = None,
    model_name: str | None = None,
    connect_to_task: Any | None = None,
) -> Path:
    model = task_model(
        task_id=task_id,
        model_role="output",
        model_name=model_name,
        required=True,
    )
    if connect_to_task is not None:
        connect_input_model(connect_to_task, model)
    return download_clearml_model(
        task_id=task_id,
        model=model,
        destination_dir=destination_dir,
        filename=filename,
    )


def maybe_download_task_input_model(
    *,
    task_id: str,
    destination_dir: Path,
    filename: str | None = None,
    model_name: str | None = None,
    connect_to_task: Any | None = None,
) -> Path | None:
    model = task_model(
        task_id=task_id,
        model_role="input",
        model_name=model_name,
        required=False,
    )
    if model is None:
        return None
    if connect_to_task is not None:
        connect_input_model(connect_to_task, model)
    return download_clearml_model(
        task_id=task_id,
        model=model,
        destination_dir=destination_dir,
        filename=filename,
    )


def clearml_task(task_id: str) -> Any:
    try:
        from clearml import Task
    except ImportError as error:
        raise click.ClickException(
            "ClearML task access requires the clearml Python package. "
            "Run `uv sync` before using this command."
        ) from error

    return Task.get_task(task_id=task_id)


def task_model(
    *,
    task_id: str,
    model_role: str,
    model_name: str | None = None,
    required: bool,
) -> Any | None:
    task = clearml_task(task_id)
    models = task_models(task, model_role)
    if model_name is not None:
        for model in models:
            if str(getattr(model, "name", "")) == model_name:
                return model
        if not required:
            return None
        available = ", ".join(model_label(model) for model in models) or "none"
        raise click.ClickException(
            f"ClearML task {task_id} has no {model_role} model named {model_name!r}. "
            f"Available {model_role} models: {available}."
        )

    if len(models) == 1:
        return models[0]
    if not models:
        if not required:
            return None
        raise click.ClickException(
            f"ClearML task {task_id} has no {model_role} models."
        )
    if not required:
        return None

    available = ", ".join(model_label(model) for model in models)
    raise click.ClickException(
        f"ClearML task {task_id} has multiple {model_role} models. "
        f"Pass a model name. Available {model_role} models: {available}."
    )


def task_models(task: Any, model_role: str) -> list[Any]:
    get_models = getattr(task, "get_models", None)
    models = get_models() if callable(get_models) else getattr(task, "models", {})
    role_models = models.get(model_role, ()) if models is not None else ()
    return list(role_models or ())


def connect_input_model(task: Any, model: Any) -> None:
    try:
        from clearml import InputModel
    except ImportError as error:
        raise click.ClickException(
            "ClearML model tracking requires the clearml Python package. "
            "Run `uv sync` before using this command."
        ) from error

    model_id = str(getattr(model, "id", "") or "")
    if not model_id:
        return
    task.connect(InputModel(model_id=model_id), ignore_remote_overrides=True)


def download_clearml_model(
    *,
    task_id: str,
    model: Any,
    destination_dir: Path,
    filename: str | None = None,
) -> Path:
    model_id = str(getattr(model, "id", "") or "")
    if not model_id:
        raise click.ClickException(
            f"ClearML task {task_id} returned a model without an ID: {model!r}"
        )

    with clearml_artifact_download_lock(task_id=model_id, artifact_name="model"):
        local_copy = model.get_local_copy(raise_on_error=True)
        return stage_downloaded_clearml_file(
            local_copy=local_copy,
            label=f"ClearML model {model_label(model)} from task {task_id}",
            destination_dir=destination_dir,
            filename=filename,
        )


def model_label(model: Any) -> str:
    model_name = str(getattr(model, "name", "") or "")
    model_id = str(getattr(model, "id", "") or "")
    if model_name and model_id:
        return f"{model_name} ({model_id})"
    return model_name or model_id or repr(model)


def download_clearml_artifact(
    *,
    task_id: str,
    artifact_name: str,
    artifact: Any,
    destination_dir: Path,
    filename: str | None = None,
) -> Path:
    with clearml_artifact_download_lock(task_id=task_id, artifact_name=artifact_name):
        local_copy = artifact.get_local_copy()
        return stage_downloaded_clearml_file(
            local_copy=local_copy,
            label=f"ClearML artifact {artifact_name!r} from task {task_id}",
            destination_dir=destination_dir,
            filename=filename,
        )


def stage_downloaded_clearml_file(
    *,
    local_copy: str | None,
    label: str,
    destination_dir: Path,
    filename: str | None,
) -> Path:
    if local_copy is None:
        raise click.ClickException(f"Could not download {label}.")

    source = Path(local_copy)
    if not source.exists():
        raise click.ClickException(f"Downloaded {label} path does not exist: {source}")
    if source.is_dir():
        raise click.ClickException(
            f"{label} is a directory; this CLI expects a single file."
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (filename or source.name)
    if source.resolve() != destination.resolve():
        try:
            shutil.copy2(source, destination)
        except FileNotFoundError as error:
            raise click.ClickException(
                f"Downloaded {label} disappeared before it could be staged: {source}"
            ) from error
    return destination


@contextmanager
def clearml_artifact_download_lock(
    *,
    task_id: str,
    artifact_name: str,
) -> Iterator[None]:
    """Serialize ClearML cache downloads across parallel local pipeline steps."""
    lock_path = clearml_artifact_download_lock_path(
        task_id=task_id,
        artifact_name=artifact_name,
    )
    deadline = time.monotonic() + CLEARML_ARTIFACT_LOCK_TIMEOUT_SECONDS
    lock_fd: int | None = None

    while lock_fd is None:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if clearml_artifact_lock_is_stale(lock_path):
                try:
                    lock_path.unlink()
                    continue
                except FileNotFoundError:
                    continue

            if time.monotonic() >= deadline:
                raise click.ClickException(
                    "Timed out waiting for ClearML artifact download lock: "
                    f"{artifact_name!r} from task {task_id}."
                )
            time.sleep(CLEARML_ARTIFACT_LOCK_POLL_SECONDS)

    try:
        os.write(
            lock_fd,
            f"pid={os.getpid()} task_id={task_id} artifact={artifact_name}\n".encode(
                "utf-8"
            ),
        )
        yield
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def clearml_artifact_download_lock_path(
    *,
    task_id: str,
    artifact_name: str,
) -> Path:
    lock_key = sha256(f"{task_id}:{artifact_name}".encode("utf-8")).hexdigest()
    return CLEARML_ARTIFACT_LOCK_DIR / f"{lock_key}.lock"


def clearml_artifact_lock_is_stale(lock_path: Path) -> bool:
    try:
        lock_age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    return lock_age > CLEARML_ARTIFACT_LOCK_TIMEOUT_SECONDS
