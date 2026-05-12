"""ClearML PipelineController task lookup and status helpers."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence

import click

from src.ml_core import cfg as core_cfg
from src.ml_core import clearml_tasks


COMPLETED_STATUSES = {"completed", "published"}
ACTIVE_STATUSES = {"created", "queued", "in_progress"}
FAILED_STATUSES = {"failed", "stopped", "aborted"}
TERMINAL_STATUSES = COMPLETED_STATUSES | FAILED_STATUSES


class StageTask(core_cfg.BaseCfg):
    id: str
    name: str
    status: str
    parent: str | None = None
    last_update: object | None = None


class ControllerCandidate(core_cfg.BaseCfg):
    id: str
    name: str
    status: str
    last_update: object | None = None


class StageEligibility(core_cfg.BaseCfg):
    eligible: bool
    reason: str
    stage_tasks: Mapping[str, tuple[StageTask, ...]]


def resolve_pipeline_controller_id(
    *,
    stage_name: str,
    pipeline_name: str,
    pipeline_version: str | None,
    clearml_project: str,
    parameter_filters: Mapping[str, object],
    stage_dependencies: Mapping[str, Sequence[str]] | None = None,
    stage_names: Sequence[str] = (),
) -> str:
    resolved_stage_dependencies = stage_dependencies or {stage_name: ()}
    resolved_stage_names = tuple(stage_names or resolved_stage_dependencies)
    candidates = list_pipeline_controller_candidates(
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=clearml_project,
    )
    reasons: list[str] = []
    for candidate in candidates:
        eligibility = pipeline_stage_eligibility(
            controller_id=candidate.id,
            stage_name=stage_name,
            parameter_filters=parameter_filters,
            stage_dependencies=resolved_stage_dependencies,
            stage_names=resolved_stage_names,
        )
        if eligibility.eligible:
            return candidate.id
        reasons.append(f"{candidate.id}: {eligibility.reason}")

    detail = ""
    if reasons:
        detail = " Checked candidates: " + "; ".join(reasons[:5])
    raise click.ClickException(
        f"Could not find an eligible ClearML pipeline controller run for stage {stage_name!r}. "
        "Start the earlier stages first or pass --pipeline-controller-id."
        f"{detail}"
    )


def list_pipeline_controller_candidates(
    *,
    pipeline_name: str,
    pipeline_version: str | None,
    clearml_project: str,
) -> tuple[ControllerCandidate, ...]:
    try:
        from clearml.backend_api.session.client import APIClient
    except ImportError as error:
        raise click.ClickException(
            "ClearML pipeline lookup requires the clearml Python package. "
            "Run `uv sync` before using the pipeline CLIs."
        ) from error

    client = APIClient()
    project_name = f"{clearml_project}/.pipelines/{pipeline_name}"
    projects = client.projects.get_all(name=project_name, search_hidden=True)
    project_rows = list(getattr(projects, "response", []) or [])
    if not project_rows:
        return ()

    # Pipeline controller runs are just a special type of ClearML task.
    tasks = client.tasks.get_all(
        project=[project_rows[0].id],
        system_tags=["pipeline"],
        search_hidden=True,
        only_fields=["id", "name", "status", "last_update", "runtime"],
        order_by=["-last_update"],
    )
    candidates: list[ControllerCandidate] = []
    name_pattern = re.compile(rf"^{re.escape(pipeline_name)}( #[0-9]+)?$")
    for task in tasks or []:
        runtime = getattr(task, "runtime", {}) or {}
        if pipeline_version is not None:
            if str(runtime.get("version") or "") != str(pipeline_version):
                continue
        if not name_pattern.match(str(task.name)):
            continue
        candidates.append(
            ControllerCandidate(
                id=str(task.id),
                name=str(task.name),
                status=status_value(getattr(task, "status", "")),
                last_update=getattr(task, "last_update", None),
            )
        )
    return tuple(candidates)


def assert_controller_can_run_stage(
    *,
    controller_id: str,
    stage_name: str,
    parameter_filters: Mapping[str, object],
    stage_dependencies: Mapping[str, Sequence[str]] | None = None,
    stage_names: Sequence[str] = (),
) -> None:
    resolved_stage_dependencies = stage_dependencies or {stage_name: ()}
    resolved_stage_names = tuple(stage_names or resolved_stage_dependencies)
    eligibility = pipeline_stage_eligibility(
        controller_id=controller_id,
        stage_name=stage_name,
        parameter_filters=parameter_filters,
        stage_dependencies=resolved_stage_dependencies,
        stage_names=resolved_stage_names,
    )
    if not eligibility.eligible:
        raise click.ClickException(
            f"Pipeline controller {controller_id} cannot run stage {stage_name!r}: "
            f"{eligibility.reason}"
        )


def pipeline_stage_eligibility(
    *,
    controller_id: str,
    stage_name: str,
    parameter_filters: Mapping[str, object],
    stage_dependencies: Mapping[str, Sequence[str]] | None = None,
    stage_names: Sequence[str] = (),
) -> StageEligibility:
    resolved_stage_dependencies = stage_dependencies or {stage_name: ()}
    resolved_stage_names = tuple(stage_names or resolved_stage_dependencies)
    if stage_name not in resolved_stage_dependencies:
        return StageEligibility(
            eligible=False,
            reason=f"unknown stage {stage_name!r}",
            stage_tasks={},
        )

    if not controller_parameters_match(controller_id, parameter_filters):
        return StageEligibility(
            eligible=False,
            reason="controller experiment parameters do not match this CLI request",
            stage_tasks={},
        )

    controller = clearml_tasks.clearml_task(controller_id)
    controller_status = status_value(getattr(controller, "status", ""))
    if controller_status in ACTIVE_STATUSES:
        return StageEligibility(
            eligible=False,
            reason=f"controller is already {controller_status}",
            stage_tasks={},
        )

    stage_tasks = pipeline_stage_tasks(controller_id, stage_names=resolved_stage_names)
    missing_dependencies = [
        dependency
        for dependency in resolved_stage_dependencies[stage_name]
        if not any(
            task.status in COMPLETED_STATUSES
            for task in stage_tasks.get(dependency, ())
        )
    ]
    if missing_dependencies:
        return StageEligibility(
            eligible=False,
            reason="missing completed dependencies: " + ", ".join(missing_dependencies),
            stage_tasks=stage_tasks,
        )

    target_tasks = stage_tasks.get(stage_name, ())
    completed_targets = [task for task in target_tasks if task.status in COMPLETED_STATUSES]
    if completed_targets:
        return StageEligibility(
            eligible=False,
            reason=f"stage {stage_name!r} already completed",
            stage_tasks=stage_tasks,
        )
    active_targets = [task for task in target_tasks if task.status in ACTIVE_STATUSES]
    if active_targets:
        return StageEligibility(
            eligible=False,
            reason=f"stage {stage_name!r} is already active",
            stage_tasks=stage_tasks,
        )
    return StageEligibility(
        eligible=True,
        reason="eligible",
        stage_tasks=stage_tasks,
    )


def controller_parameters_match(
    controller_id: str,
    parameter_filters: Mapping[str, object],
) -> bool:
    if not parameter_filters:
        return True
    task = clearml_tasks.clearml_task(controller_id)
    get_parameters = getattr(task, "get_parameters", None)
    if not callable(get_parameters):
        return False
    parameters = get_parameters(cast=False) or {}
    for key, expected_value in parameter_filters.items():
        actual_value = parameters.get(f"Experiment/{key}")
        if actual_value is None:
            return False
        if str(actual_value) != str(expected_value if expected_value is not None else ""):
            return False
    return True


def pipeline_stage_tasks(
    controller_id: str,
    *,
    stage_names: Sequence[str],
) -> dict[str, tuple[StageTask, ...]]:
    try:
        from clearml.backend_api.session.client import APIClient
    except ImportError as error:
        raise click.ClickException(
            "ClearML stage lookup requires the clearml Python package. "
            "Run `uv sync` before using the pipeline CLIs."
        ) from error

    client = APIClient()
    tasks = client.tasks.get_all(
        parent=controller_id,
        search_hidden=True,
        only_fields=["id", "name", "status", "parent", "last_update"],
        order_by=["-last_update"],
    )
    grouped: dict[str, list[StageTask]] = {stage: [] for stage in stage_names}
    for task in tasks or []:
        name = str(task.name)
        if name not in grouped:
            continue
        grouped[name].append(
            StageTask(
                id=str(task.id),
                name=name,
                status=status_value(getattr(task, "status", "")),
                parent=str(getattr(task, "parent", "") or "") or None,
                last_update=getattr(task, "last_update", None),
            )
        )
    return {stage: tuple(tasks) for stage, tasks in grouped.items()}


def print_stage_task_ids(
    pipeline_task_id: str,
    stages: Sequence[str],
    *,
    stage_names: Sequence[str],
) -> None:
    stage_tasks = pipeline_stage_tasks(pipeline_task_id, stage_names=stage_names)
    for stage in stages:
        task_ids = [task.id for task in stage_tasks.get(stage, ())]
        if task_ids:
            click.echo(f"ClearML stage task ID ({stage}): {task_ids[0]}")


def assert_pipeline_finished_successfully(pipeline: object) -> None:
    task = getattr(pipeline, "task", None)
    if task is None:
        return

    reload_task = getattr(task, "reload", None)
    if callable(reload_task):
        reload_task()

    status = status_value(getattr(task, "status", ""))
    if status in FAILED_STATUSES:
        raise click.ClickException(
            f"ClearML pipeline finished with status {status}. "
            "Open the controller task or failed stage task for details."
        )


def wait_for_controller_completion(controller_id: str, *, poll_seconds: float = 5.0) -> None:
    task = clearml_tasks.clearml_task(controller_id)
    while True:
        reload_task = getattr(task, "reload", None)
        if callable(reload_task):
            reload_task()
        status = status_value(getattr(task, "status", ""))
        if status in TERMINAL_STATUSES:
            return
        time.sleep(poll_seconds)


def assert_controller_task_succeeded(controller_id: str) -> None:
    task = clearml_tasks.clearml_task(controller_id)
    reload_task = getattr(task, "reload", None)
    if callable(reload_task):
        reload_task()
    status = status_value(getattr(task, "status", ""))
    if status in FAILED_STATUSES:
        raise click.ClickException(
            f"ClearML pipeline controller {controller_id} finished with status {status}."
        )


def enqueue_pipeline_controller(controller_id: str, queue_name: str) -> None:
    try:
        from clearml.automation import PipelineController
    except ImportError as error:
        raise click.ClickException(
            "ClearML controller enqueue requires the clearml Python package. "
            "Run `uv sync` before using the pipeline CLIs."
        ) from error
    PipelineController.enqueue(controller_id, queue_name=queue_name, force=True)


def status_value(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value or "").lower()
