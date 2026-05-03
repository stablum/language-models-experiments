"""Shared helpers for ClearML PipelineController runs."""

from __future__ import annotations

import re
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from src.tracking.clearml import (
    assert_clearml_endpoints_reachable,
    clearml_settings,
    configure_clearml_config_file,
)


DEFAULT_MODEL_TRAINING_NAME = "model-training"
DEFAULT_TOKENIZER_TRAINING_NAME = "tokenizer-training"
DEFAULT_CONTROLLER_QUEUE = "services"
DEFAULT_PIPELINE_VERSION_FALLBACK = "0.4.21"

TOKENIZER_STAGE = "train_tokenizer"
MODEL_STAGE = "train_model"
EVALUATION_STAGE = "evaluate"
QUERY_STAGE = "query"
ALL_PIPELINE_STAGES = (TOKENIZER_STAGE, MODEL_STAGE, EVALUATION_STAGE, QUERY_STAGE)
TOKENIZER_TRAINING_STAGES = (TOKENIZER_STAGE,)
MODEL_TRAINING_STAGES = (MODEL_STAGE, EVALUATION_STAGE, QUERY_STAGE)
ALL_PIPELINE_STAGE_DEPENDENCIES = {
    TOKENIZER_STAGE: (),
    MODEL_STAGE: (TOKENIZER_STAGE,),
    EVALUATION_STAGE: (MODEL_STAGE,),
    QUERY_STAGE: (MODEL_STAGE,),
}
TOKENIZER_TRAINING_STAGE_DEPENDENCIES = {
    TOKENIZER_STAGE: (),
}
MODEL_TRAINING_STAGE_DEPENDENCIES = {
    MODEL_STAGE: (),
    EVALUATION_STAGE: (MODEL_STAGE,),
    QUERY_STAGE: (MODEL_STAGE,),
}
PIPELINE_STAGE_INDEX = {stage: index for index, stage in enumerate(ALL_PIPELINE_STAGES)}
TOKENIZER_MODEL_ARTIFACT = "sentencepiece-model"

PIPELINE_CONTROL_SECTION = "Pipeline Control"
PIPELINE_CONTROL_MODE = f"{PIPELINE_CONTROL_SECTION}/run_mode"
PIPELINE_CONTROL_RUN_STAGE = f"{PIPELINE_CONTROL_SECTION}/run_stage"
PIPELINE_CONTROL_RUN_UNTIL_STAGE = f"{PIPELINE_CONTROL_SECTION}/run_until_stage"
PIPELINE_CONTROL_UPDATED_BY = f"{PIPELINE_CONTROL_SECTION}/updated_by"

PIPELINE_MODE_ALL = "all"
PIPELINE_MODE_RUN_STAGE = "run_stage"
PIPELINE_MODE_RUN_UNTIL = "run_until"

COMPLETED_STATUSES = {"completed", "published"}
ACTIVE_STATUSES = {"created", "queued", "in_progress"}
FAILED_STATUSES = {"failed", "stopped", "aborted"}
TERMINAL_STATUSES = COMPLETED_STATUSES | FAILED_STATUSES


@dataclass(frozen=True)
class PipelineControl:
    mode: str
    run_stage: str | None = None
    run_until_stage: str | None = None


@dataclass(frozen=True)
class StageTask:
    id: str
    name: str
    status: str
    parent: str | None = None
    last_update: object | None = None


@dataclass(frozen=True)
class ControllerCandidate:
    id: str
    name: str
    status: str
    last_update: object | None = None


@dataclass(frozen=True)
class TokenizerTrainingResolution:
    controller_id: str
    tokenizer_task_id: str
    tokenizer_model_name: str
    corpus: str


@dataclass(frozen=True)
class StageEligibility:
    eligible: bool
    reason: str
    stage_tasks: Mapping[str, tuple[StageTask, ...]]


def project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as pyproject_file:
            data = tomllib.load(pyproject_file)
    except OSError:
        return DEFAULT_PIPELINE_VERSION_FALLBACK

    project = data.get("project")
    if not isinstance(project, dict):
        return DEFAULT_PIPELINE_VERSION_FALLBACK
    version = project.get("version")
    return str(version) if version else DEFAULT_PIPELINE_VERSION_FALLBACK


DEFAULT_PIPELINE_VERSION = project_version()


def pipeline_options(
    *,
    default_name: str = DEFAULT_MODEL_TRAINING_NAME,
    default_local: bool = True,
    default_wait: bool = True,
) -> Callable[[Any], Any]:
    def decorator(command: Any) -> Any:
        command = click.option(
            "--add-run-number/--no-add-run-number",
            default=True,
            show_default=True,
            help="Append ClearML's run number to newly created controller task names.",
        )(command)
        command = click.option(
            "--wait/--no-wait",
            default=default_wait,
            show_default=True,
            help="Wait for local pipeline completion or poll an enqueued controller.",
        )(command)
        command = click.option(
            "--execution-queue",
            default=None,
            help="ClearML queue for step tasks when --pipeline-queued is used.",
        )(command)
        command = click.option(
            "--controller-queue",
            default=DEFAULT_CONTROLLER_QUEUE,
            show_default=True,
            help="ClearML queue for the controller when --pipeline-queued is used or a controller is resumed.",
        )(command)
        command = click.option(
            "--pipeline-local/--pipeline-queued",
            default=default_local,
            show_default=True,
            help="Run a new controller locally, or enqueue it. Existing controller resumes are always queued.",
        )(command)
        command = click.option(
            "--pipeline-version",
            default=DEFAULT_PIPELINE_VERSION,
            show_default=True,
            help="Reusable ClearML pipeline DAG version.",
        )(command)
        command = click.option(
            "--pipeline-name",
            default=default_name,
            show_default=True,
            help="Reusable ClearML pipeline DAG name.",
        )(command)
        return command

    return decorator


def pipeline_resume_option(command: Any) -> Any:
    return click.option(
        "--pipeline-controller-id",
        default=None,
        help=(
            "Existing ClearML PipelineController task ID to resume. "
            "If omitted, the newest eligible controller run is selected."
        ),
    )(command)


def build_pipeline_controller(
    *,
    pipeline_name: str,
    pipeline_version: str,
    clearml_project: str,
    clearml_tags: tuple[str, ...],
    clearml_output_uri: str | None,
    add_run_number: bool,
) -> object:
    try:
        from clearml.automation import PipelineController
    except ImportError as error:
        raise click.ClickException(
            "ClearML pipelines require the clearml Python package. "
            "Run `uv sync` before using the pipeline CLIs."
        ) from error

    pipeline = PipelineController(
        name=pipeline_name,
        project=clearml_project,
        version=pipeline_version,
        add_pipeline_tags=False,
        target_project=clearml_project,
        abort_on_failure=True,
        add_run_number=add_run_number,
        output_uri=output_uri_value(clearml_output_uri),
        working_dir=str(Path.cwd()),
    )
    if clearml_tags:
        pipeline.add_tags(list(dict.fromkeys(clearml_tags)))
    return pipeline


def configure_pipeline_control(
    task: object,
    *,
    run_stage: str | None,
    run_until_stage: str | None,
    updated_by: str,
    preserve_remote_control: bool = True,
) -> PipelineControl:
    validate_stage_selection(run_stage=run_stage, run_until_stage=run_until_stage)
    existing_control = pipeline_control_from_task(task)
    if preserve_remote_control and _task_is_remote_current_task(task) and existing_control is not None:
        return existing_control

    control = PipelineControl(
        mode=(
            PIPELINE_MODE_RUN_STAGE
            if run_stage is not None
            else PIPELINE_MODE_RUN_UNTIL
            if run_until_stage is not None
            else PIPELINE_MODE_ALL
        ),
        run_stage=run_stage,
        run_until_stage=run_until_stage,
    )
    _set_task_parameter(task, PIPELINE_CONTROL_MODE, control.mode)
    _set_task_parameter(task, PIPELINE_CONTROL_RUN_STAGE, run_stage or "")
    _set_task_parameter(task, PIPELINE_CONTROL_RUN_UNTIL_STAGE, run_until_stage or "")
    _set_task_parameter(task, PIPELINE_CONTROL_UPDATED_BY, updated_by)
    return control


def connect_controller_experiment_parameters(
    task: object,
    parameters: Mapping[str, object],
) -> None:
    for key, value in parameters.items():
        if value is None:
            value = ""
        _set_task_parameter(task, f"Experiment/{key}", str(value))


def validate_stage_selection(
    *,
    run_stage: str | None,
    run_until_stage: str | None,
) -> None:
    if run_stage is not None and run_until_stage is not None:
        raise click.ClickException("--run-stage and --run-until-stage are mutually exclusive.")
    for option_name, stage in (("--run-stage", run_stage), ("--run-until-stage", run_until_stage)):
        if stage is not None and stage not in PIPELINE_STAGE_INDEX:
            choices = ", ".join(ALL_PIPELINE_STAGES)
            raise click.ClickException(f"{option_name} must be one of: {choices}.")


def stage_gate_callback(pipeline: object, node: object, _parameters: dict[str, object]) -> bool:
    task = getattr(pipeline, "task", None)
    control = pipeline_control_from_task(task) if task is not None else None
    stage = str(getattr(node, "name", ""))
    if control is None or stage_allowed_by_control(stage, control):
        return True

    click.echo(f"Skipping pipeline stage [{stage}] for control mode [{control.mode}].")
    return False


def pipeline_control_from_task(task: object | None) -> PipelineControl | None:
    if task is None:
        return None
    get_parameters = getattr(task, "get_parameters", None)
    if not callable(get_parameters):
        return None
    parameters = get_parameters(cast=False) or {}
    mode = str(parameters.get(PIPELINE_CONTROL_MODE) or "").strip()
    if not mode:
        return None
    run_stage = _optional_stage(parameters.get(PIPELINE_CONTROL_RUN_STAGE))
    run_until_stage = _optional_stage(parameters.get(PIPELINE_CONTROL_RUN_UNTIL_STAGE))
    if mode == PIPELINE_MODE_RUN_STAGE:
        return PipelineControl(mode=mode, run_stage=run_stage)
    if mode == PIPELINE_MODE_RUN_UNTIL:
        return PipelineControl(mode=mode, run_until_stage=run_until_stage)
    return PipelineControl(mode=PIPELINE_MODE_ALL)


def stage_allowed_by_control(stage: str, control: PipelineControl) -> bool:
    if stage not in PIPELINE_STAGE_INDEX:
        return True
    if control.mode == PIPELINE_MODE_ALL:
        return True
    if control.mode == PIPELINE_MODE_RUN_STAGE:
        return stage == control.run_stage
    if control.mode == PIPELINE_MODE_RUN_UNTIL and control.run_until_stage is not None:
        return PIPELINE_STAGE_INDEX[stage] <= PIPELINE_STAGE_INDEX[control.run_until_stage]
    return True


def resume_pipeline_controller_stage(
    *,
    stage_name: str,
    pipeline_controller_id: str | None,
    pipeline_name: str,
    pipeline_version: str,
    controller_queue: str,
    wait: bool,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    parameter_filters: Mapping[str, object] | None = None,
    stage_dependencies: Mapping[str, Sequence[str]] = ALL_PIPELINE_STAGE_DEPENDENCIES,
    stage_names: Sequence[str] = ALL_PIPELINE_STAGES,
) -> str:
    settings = clearml_settings(
        project_name=clearml_project,
        task_name=clearml_task_name,
        config_file=clearml_config_file,
        connectivity_check=clearml_connectivity_check,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
    )
    resolved_config_file = configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        assert_clearml_endpoints_reachable(resolved_config_file, settings.output_uri)

    resolved_pipeline_name = settings.task_name or pipeline_name
    resolved_controller_id = pipeline_controller_id or resolve_pipeline_controller_id(
        stage_name=stage_name,
        pipeline_name=resolved_pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=settings.project_name,
        parameter_filters=parameter_filters or {},
        stage_dependencies=stage_dependencies,
        stage_names=stage_names,
    )
    task = _get_clearml_task(resolved_controller_id)
    assert_controller_can_run_stage(
        controller_id=resolved_controller_id,
        stage_name=stage_name,
        parameter_filters=parameter_filters or {},
        stage_dependencies=stage_dependencies,
        stage_names=stage_names,
    )
    configure_pipeline_control(
        task,
        run_stage=stage_name,
        run_until_stage=None,
        updated_by="stage-cli",
        preserve_remote_control=False,
    )
    _enqueue_pipeline_controller(resolved_controller_id, controller_queue)

    click.echo(f"ClearML pipeline controller task ID: {resolved_controller_id}")
    click.echo(f"Resumed stage: {stage_name}")
    click.echo(f"Controller queue: {controller_queue}")
    if wait:
        wait_for_controller_completion(resolved_controller_id)
        assert_controller_task_succeeded(resolved_controller_id)
        print_stage_task_ids(resolved_controller_id, (stage_name,), stage_names=stage_names)
        click.echo("ClearML pipeline run completed.")
    else:
        click.echo("ClearML pipeline controller enqueued.")
    return resolved_controller_id


def resolve_pipeline_controller_id(
    *,
    stage_name: str,
    pipeline_name: str,
    pipeline_version: str | None,
    clearml_project: str,
    parameter_filters: Mapping[str, object],
    stage_dependencies: Mapping[str, Sequence[str]] = ALL_PIPELINE_STAGE_DEPENDENCIES,
    stage_names: Sequence[str] = ALL_PIPELINE_STAGES,
) -> str:
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
            stage_dependencies=stage_dependencies,
            stage_names=stage_names,
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


def resolve_tokenizer_training_task(
    *,
    tokenizer_training_name: str,
    clearml_project: str,
    corpus: str,
    tokenizer_model_name: str,
) -> TokenizerTrainingResolution:
    candidates = list_pipeline_controller_candidates(
        pipeline_name=tokenizer_training_name,
        pipeline_version=None,
        clearml_project=clearml_project,
    )
    parameter_filters = {
        "corpus": corpus,
        "tokenizer_model_name": tokenizer_model_name,
    }
    reasons: list[str] = []
    for candidate in candidates:
        if candidate.status not in COMPLETED_STATUSES:
            reasons.append(f"{candidate.id}: controller status is {candidate.status}")
            continue
        if not controller_parameters_match(candidate.id, parameter_filters):
            reasons.append(f"{candidate.id}: tokenizer parameters do not match")
            continue

        stage_tasks = pipeline_stage_tasks(
            candidate.id,
            stage_names=TOKENIZER_TRAINING_STAGES,
        )
        completed_tokenizer_tasks = [
            task
            for task in stage_tasks.get(TOKENIZER_STAGE, ())
            if task.status in COMPLETED_STATUSES
        ]
        if not completed_tokenizer_tasks:
            reasons.append(f"{candidate.id}: no completed {TOKENIZER_STAGE} stage task")
            continue

        for stage_task in completed_tokenizer_tasks:
            if _task_has_artifact(stage_task.id, TOKENIZER_MODEL_ARTIFACT):
                return TokenizerTrainingResolution(
                    controller_id=candidate.id,
                    tokenizer_task_id=stage_task.id,
                    tokenizer_model_name=tokenizer_model_name,
                    corpus=corpus,
                )
        reasons.append(
            f"{candidate.id}: completed {TOKENIZER_STAGE} task has no "
            f"{TOKENIZER_MODEL_ARTIFACT!r} artifact"
        )

    detail = ""
    if reasons:
        detail = " Checked candidates: " + "; ".join(reasons[:5])
    raise click.ClickException(
        "Could not find a completed tokenizer-training run for "
        f"corpus={corpus!r} and tokenizer_model_name={tokenizer_model_name!r}. "
        f"Run `python -m src.cli.tokenizer_training` first, or change "
        f"--tokenizer-training-name from {tokenizer_training_name!r}."
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

    # pipeline controller runs are just a special type of task
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
                status=_status_value(getattr(task, "status", "")),
                last_update=getattr(task, "last_update", None),
            )
        )
    return tuple(candidates)


def assert_controller_can_run_stage(
    *,
    controller_id: str,
    stage_name: str,
    parameter_filters: Mapping[str, object],
    stage_dependencies: Mapping[str, Sequence[str]] = ALL_PIPELINE_STAGE_DEPENDENCIES,
    stage_names: Sequence[str] = ALL_PIPELINE_STAGES,
) -> None:
    eligibility = pipeline_stage_eligibility(
        controller_id=controller_id,
        stage_name=stage_name,
        parameter_filters=parameter_filters,
        stage_dependencies=stage_dependencies,
        stage_names=stage_names,
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
    stage_dependencies: Mapping[str, Sequence[str]] = ALL_PIPELINE_STAGE_DEPENDENCIES,
    stage_names: Sequence[str] = ALL_PIPELINE_STAGES,
) -> StageEligibility:
    if stage_name not in stage_dependencies:
        return StageEligibility(False, f"unknown stage {stage_name!r}", {})

    if not controller_parameters_match(controller_id, parameter_filters):
        return StageEligibility(False, "controller experiment parameters do not match this CLI request", {})

    controller = _get_clearml_task(controller_id)
    controller_status = _status_value(getattr(controller, "status", ""))
    if controller_status in ACTIVE_STATUSES:
        return StageEligibility(False, f"controller is already {controller_status}", {})

    stage_tasks = pipeline_stage_tasks(controller_id, stage_names=stage_names)
    missing_dependencies = [
        dependency
        for dependency in stage_dependencies[stage_name]
        if not any(
            task.status in COMPLETED_STATUSES
            for task in stage_tasks.get(dependency, ())
        )
    ]
    if missing_dependencies:
        return StageEligibility(
            False,
            "missing completed dependencies: " + ", ".join(missing_dependencies),
            stage_tasks,
        )

    target_tasks = stage_tasks.get(stage_name, ())
    completed_targets = [task for task in target_tasks if task.status in COMPLETED_STATUSES]
    if completed_targets:
        return StageEligibility(False, f"stage {stage_name!r} already completed", stage_tasks)
    active_targets = [task for task in target_tasks if task.status in ACTIVE_STATUSES]
    if active_targets:
        return StageEligibility(False, f"stage {stage_name!r} is already active", stage_tasks)
    return StageEligibility(True, "eligible", stage_tasks)


def controller_parameters_match(
    controller_id: str,
    parameter_filters: Mapping[str, object],
) -> bool:
    if not parameter_filters:
        return True
    task = _get_clearml_task(controller_id)
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
    stage_names: Sequence[str] = ALL_PIPELINE_STAGES,
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
                status=_status_value(getattr(task, "status", "")),
                parent=str(getattr(task, "parent", "") or "") or None,
                last_update=getattr(task, "last_update", None),
            )
        )
    return {stage: tuple(tasks) for stage, tasks in grouped.items()}


def print_stage_task_ids(
    pipeline_task_id: str,
    stages: Sequence[str],
    *,
    stage_names: Sequence[str] = ALL_PIPELINE_STAGES,
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

    status = _status_value(getattr(task, "status", ""))
    if status in FAILED_STATUSES:
        raise click.ClickException(
            f"ClearML pipeline finished with status {status}. "
            "Open the controller task or failed stage task for details."
        )


def wait_for_controller_completion(controller_id: str, *, poll_seconds: float = 5.0) -> None:
    task = _get_clearml_task(controller_id)
    while True:
        reload_task = getattr(task, "reload", None)
        if callable(reload_task):
            reload_task()
        status = _status_value(getattr(task, "status", ""))
        if status in TERMINAL_STATUSES:
            return
        time.sleep(poll_seconds)


def assert_controller_task_succeeded(controller_id: str) -> None:
    task = _get_clearml_task(controller_id)
    reload_task = getattr(task, "reload", None)
    if callable(reload_task):
        reload_task()
    status = _status_value(getattr(task, "status", ""))
    if status in FAILED_STATUSES:
        raise click.ClickException(
            f"ClearML pipeline controller {controller_id} finished with status {status}."
        )


def output_uri_value(clearml_output_uri: str | None) -> str | bool:
    return clearml_output_uri if clearml_output_uri is not None else True


def _set_task_parameter(task: object, name: str, value: str) -> None:
    setter = getattr(task, "set_parameter", None)
    if not callable(setter):
        return
    setter(name, value)


def _task_is_remote_current_task(task: object) -> bool:
    try:
        from clearml import Task
    except ImportError:
        return False
    current_task = Task.current_task()
    return (
        current_task is not None
        and str(getattr(current_task, "id", "")) == str(getattr(task, "id", ""))
        and not Task.running_locally()
    )


def _optional_stage(value: object) -> str | None:
    stage = str(value or "").strip()
    return stage or None


def _get_clearml_task(task_id: str) -> object:
    try:
        from clearml import Task
    except ImportError as error:
        raise click.ClickException(
            "ClearML task access requires the clearml Python package. "
            "Run `uv sync` before using the pipeline CLIs."
        ) from error
    return Task.get_task(task_id=task_id)


def _task_has_artifact(task_id: str, artifact_name: str) -> bool:
    task = _get_clearml_task(task_id)
    artifacts = getattr(task, "artifacts", {}) or {}
    return artifact_name in artifacts


def _enqueue_pipeline_controller(controller_id: str, queue_name: str) -> None:
    try:
        from clearml.automation import PipelineController
    except ImportError as error:
        raise click.ClickException(
            "ClearML controller enqueue requires the clearml Python package. "
            "Run `uv sync` before using the pipeline CLIs."
        ) from error
    PipelineController.enqueue(controller_id, queue_name=queue_name, force=True)


def _status_value(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value or "").lower()
