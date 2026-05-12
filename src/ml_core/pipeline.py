"""Shared helpers for ClearML PipelineController runs."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from src.ml_core import cfg as core_cfg
from src.ml_core import clearml_tasks
from src.ml_core import pipeline_tasks
from src.ml_core import tracking


DEFAULT_PIPELINE_NAME = "pipeline"
DEFAULT_CONTROLLER_QUEUE = "services"
DEFAULT_PIPELINE_VERSION_FALLBACK = "0.0.0+unknown"

PIPELINE_CONTROL_SECTION = "Pipeline Control"
PIPELINE_CONTROL_MODE = f"{PIPELINE_CONTROL_SECTION}/run_mode"
PIPELINE_CONTROL_RUN_STAGE = f"{PIPELINE_CONTROL_SECTION}/run_stage"
PIPELINE_CONTROL_RUN_UNTIL_STAGE = f"{PIPELINE_CONTROL_SECTION}/run_until_stage"
PIPELINE_CONTROL_UPDATED_BY = f"{PIPELINE_CONTROL_SECTION}/updated_by"

PIPELINE_MODE_ALL = "all"
PIPELINE_MODE_RUN_STAGE = "run_stage"
PIPELINE_MODE_RUN_UNTIL = "run_until"


class PipelineControl(core_cfg.BaseCfg):
    mode: str
    run_stage: str | None = None
    run_until_stage: str | None = None


def project_version() -> str:
    pyproject_path = project_metadata_path()
    if pyproject_path is None:
        return DEFAULT_PIPELINE_VERSION_FALLBACK

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


def project_metadata_path() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
    return None


DEFAULT_PIPELINE_VERSION = project_version()


def pipeline_options(
    *,
    default_name: str = DEFAULT_PIPELINE_NAME,
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
    stage_names: Sequence[str] = (),
    preserve_remote_control: bool = True,
) -> PipelineControl:
    validate_stage_selection(
        run_stage=run_stage,
        run_until_stage=run_until_stage,
        stage_names=stage_names,
    )
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
    stage_names: Sequence[str] = (),
) -> None:
    if run_stage is not None and run_until_stage is not None:
        raise click.ClickException("--run-stage and --run-until-stage are mutually exclusive.")

    if not stage_names:
        return

    stage_name_set = set(stage_names)
    for option_name, stage in (("--run-stage", run_stage), ("--run-until-stage", run_until_stage)):
        if stage is not None and stage not in stage_name_set:
            choices = ", ".join(stage_names)
            raise click.ClickException(f"{option_name} must be one of: {choices}.")


def make_stage_gate_callback(stage_names: Sequence[str]) -> Callable[[object, object, dict[str, object]], bool]:
    def stage_gate_callback(
        pipeline: object,
        node: object,
        _parameters: dict[str, object],
    ) -> bool:
        task = getattr(pipeline, "task", None)
        control = pipeline_control_from_task(task) if task is not None else None
        stage = str(getattr(node, "name", ""))
        if control is None or stage_allowed_by_control(stage, control, stage_names=stage_names):
            return True

        click.echo(f"Skipping pipeline stage [{stage}] for control mode [{control.mode}].")
        return False

    return stage_gate_callback


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


def stage_allowed_by_control(
    stage: str,
    control: PipelineControl,
    *,
    stage_names: Sequence[str] = (),
) -> bool:
    pipeline_stage_index = {stage_name: index for index, stage_name in enumerate(stage_names)}
    if stage not in pipeline_stage_index:
        return True
    if control.mode == PIPELINE_MODE_ALL:
        return True
    if control.mode == PIPELINE_MODE_RUN_STAGE:
        return stage == control.run_stage
    if control.mode == PIPELINE_MODE_RUN_UNTIL and control.run_until_stage is not None:
        if control.run_until_stage not in pipeline_stage_index:
            return True
        return pipeline_stage_index[stage] <= pipeline_stage_index[control.run_until_stage]
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
    stage_dependencies: Mapping[str, Sequence[str]] | None = None,
    stage_names: Sequence[str] = (),
) -> str:
    resolved_stage_dependencies = stage_dependencies or {stage_name: ()}
    resolved_stage_names = tuple(stage_names or resolved_stage_dependencies)
    settings = tracking.clearml_settings(
        project_name=clearml_project,
        task_name=clearml_task_name,
        config_file=clearml_config_file,
        connectivity_check=clearml_connectivity_check,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
    )
    resolved_config_file = tracking.configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        tracking.assert_clearml_endpoints_reachable(
            resolved_config_file,
            settings.output_uri,
        )

    resolved_pipeline_name = settings.task_name or pipeline_name
    resolved_controller_id = pipeline_controller_id or pipeline_tasks.resolve_pipeline_controller_id(
        stage_name=stage_name,
        pipeline_name=resolved_pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=settings.project_name,
        parameter_filters=parameter_filters or {},
        stage_dependencies=resolved_stage_dependencies,
        stage_names=resolved_stage_names,
    )
    task = clearml_tasks.clearml_task(resolved_controller_id)
    pipeline_tasks.assert_controller_can_run_stage(
        controller_id=resolved_controller_id,
        stage_name=stage_name,
        parameter_filters=parameter_filters or {},
        stage_dependencies=resolved_stage_dependencies,
        stage_names=resolved_stage_names,
    )
    configure_pipeline_control(
        task,
        run_stage=stage_name,
        run_until_stage=None,
        updated_by="stage-cli",
        stage_names=resolved_stage_names,
        preserve_remote_control=False,
    )
    pipeline_tasks.enqueue_pipeline_controller(resolved_controller_id, controller_queue)

    click.echo(f"ClearML pipeline controller task ID: {resolved_controller_id}")
    click.echo(f"Resumed stage: {stage_name}")
    click.echo(f"Controller queue: {controller_queue}")
    if wait:
        pipeline_tasks.wait_for_controller_completion(resolved_controller_id)
        pipeline_tasks.assert_controller_task_succeeded(resolved_controller_id)
        pipeline_tasks.print_stage_task_ids(
            resolved_controller_id,
            (stage_name,),
            stage_names=resolved_stage_names,
        )
        click.echo("ClearML pipeline run completed.")
    else:
        click.echo("ClearML pipeline controller enqueued.")
    return resolved_controller_id


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
