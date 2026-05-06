"""Optuna search helpers for the model-training pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import click

from src.ml_core.cli.config import normalize_key
from src.pipelines.language_model.definition import (
    COMPLETED_STATUSES,
    EVALUATION_STAGE,
    MODEL_TRAINING_STAGES,
    pipeline_stage_tasks,
)
from src.models import registry as model_registry
from src.ml_core.tracking import clearml_task


OPTUNA_EVALUATION_ARTIFACT = "evaluation-summary"
DEFAULT_OPTUNA_METRIC = "perplexity"
DEFAULT_OPTUNA_DIRECTION = "minimize"

DistributionName = Literal["float", "int", "categorical"]


@dataclass(frozen=True)
class SearchParameter:
    value_type: type
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SearchSpec:
    parameter_name: str
    distribution: DistributionName
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    step: int | None = None
    choices: tuple[Any, ...] = ()


SEARCH_PARAMETERS = {
    "model_name": SearchParameter(str, choices=model_registry.model_names()),
    "smoothing": SearchParameter(float, minimum=0.0),
    "unigram_weight": SearchParameter(float, minimum=0.0),
    "bigram_weight": SearchParameter(float, minimum=0.0),
    "trigram_weight": SearchParameter(float, minimum=0.0),
    "discount": SearchParameter(float, minimum=0.0, maximum=1.0),
    "top_k": SearchParameter(int, minimum=1),
    "query_max_tokens": SearchParameter(int, minimum=0),
    "query_top_k": SearchParameter(int, minimum=1),
    "query_decoding": SearchParameter(str, choices=("sample", "most-probable")),
    "query_temperature": SearchParameter(float, minimum=0.0),
    "query_seed": SearchParameter(int),
}


def normalize_optuna_search_values(raw_values: object) -> tuple[str, ...]:
    if raw_values is None:
        return ()
    if isinstance(raw_values, str):
        return (raw_values,) if raw_values.strip() else ()
    if isinstance(raw_values, Sequence):
        return tuple(str(value) for value in raw_values if str(value).strip())
    raise click.ClickException(
        "Optuna search space must be one string or a list of strings."
    )


def parse_optuna_search_specs(raw_values: object) -> tuple[SearchSpec, ...]:
    specs = tuple(_parse_search_spec(value) for value in normalize_optuna_search_values(raw_values))
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in specs:
        if spec.parameter_name in seen:
            duplicates.append(spec.parameter_name)
        seen.add(spec.parameter_name)
    if duplicates:
        raise click.ClickException(
            "Optuna search parameters may only be specified once: "
            + ", ".join(sorted(set(duplicates)))
        )
    return specs


def sample_trial_parameters(trial: object, specs: Sequence[SearchSpec]) -> dict[str, object]:
    sampled: dict[str, object] = {}
    for spec in specs:
        if spec.distribution == "float":
            sampled[spec.parameter_name] = trial.suggest_float(
                spec.parameter_name,
                float(spec.low),
                float(spec.high),
                log=spec.log,
            )
        elif spec.distribution == "int":
            sampled[spec.parameter_name] = trial.suggest_int(
                spec.parameter_name,
                int(spec.low),
                int(spec.high),
                step=spec.step or 1,
                log=spec.log,
            )
        else:
            sampled[spec.parameter_name] = trial.suggest_categorical(
                spec.parameter_name,
                list(spec.choices),
            )
    return sampled


def describe_search_space(specs: Sequence[SearchSpec]) -> str:
    return ", ".join(_describe_search_spec(spec) for spec in specs)


def load_objective_metric(
    *,
    controller_id: str,
    metric_name: str,
    evaluation_partition: str,
) -> float:
    evaluation_task_id = latest_completed_evaluation_task_id(controller_id)
    payload = evaluation_summary_payload(evaluation_task_id)
    value = metric_from_payload(payload, metric_name, evaluation_partition)
    if value is None:
        value = reported_scalar_metric(
            evaluation_task_id,
            metric_name=metric_name,
            evaluation_partition=evaluation_partition,
        )
    if value is None or not math.isfinite(value):
        available = ", ".join(sorted(str(key) for key in payload)) or "none"
        raise click.ClickException(
            f"Objective metric {metric_name!r} was not found as a finite value "
            f"on evaluation task {evaluation_task_id}. Available evaluation "
            f"summary keys: {available}."
        )
    return value


def latest_completed_evaluation_task_id(controller_id: str) -> str:
    stage_tasks = pipeline_stage_tasks(
        controller_id,
        stage_names=MODEL_TRAINING_STAGES,
    )
    completed_evaluations = [
        task
        for task in stage_tasks.get(EVALUATION_STAGE, ())
        if task.status in COMPLETED_STATUSES
    ]
    if not completed_evaluations:
        raise click.ClickException(
            f"Pipeline controller {controller_id} has no completed {EVALUATION_STAGE!r} stage task."
        )
    return completed_evaluations[0].id


def evaluation_summary_payload(evaluation_task_id: str) -> Mapping[str, object]:
    task = clearml_task(evaluation_task_id)
    artifacts = getattr(task, "artifacts", {}) or {}
    artifact = artifacts.get(OPTUNA_EVALUATION_ARTIFACT)
    if artifact is None:
        available = ", ".join(sorted(artifacts)) or "none"
        raise click.ClickException(
            f"Evaluation task {evaluation_task_id} has no "
            f"{OPTUNA_EVALUATION_ARTIFACT!r} artifact. Available artifacts: {available}."
        )

    getter = getattr(artifact, "get", None)
    if callable(getter):
        payload = getter()
        if isinstance(payload, Mapping):
            return payload

    local_copy = _artifact_local_copy(artifact)
    if local_copy is not None:
        payload = _load_json_mapping(local_copy)
        if payload is not None:
            return payload

    raise click.ClickException(
        f"Could not read {OPTUNA_EVALUATION_ARTIFACT!r} from evaluation task "
        f"{evaluation_task_id} as a mapping."
    )


def metric_from_payload(
    payload: Mapping[str, object],
    metric_name: str,
    evaluation_partition: str,
) -> float | None:
    for candidate in metric_key_candidates(metric_name, evaluation_partition):
        if candidate not in payload:
            continue
        return finite_float(payload[candidate])
    return None


def reported_scalar_metric(
    evaluation_task_id: str,
    *,
    metric_name: str,
    evaluation_partition: str,
) -> float | None:
    task = clearml_task(evaluation_task_id)
    get_scalars = getattr(task, "get_reported_scalars", None)
    if not callable(get_scalars):
        return None

    scalars = get_scalars() or {}
    for title, series in scalar_metric_candidates(metric_name, evaluation_partition):
        title_values = scalars.get(title)
        if not isinstance(title_values, Mapping):
            continue
        series_values = title_values.get(series)
        if not isinstance(series_values, Mapping):
            continue
        y_values = series_values.get("y")
        if isinstance(y_values, Sequence) and y_values:
            value = finite_float(y_values[-1])
            if value is not None:
                return value
    return None


def metric_key_candidates(metric_name: str, evaluation_partition: str) -> tuple[str, ...]:
    cleaned = metric_name.strip()
    candidates = [cleaned]
    if cleaned.startswith("Evaluation/"):
        candidates.append(cleaned.removeprefix("Evaluation/"))
    for candidate in tuple(candidates):
        if candidate.startswith(f"{evaluation_partition}/"):
            candidates.append(candidate.removeprefix(f"{evaluation_partition}/"))
        if "/" in candidate:
            candidates.append(candidate.rsplit("/", maxsplit=1)[-1])
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def scalar_metric_candidates(
    metric_name: str,
    evaluation_partition: str,
) -> tuple[tuple[str, str], ...]:
    series_candidates: list[str] = []
    for candidate in metric_key_candidates(metric_name, evaluation_partition):
        series_candidates.append(candidate)
        if "/" not in candidate:
            series_candidates.append(f"{evaluation_partition}/{candidate}")
    if metric_name.startswith("Evaluation/"):
        series_candidates.append(metric_name.removeprefix("Evaluation/"))
    return tuple(
        ("Evaluation", series)
        for series in dict.fromkeys(series for series in series_candidates if series)
    )


def finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, int | float):
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def _parse_search_spec(raw_value: str) -> SearchSpec:
    if "=" not in raw_value:
        raise click.ClickException(
            "Optuna search specs must look like "
            "name=float:low:high[:log], name=int:low:high[:step][:log], "
            "or name=categorical:value1,value2."
        )
    raw_name, raw_expression = raw_value.split("=", maxsplit=1)
    parameter_name = normalize_key(raw_name.strip())
    if parameter_name not in SEARCH_PARAMETERS:
        supported = ", ".join(sorted(SEARCH_PARAMETERS))
        raise click.ClickException(
            f"Unsupported Optuna search parameter {raw_name!r}. Supported parameters: {supported}."
        )

    parts = [part.strip() for part in raw_expression.split(":")]
    distribution = parts[0].lower()
    if distribution in {"float", "logfloat", "log-float"}:
        return _parse_float_spec(parameter_name, parts, log_by_name=distribution != "float")
    if distribution in {"int", "integer", "logint", "log-int"}:
        return _parse_int_spec(parameter_name, parts, log_by_name=distribution not in {"int", "integer"})
    if distribution in {"categorical", "choice", "choices"}:
        return _parse_categorical_spec(parameter_name, raw_expression)
    raise click.ClickException(
        f"Unsupported Optuna distribution {distribution!r} in {raw_value!r}."
    )


def _parse_float_spec(
    parameter_name: str,
    parts: Sequence[str],
    *,
    log_by_name: bool,
) -> SearchSpec:
    parameter = _search_parameter(parameter_name)
    if parameter.value_type is not float:
        raise click.ClickException(f"{parameter_name!r} must use an int or categorical search.")
    if len(parts) not in (3, 4):
        raise click.ClickException(
            f"Float Optuna search for {parameter_name!r} must be float:low:high[:log]."
        )
    low = _parse_float(parts[1], parameter_name)
    high = _parse_float(parts[2], parameter_name)
    log = log_by_name or (len(parts) == 4 and parts[3].lower() == "log")
    if len(parts) == 4 and parts[3].lower() != "log":
        raise click.ClickException(
            f"Unknown float Optuna modifier {parts[3]!r} for {parameter_name!r}; expected log."
        )
    _validate_bounds(parameter_name, low, high, parameter, log=log)
    return SearchSpec(parameter_name, "float", low=low, high=high, log=log)


def _parse_int_spec(
    parameter_name: str,
    parts: Sequence[str],
    *,
    log_by_name: bool,
) -> SearchSpec:
    parameter = _search_parameter(parameter_name)
    if parameter.value_type is not int:
        raise click.ClickException(f"{parameter_name!r} must use a float or categorical search.")
    if len(parts) < 3:
        raise click.ClickException(
            f"Integer Optuna search for {parameter_name!r} must be int:low:high[:step][:log]."
        )
    low = _parse_int(parts[1], parameter_name)
    high = _parse_int(parts[2], parameter_name)
    step = 1
    log = log_by_name
    for modifier in parts[3:]:
        lowered = modifier.lower()
        if lowered == "log":
            log = True
            continue
        if lowered.startswith("step="):
            step = _parse_int(lowered.removeprefix("step="), parameter_name)
            continue
        step = _parse_int(lowered, parameter_name)

    if step < 1:
        raise click.ClickException(f"Integer Optuna step for {parameter_name!r} must be >= 1.")
    if log and step != 1:
        raise click.ClickException(
            f"Log integer Optuna search for {parameter_name!r} cannot use a custom step."
        )
    _validate_bounds(parameter_name, low, high, parameter, log=log)
    return SearchSpec(parameter_name, "int", low=low, high=high, step=step, log=log)


def _parse_categorical_spec(parameter_name: str, raw_expression: str) -> SearchSpec:
    parameter = _search_parameter(parameter_name)
    if ":" not in raw_expression:
        raise click.ClickException(
            f"Categorical Optuna search for {parameter_name!r} must be categorical:value1,value2."
        )
    raw_values = raw_expression.split(":", maxsplit=1)[1]
    values = tuple(
        _coerce_choice(raw_choice.strip(), parameter, parameter_name)
        for raw_choice in raw_values.replace("|", ",").split(",")
        if raw_choice.strip()
    )
    if not values:
        raise click.ClickException(
            f"Categorical Optuna search for {parameter_name!r} must include at least one value."
        )
    if parameter.choices:
        unknown = [value for value in values if value not in parameter.choices]
        if unknown:
            choices = ", ".join(str(choice) for choice in parameter.choices)
            raise click.ClickException(
                f"Unsupported categorical value for {parameter_name!r}: "
                f"{unknown[0]!r}. Choices: {choices}."
            )
    return SearchSpec(parameter_name, "categorical", choices=values)


def _search_parameter(parameter_name: str) -> SearchParameter:
    return SEARCH_PARAMETERS[parameter_name]


def _validate_bounds(
    parameter_name: str,
    low: float | int,
    high: float | int,
    parameter: SearchParameter,
    *,
    log: bool,
) -> None:
    if high < low:
        raise click.ClickException(
            f"Optuna search high bound for {parameter_name!r} must be >= low bound."
        )
    if parameter.minimum is not None and low < parameter.minimum:
        raise click.ClickException(
            f"Optuna search low bound for {parameter_name!r} must be >= {parameter.minimum}."
        )
    if parameter.maximum is not None and high > parameter.maximum:
        raise click.ClickException(
            f"Optuna search high bound for {parameter_name!r} must be <= {parameter.maximum}."
        )
    if log and low <= 0:
        raise click.ClickException(
            f"Log Optuna search for {parameter_name!r} requires a low bound greater than 0."
        )


def _coerce_choice(
    raw_choice: str,
    parameter: SearchParameter,
    parameter_name: str,
) -> object:
    if parameter.value_type is str:
        return raw_choice
    if parameter.value_type is int:
        return _parse_int(raw_choice, parameter_name)
    if parameter.value_type is float:
        return _parse_float(raw_choice, parameter_name)
    return raw_choice


def _parse_int(raw_value: str, parameter_name: str) -> int:
    try:
        return int(raw_value)
    except ValueError as error:
        raise click.ClickException(
            f"Optuna search value for {parameter_name!r} must be an integer: {raw_value!r}."
        ) from error


def _parse_float(raw_value: str, parameter_name: str) -> float:
    try:
        return float(raw_value)
    except ValueError as error:
        raise click.ClickException(
            f"Optuna search value for {parameter_name!r} must be a float: {raw_value!r}."
        ) from error


def _artifact_local_copy(artifact: object) -> Path | None:
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        return None
    local_copy = getter()
    if local_copy is None:
        return None
    return Path(local_copy)


def _load_json_mapping(path: Path) -> Mapping[str, object] | None:
    if not path.exists() or path.is_dir():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _describe_search_spec(spec: SearchSpec) -> str:
    if spec.distribution == "categorical":
        values = ",".join(str(value) for value in spec.choices)
        return f"{spec.parameter_name}=categorical:{values}"
    if spec.distribution == "int":
        modifiers = []
        if spec.step not in (None, 1):
            modifiers.append(f"step={spec.step}")
        if spec.log:
            modifiers.append("log")
        suffix = ":" + ":".join(modifiers) if modifiers else ""
        return f"{spec.parameter_name}=int:{spec.low}:{spec.high}{suffix}"
    suffix = ":log" if spec.log else ""
    return f"{spec.parameter_name}=float:{spec.low}:{spec.high}{suffix}"
