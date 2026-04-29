"""CLI helpers for reusable project data splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from src.corpora.registry import CorpusDefinition
from src.corpora.splits import (
    DataSplitPlan,
    SPLIT_PLAN_ARTIFACT,
    build_data_split_plan,
    read_split_plan,
    split_plan_clearml_parameters,
    write_split_plan,
)
from src.tracking.clearml import maybe_download_task_artifact


EXPLICIT_PARAMETER_SOURCES = (
    click.core.ParameterSource.COMMANDLINE,
    click.core.ParameterSource.ENVIRONMENT,
)


def explicit_parameter(ctx: click.Context, parameter_name: str) -> bool:
    return ctx.get_parameter_source(parameter_name) in EXPLICIT_PARAMETER_SOURCES


def resolve_from_plan(
    ctx: click.Context,
    *,
    parameter_name: str,
    value: Any,
    inherited_plan: DataSplitPlan | None,
    inherited_attribute: str,
) -> Any:
    if explicit_parameter(ctx, parameter_name) or inherited_plan is None:
        return value
    return getattr(inherited_plan, inherited_attribute)


def build_cli_split_plan(
    corpus_definition: CorpusDefinition,
    *,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    train_ratio: float,
    split_seed: int,
) -> DataSplitPlan:
    source_splits = (
        (source_split,)
        if source_split is not None
        else corpus_definition.available_splits
    )
    return build_data_split_plan(
        corpus=corpus,
        dataset_id=dataset_id,
        source_split=source_split,
        source_splits=source_splits,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )


def write_split_plan_artifact(staging_dir: Path, plan: DataSplitPlan) -> Path:
    return write_split_plan(staging_dir / "data-split-plan.json", plan)


def upload_split_plan_artifact(
    clearml_run: Any,
    *,
    staging_dir: Path,
    plan: DataSplitPlan,
    metadata: dict[str, object] | None = None,
) -> Path:
    path = write_split_plan_artifact(staging_dir, plan)
    clearml_run.upload_artifact(
        SPLIT_PLAN_ARTIFACT,
        path,
        metadata={
            **(metadata or {}),
            "split_id": plan.split_id,
            "split_method": plan.split_method,
        },
    )
    return path


def inherited_split_plan_from_task(
    *,
    task_id: str | None,
    staging_dir: Path,
) -> DataSplitPlan | None:
    if task_id is None:
        return None

    path = maybe_download_task_artifact(
        task_id=task_id,
        artifact_name=SPLIT_PLAN_ARTIFACT,
        destination_dir=staging_dir,
    )
    if path is None:
        return None
    return read_split_plan(path)


def split_plan_parameter_sections(plan: DataSplitPlan) -> dict[str, dict[str, object]]:
    return {
        "Data split": split_plan_clearml_parameters(plan),
    }
