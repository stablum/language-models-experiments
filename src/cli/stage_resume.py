"""Shared helpers for stage-resume command-line interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import click

from src.cli import corpus_source
from src.ml_core import cfg as core_cfg
from src.ml_core import pipeline as core_pipeline
from src.ml_core.cli import config as cli_config
from src.ml_core.models import definition as model_def
from src.models.core import registry as model_registry
from src.pipelines.language_model import model_training as model_pipeline


class CorpusFilterCfg(core_cfg.FrozenBaseCfg):
    """Corpus cfg used to match an existing model-training controller."""

    corpus: str
    dataset_id: str | None
    source_split: str | None
    text_column: str | None
    streaming: bool
    train_ratio: float
    split_seed: int


class ModelTrainingStageFilterCfg(core_cfg.FrozenBaseCfg):
    """Model-stage cfg shared by train/evaluate stage CLIs."""

    model_name: str
    tokenizer_model_name: str | None
    action: str
    corpus: CorpusFilterCfg
    limit_param: str
    limit: int | None


class StageFilterResolution(core_cfg.BaseCfg):
    """Resolved model object and ClearML Experiment parameter filters."""

    model: model_def.ModelDefinition
    filters: dict[str, object]


def load_stage_command_defaults(stage_section: str) -> dict[str, object]:
    defaults = cli_config.load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = cli_config.load_defaults_from_sections(("train",))
    for key in ("model_name", "tokenizer_model_name"):
        if key in train_defaults:
            defaults[key] = train_defaults[key]
    defaults.update(cli_config.load_defaults_from_sections((stage_section,)))
    return defaults


def require_tokenizer_model_name(tokenizer_model_name: str | None, *, action: str) -> str:
    resolved = str(tokenizer_model_name or "").strip()
    if not resolved:
        raise click.ClickException(
            f"{action} requires --tokenizer-model-name, or tokenizer_model_name in [train]."
        )
    return resolved


def reject_pipeline_local(pipeline_local: bool) -> None:
    if pipeline_local:
        raise click.ClickException(
            "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
            "Use --pipeline-queued for stage CLIs."
        )


def resolve_model_training_stage_filters(
    cfg: ModelTrainingStageFilterCfg,
    *,
    extra_filters: Mapping[str, object] | None = None,
) -> StageFilterResolution:
    source = corpus_source.resolve(
        corpus=cfg.corpus.corpus,
        dataset_id=cfg.corpus.dataset_id,
        source_split=cfg.corpus.source_split,
        text_column=cfg.corpus.text_column,
    )
    model = model_registry.get_model(cfg.model_name)
    tokenizer_model_name = require_tokenizer_model_name(
        cfg.tokenizer_model_name,
        action=cfg.action,
    )
    filters = {
        "model": model.name,
        "tokenizer_model_name": tokenizer_model_name,
        "corpus": cfg.corpus.corpus,
        "dataset_id": source.dataset_id,
        "source_split": source.source_split or "",
        "text_column": source.text_column,
        "streaming": cfg.corpus.streaming,
        "train_ratio": cfg.corpus.train_ratio,
        "split_seed": cfg.corpus.split_seed,
        cfg.limit_param: cfg.limit,
    }
    if extra_filters:
        filters.update(extra_filters)
    return StageFilterResolution(
        model=model,
        filters=filters,
    )


def resume_model_training_stage(
    *,
    stage_name: str,
    pipeline_name: str,
    pipeline_version: str,
    controller_queue: str,
    wait: bool,
    pipeline_controller_id: str | None,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    parameter_filters: Mapping[str, object],
) -> None:
    core_pipeline.resume_pipeline_controller_stage(
        stage_name=stage_name,
        pipeline_controller_id=pipeline_controller_id,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        controller_queue=controller_queue,
        wait=wait,
        clearml_project=clearml_project,
        clearml_task_name=clearml_task_name,
        clearml_config_file=clearml_config_file,
        clearml_connectivity_check=clearml_connectivity_check,
        clearml_output_uri=clearml_output_uri,
        clearml_tags=clearml_tags,
        parameter_filters=parameter_filters,
        stage_dependencies=(
            model_pipeline.MODEL_TRAINING_PIPELINE.stage_dependencies
        ),
        stage_names=model_pipeline.MODEL_TRAINING_PIPELINE.stages,
    )
