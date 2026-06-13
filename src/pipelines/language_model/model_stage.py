"""Model-training ClearML function step."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import click

from src.corpora import registry as corpora_registry
from src.corpora import splits as corpus_splits
from src.ml_core.cli import staging
from src.ml_core.data import split_artifacts
from src.ml_core.data import splits as data_splits
from src.ml_core.models import definition as model_def
from src.models.core import registry as model_registry
from src.models.core import model_runtime
from src.pipelines.language_model import artifacts as lm_artifacts
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import stage_runtime


def train_model_pipeline_step(
    *,
    tokenizer_task_id: str,
    tokenizer_model_name: str | None = None,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    text_normalization: str,
    model_hyperparameters: Mapping[str, object],
    clearml_output_uri: str | None = None,
    clearml_tags: stage_runtime.ClearmlTags = None,
    clearml_config_file: str | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_total: int | None = None,
    pipeline_stage_title: str | None = None,
) -> str:
    """Train the language model from the tokenizer step artifact."""

    stage = lm_def.MODEL_STAGE
    corpus_definition = corpora_registry.get_corpus(corpus)
    model = model_registry.get_model(model_name)

    with staging.temporary_staging_directory(prefix="lme-pipeline-model-") as staging_dir:
        clearml_run = stage_runtime.start_step(
            stage_runtime.StepRuntimeCfg(
                stage=stage,
                clearml_output_uri=clearml_output_uri,
                clearml_tags=clearml_tags,
                clearml_config_file=clearml_config_file,
                pipeline_stage_index=pipeline_stage_index,
                pipeline_stage_total=pipeline_stage_total,
                pipeline_stage_title=pipeline_stage_title,
            )
        )
        staged_tokenizer_model = lm_artifacts.stage_tokenizer_model(
            tokenizer_task_id=tokenizer_task_id,
            tokenizer_model_name=tokenizer_model_name,
            staging_dir=staging_dir,
            clearml_run=clearml_run,
        )
        inherited_plan = split_artifacts.inherited_split_plan_from_task(
            task_id=tokenizer_task_id,
            staging_dir=staging_dir,
        )
        if inherited_plan is not None:
            dataset_id = inherited_plan.dataset_id
            source_split = inherited_plan.source_split
            train_ratio = inherited_plan.train_ratio
            split_seed = inherited_plan.split_seed

        split_plan = split_artifacts.build_cli_split_plan(
            corpus_definition,
            corpus=corpus,
            dataset_id=dataset_id,
            source_split=source_split,
            train_ratio=train_ratio,
            split_seed=split_seed,
        )
        output_model_name = lm_def.model_output_name(
            tokenizer_model_name=tokenizer_model_name,
            model_name=model.name,
        ) or f"{corpus}-{model.name}"
        output_path = staging_dir / f"{output_model_name}.json"
        resolved_model_hyperparameters = lm_model_options.model_hyperparameters_from(
            model_hyperparameters
        )
        model_options = {
            "corpus": corpus,
            "tokenizer_model": staged_tokenizer_model,
            "output": output_path,
            "stored_tokenizer_model": Path(staged_tokenizer_model.name),
            **resolved_model_hyperparameters,
            "text_normalization": text_normalization,
        }
        try:
            model_runtime.validate_fit_options(model, model_options)
        except model_def.ModelOptionError as error:
            raise click.ClickException(str(error)) from error

        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.model_training",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                    "tokenizer_task_id": tokenizer_task_id,
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": dataset_id,
                    "dataset_revision": split_plan.dataset_revision or "",
                    "source_split": data_splits.source_split_label(source_split),
                    "training_partition": data_splits.TRAIN_PARTITION,
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Model": {
                    "model": model.name,
                    **resolved_model_hyperparameters,
                },
                "Tokenizer": {
                    "tokenizer_task_id": tokenizer_task_id,
                    "tokenizer_model_name": tokenizer_model_name,
                    "tokenizer_model_file": staged_tokenizer_model.name,
                },
                **split_artifacts.split_plan_parameter_sections(split_plan),
                "Outputs": {
                    "model_file": output_path.name,
                },
            }
        )

        train_texts = corpus_splits.load_partition_texts(
            corpus_definition,
            dataset_id=dataset_id,
            plan=split_plan,
            partition=data_splits.TRAIN_PARTITION,
            streaming=streaming,
            text_column=text_column,
            limit=limit,
        )
        validation_items = None
        if model.uses_validation_tokens:
            validation_items = corpus_splits.load_partition_texts(
                corpus_definition,
                dataset_id=dataset_id,
                plan=split_plan,
                partition=data_splits.VALIDATION_PARTITION,
                streaming=streaming,
                text_column=text_column,
                limit=limit,
            )
        fit_data = model_def.ModelFitData(
            train_items=train_texts,
            validation_items=validation_items,
        )
        summary = model_runtime.fit(model, fit_data, model_options)
        data_splits.attach_split_plan_to_json_model(summary.output_path, split_plan)

        clearml_run.log_metrics(
            "Model training",
            lm_artifacts.training_summary_metrics(summary),
        )
        split_artifacts.upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"model": model.name, "corpus": corpus, "stage": stage},
        )
        clearml_run.register_model(
            name=output_model_name,
            model_path=summary.output_path,
            framework="custom",
            tags=("language-model", model.name, corpus),
            comment="Token n-gram language model JSON.",
        )
        return stage_runtime.require_task_id(clearml_run)
