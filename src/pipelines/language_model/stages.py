"""Importable ClearML pipeline stage functions for language-model experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import click

from src.corpora import registry as corpora_registry
from src.corpora import splits as corpus_splits
from src.corpora import text as corpus_text
from src.ml_core.cli import output as cli_out
from src.ml_core.cli import staging
from src.ml_core.data import split_artifacts
from src.ml_core.data import splits as data_splits
from src.ml_core.models import definition as model_def
from src.models.core import registry as model_registry
from src.pipelines.language_model import artifacts as lm_artifacts
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import stage_runtime
from src.tokenizers import registry as tokenizer_registry


def train_tokenizer_step(
    *,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    vocab_size: int,
    artifact_name: str,
    tokenizer_algo: str,
    sentencepiece_model_type: str,
    sentencepiece_character_coverage: float,
    sentencepiece_hard_vocab_limit: bool,
    sentencepiece_max_sentence_length: int | None,
    text_normalization: str,
    clearml_output_uri: str | None = None,
    clearml_tags: str | list[str] | tuple[str, ...] | None = None,
    clearml_config_file: str | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_total: int | None = None,
    pipeline_stage_title: str | None = None,
) -> str:
    """Train and publish the tokenizer step artifacts."""
    stage = lm_def.TOKENIZER_STAGE
    stage_runtime.configure_step_clearml(clearml_config_file)
    stage_runtime.emit_pipeline_stage_title(
        stage,
        index=pipeline_stage_index,
        total=pipeline_stage_total,
        title=pipeline_stage_title,
    )
    corpus_definition = corpora_registry.get_corpus(corpus)
    split_plan = split_artifacts.build_cli_split_plan(
        corpus_definition,
        corpus=corpus,
        dataset_id=dataset_id,
        source_split=source_split,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )

    with staging.temporary_staging_directory(
        prefix="lme-pipeline-tokenizer-"
    ) as staging_dir:
        output_prefix = staging_dir / artifact_name
        tokenizer_options = tokenizer_registry.tokenizer_options(
            tokenizer_algo=tokenizer_algo,
            sentencepiece_model_type=sentencepiece_model_type,
            sentencepiece_character_coverage=sentencepiece_character_coverage,
            sentencepiece_hard_vocab_limit=sentencepiece_hard_vocab_limit,
            sentencepiece_max_sentence_length=sentencepiece_max_sentence_length,
        )
        clearml_run = stage_runtime.current_step_run(
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            stage=stage,
        )
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.tokenizer_training",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": dataset_id,
                    "source_split": data_splits.source_split_label(source_split),
                    "training_partition": data_splits.TRAIN_PARTITION,
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Tokenizer": {
                    "tokenizer_algo": tokenizer_algo,
                    "vocab_size": vocab_size,
                    "artifact_name": artifact_name,
                },
                "Tokenizer Options": {
                    **tokenizer_options,
                },
                **split_artifacts.split_plan_parameter_sections(split_plan),
            }
        )

        texts = corpus_splits.load_partition_texts(
            corpus_definition,
            dataset_id=dataset_id,
            plan=split_plan,
            partition=data_splits.TRAIN_PARTITION,
            streaming=streaming,
            text_column=text_column,
            limit=limit,
        )
        tokenizer_output = tokenizer_registry.train_tokenizer(
            texts,
            tokenizer_algo=tokenizer_algo,
            output_prefix=output_prefix,
            vocab_size=vocab_size,
            tokenizer_options=tokenizer_options,
            text_normalization=text_normalization,
        )

        clearml_run.log_metrics(
            "Tokenizer training",
            {
                "vocab_size": vocab_size,
                "limit": limit,
            },
        )
        split_artifacts.upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"corpus": corpus, "stage": stage},
        )
        clearml_run.upload_artifact(
            tokenizer_registry.TOKENIZER_VOCAB_ARTIFACT,
            tokenizer_output.vocab_path,
            metadata={
                "corpus": corpus,
                "tokenizer_algo": tokenizer_algo,
                "vocab_size": vocab_size,
            },
        )
        clearml_run.register_model(
            name=artifact_name,
            model_path=tokenizer_output.model_path,
            framework="custom",
            tags=("tokenizer", corpus),
            comment=f"{tokenizer_algo} tokenizer model.",
        )
        return stage_runtime.require_task_id(clearml_run)


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
    clearml_tags: str | list[str] | tuple[str, ...] | None = None,
    clearml_config_file: str | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_total: int | None = None,
    pipeline_stage_title: str | None = None,
) -> str:
    """Train the language model from the tokenizer step artifact."""
    stage = lm_def.MODEL_STAGE
    stage_runtime.configure_step_clearml(clearml_config_file)
    stage_runtime.emit_pipeline_stage_title(
        stage,
        index=pipeline_stage_index,
        total=pipeline_stage_total,
        title=pipeline_stage_title,
    )
    corpus_definition = corpora_registry.get_corpus(corpus)
    model_definition = model_registry.get_model(model_name)

    with staging.temporary_staging_directory(
        prefix="lme-pipeline-model-"
    ) as staging_dir:
        clearml_run = stage_runtime.current_step_run(
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            stage=stage,
        )
        staged_tokenizer_model = lm_artifacts.stage_tokenizer_model(
            tokenizer_task_id=tokenizer_task_id,
            tokenizer_model_name=tokenizer_model_name,
            tokenizer_model=None,
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
            model_name=model_definition.name,
        ) or f"{corpus}-{model_definition.name}"
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
            model_definition.validate_options(model_options)
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
                    "source_split": data_splits.source_split_label(source_split),
                    "training_partition": data_splits.TRAIN_PARTITION,
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Model": {
                    "model": model_definition.name,
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

        texts = corpus_splits.load_partition_texts(
            corpus_definition,
            dataset_id=dataset_id,
            plan=split_plan,
            partition=data_splits.TRAIN_PARTITION,
            streaming=streaming,
            text_column=text_column,
            limit=limit,
        )
        summary = model_definition.train(texts, model_options)
        data_splits.attach_split_plan_to_json_model(summary.output_path, split_plan)

        clearml_run.log_metrics(
            "Model training",
            lm_artifacts.training_summary_metrics(summary),
        )
        split_artifacts.upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"model": model_definition.name, "corpus": corpus, "stage": stage},
        )
        clearml_run.register_model(
            name=output_model_name,
            model_path=summary.output_path,
            framework="custom",
            tags=("language-model", model_definition.name, corpus),
            comment="Token n-gram language model JSON.",
        )
        return stage_runtime.require_task_id(clearml_run)


def evaluate_pipeline_step(
    *,
    model_task_id: str,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    top_k: int,
    tokenizer_model_name: str | None = None,
    clearml_output_uri: str | None = None,
    clearml_tags: str | list[str] | tuple[str, ...] | None = None,
    clearml_config_file: str | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_total: int | None = None,
    pipeline_stage_title: str | None = None,
) -> str:
    """Evaluate the trained model step artifact."""
    stage = lm_def.EVALUATION_STAGE
    stage_runtime.configure_step_clearml(clearml_config_file)
    stage_runtime.emit_pipeline_stage_title(
        stage,
        index=pipeline_stage_index,
        total=pipeline_stage_total,
        title=pipeline_stage_title,
    )
    corpus_definition = corpora_registry.get_corpus(corpus)
    model_definition = model_registry.get_model(model_name)
    if model_definition.evaluate is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")

    evaluation_partitions = tuple(data_splits.PROJECT_PARTITIONS)
    click.echo(
        f"Evaluation stage started: model={model_definition.name}, corpus={corpus}, "
        f"partitions={', '.join(evaluation_partitions)}, "
        f"primary_partition={evaluation_partition}, top_k={top_k}"
    )
    if limit is not None:
        click.echo(f"Evaluation row limit: first {limit:,} selected rows")

    with staging.temporary_staging_directory(
        prefix="lme-pipeline-evaluate-"
    ) as staging_dir:
        clearml_run = stage_runtime.current_step_run(
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            stage=stage,
        )
        click.echo(f"Staging model files from ClearML task {model_task_id}...")
        staged_model_path = lm_artifacts.stage_model_files(
            model_task_id=model_task_id,
            model_path=None,
            staging_dir=staging_dir,
            clearml_run=clearml_run,
            output_model_name=lm_def.model_output_name(
                tokenizer_model_name=tokenizer_model_name,
                model_name=model_definition.name,
            ),
        )
        click.echo(f"Staged model file: {staged_model_path.name}")
        inherited_plan = data_splits.read_model_split_plan(staged_model_path)
        if inherited_plan is not None:
            dataset_id = inherited_plan.dataset_id
            source_split = inherited_plan.source_split
            train_ratio = inherited_plan.train_ratio
            split_seed = inherited_plan.split_seed
            click.echo(f"Using inherited data split plan: {inherited_plan.split_id}")

        split_plan = split_artifacts.build_cli_split_plan(
            corpus_definition,
            corpus=corpus,
            dataset_id=dataset_id,
            source_split=source_split,
            train_ratio=train_ratio,
            split_seed=split_seed,
        )
        evaluation_options = {
            "corpus": corpus,
            "model_path": staged_model_path,
            "top_k": top_k,
        }
        if model_definition.validate_evaluation_options is not None:
            try:
                model_definition.validate_evaluation_options(evaluation_options)
            except model_def.ModelOptionError as error:
                raise click.ClickException(str(error)) from error
        click.echo(
            f"Evaluation data: dataset={dataset_id}, "
            f"source_split={data_splits.source_split_label(source_split)}, "
            f"text_column={text_column}"
        )

        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.model_training",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                    "model_task_id": model_task_id,
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": dataset_id,
                    "source_split": data_splits.source_split_label(source_split),
                    "evaluation_partition": evaluation_partition,
                    "evaluation_partitions": list(evaluation_partitions),
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                },
                "Model": {
                    "model": model_definition.name,
                    "model_task_id": model_task_id,
                    "model_file": staged_model_path.name,
                },
                "Evaluation": {
                    "top_k": top_k,
                },
                **split_artifacts.split_plan_parameter_sections(split_plan),
            }
        )

        summaries: dict[str, object] = {}
        for partition in evaluation_partitions:
            click.echo(f"Loading dataset rows for {partition} evaluation...")
            dataset = corpus_definition.load(
                dataset_id=dataset_id,
                split=split_plan.source_split,
                streaming=streaming,
            )
            click.echo("Counting selected evaluation rows...")
            total_rows = data_splits.count_partition_rows(
                dataset,
                partition=partition,
                plan=split_plan,
                limit=limit,
            )
            if total_rows is None:
                click.echo(
                    "Evaluation row total is unknown; progress will report processed rows."
                )
            else:
                click.echo(f"Evaluation rows selected: {total_rows:,}")
            rows = data_splits.iter_partition_rows(
                dataset,
                partition=partition,
                plan=split_plan,
            )
            texts = corpus_text.iter_text_column(
                rows,
                text_column=text_column,
                limit=limit,
            )

            click.echo(f"Running {partition} model evaluation...")
            summary = model_definition.evaluate(
                cli_out.iter_with_progress(
                    texts,
                    label=f"Evaluating {partition} rows",
                    total=total_rows,
                    unit="rows",
                ),
                evaluation_options,
            )
            summaries[partition] = summary
            click.echo(
                f"{partition} evaluation complete: {summary.sequence_count:,} sequences, "
                f"{summary.token_count:,} tokens, {summary.transition_count:,} transitions"
            )

            clearml_run.log_metrics(
                "Evaluation",
                lm_artifacts.evaluation_metrics_for_partition(
                    summary,
                    partition=partition,
                ),
            )

        primary_summary = summaries.get(evaluation_partition)
        if primary_summary is None:
            primary_partition = evaluation_partitions[0]
            primary_summary = summaries[primary_partition]
        else:
            primary_partition = evaluation_partition

        click.echo("Uploading evaluation artifacts...")
        split_artifacts.upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"model": model_definition.name, "corpus": corpus, "stage": stage},
        )
        clearml_run.upload_artifact(
            "evaluation-summary",
            {
                **lm_artifacts.evaluation_payload(primary_summary),
                **{
                    metric_name: value
                    for partition, summary in summaries.items()
                    for metric_name, value in lm_artifacts.evaluation_metrics_for_partition(
                        summary,
                        partition=partition,
                    ).items()
                },
                "evaluation_partition": primary_partition,
                "evaluation_partitions": list(evaluation_partitions),
                "evaluation_limit": limit,
                "data_split": split_plan.to_payload(),
                "partitions": {
                    partition: lm_artifacts.evaluation_payload(summary)
                    for partition, summary in summaries.items()
                },
            },
            metadata={
                "model": model_definition.name,
                "corpus": corpus,
                "evaluation_partition": primary_partition,
                "evaluation_partitions": ",".join(evaluation_partitions),
                "split_id": split_plan.split_id,
            },
        )
        click.echo("Evaluation artifacts uploaded.")
        return stage_runtime.require_task_id(clearml_run)
