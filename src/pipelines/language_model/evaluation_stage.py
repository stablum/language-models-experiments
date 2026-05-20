"""Evaluation ClearML function step."""

from __future__ import annotations

import click

from src.corpora import registry as corpora_registry
from src.corpora import text as corpus_text
from src.ml_core.cli import output as cli_out
from src.ml_core.cli import staging
from src.ml_core.data import split_artifacts
from src.ml_core.data import splits as data_splits
from src.ml_core.models import definition as model_def
from src.models.core import registry as model_registry
from src.pipelines.language_model import artifacts as lm_artifacts
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import stage_runtime


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
    clearml_tags: stage_runtime.ClearmlTags = None,
    clearml_config_file: str | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_total: int | None = None,
    pipeline_stage_title: str | None = None,
) -> str:
    """Evaluate the trained model step artifact."""

    stage = lm_def.EVALUATION_STAGE
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
                    "dataset_revision": split_plan.dataset_revision or "",
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
                revision=split_plan.dataset_revision,
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
