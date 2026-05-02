"""Generic Click CLI for evaluating registered language models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.cli.config import configured_command, load_defaults_from_sections
from src.cli.data_splits import (
    build_cli_split_plan,
    explicit_parameter,
    resolve_from_plan,
    split_plan_parameter_sections,
    upload_split_plan_artifact,
)
from src.cli.output import stage_title
from src.cli.pipeline_common import (
    DEFAULT_PIPELINE_NAME,
    EVALUATION_STAGE,
    pipeline_options,
    pipeline_resume_option,
    resume_pipeline_controller_stage,
)
from src.corpora.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    PROJECT_PARTITIONS,
    VALIDATION_PARTITION,
    load_partition_texts,
    partitioned_metric_names,
    read_model_split_plan,
    source_split_label,
    split_ratio_label,
)
from src.corpora.registry import (
    DEFAULT_CORPUS_NAME,
    corpus_names,
    get_corpus,
    split_note_for,
)
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import (
    clearml_options,
    clearml_settings,
    download_task_artifact,
    start_clearml_run,
)


def load_evaluate_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = load_defaults_from_sections(("train",))
    if "model_name" in train_defaults:
        defaults["model_name"] = train_defaults["model_name"]
    defaults.update(load_defaults_from_sections(("evaluate",)))
    return defaults


@configured_command(
    "evaluate",
    default_loader=load_evaluate_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Evaluate a registered language model on a registered corpus.",
)
@pipeline_resume_option
@pipeline_options(default_name=DEFAULT_PIPELINE_NAME, default_local=False, default_wait=False)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_names()),
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Registered model to evaluate.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus to evaluate on.",
)
@click.option("--dataset-id", default=None, help="Override the registered Hugging Face dataset ID.")
@click.option(
    "--source-split",
    "--split",
    "source_split",
    default=None,
    help=(
        "Restrict the source dataset to one named split before project "
        "train/validation partitioning. Omit to merge all source splits."
    ),
)
@click.option("--text-column", default=None, help="Override the registered text column.")
@click.option(
    "--streaming",
    is_flag=True,
    help="Stream rows instead of downloading the full dataset first.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=0),
    default=None,
    help="Evaluate on only the first N rows. Useful for smoke tests.",
)
@click.option(
    "--train-ratio",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
    default=DEFAULT_TRAIN_RATIO,
    show_default=True,
    help="Fraction of merged source rows assigned to the reusable training partition.",
)
@click.option(
    "--split-seed",
    type=int,
    default=DEFAULT_SPLIT_SEED,
    show_default=True,
    help="Seed for the reusable deterministic train/validation partition.",
)
@click.option(
    "--evaluation-partition",
    "--evaluation-split",
    type=click.Choice(PROJECT_PARTITIONS),
    default=VALIDATION_PARTITION,
    show_default=True,
    help="Reusable project partition to evaluate.",
)
@click.option(
    "--model-task-id",
    default=None,
    help="Deprecated. Evaluation resumes the model dependency from the pipeline controller.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Deprecated. Evaluation resumes the model dependency from the pipeline controller.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="K value for top-k next-token accuracy.",
)
@clearml_options
@click.pass_context
def main(
    ctx: click.Context,
    pipeline_name: str,
    pipeline_version: str,
    pipeline_local: bool,
    controller_queue: str,
    execution_queue: str | None,
    wait: bool,
    add_run_number: bool,
    pipeline_controller_id: str | None,
    model_name: str,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    model_task_id: str | None,
    model_path: Path | None,
    top_k: int,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = get_corpus(corpus)
    model_definition = get_model(model_name)
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")

    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_train_ratio = train_ratio
    resolved_split_seed = split_seed
    resolved_text_column = text_column or corpus_definition.text_column
    if model_task_id is not None or model_path is not None:
        raise click.ClickException(
            "Evaluation now resumes the canonical ClearML pipeline DAG. "
            "Run train first in the same pipeline instead of passing --model-task-id or --model-path."
        )
    if pipeline_local:
        raise click.ClickException(
            "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
            "Use --pipeline-queued for stage CLIs."
        )
    resume_pipeline_controller_stage(
        stage_name=EVALUATION_STAGE,
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
        parameter_filters={
            "model": model_definition.name,
            "corpus": corpus,
            "dataset_id": resolved_dataset_id,
            "source_split": resolved_source_split or "",
            "evaluation_partition": evaluation_partition,
        },
    )
    return

    click.echo(stage_title(1, 1, "Evaluation"), color=True)
    task_id: str | None = None
    task_url: str | None = None
    with (
        TemporaryDirectory(prefix="lme-evaluate-") as staging_root,
        start_clearml_run(
            clearml_settings(
                project_name=clearml_project,
                task_name=clearml_task_name,
                config_file=clearml_config_file,
                connectivity_check=clearml_connectivity_check,
                output_uri=clearml_output_uri,
                tags=clearml_tags,
            ),
            default_task_name=f"evaluate {model_definition.name} {corpus}",
            task_type="testing",
        ) as clearml_run,
    ):
        staged_model_path = stage_model_artifacts(
            model_task_id=model_task_id,
            model_path=model_path,
            staging_dir=Path(staging_root),
        )
        inherited_plan = read_model_split_plan(staged_model_path)
        if inherited_plan is not None and not any(
            explicit_parameter(ctx, parameter)
            for parameter in ("dataset_id", "source_split", "train_ratio", "split_seed")
        ):
            split_plan = inherited_plan
            resolved_dataset_id = split_plan.dataset_id
            resolved_source_split = split_plan.source_split
            resolved_train_ratio = split_plan.train_ratio
            resolved_split_seed = split_plan.split_seed
        else:
            resolved_dataset_id = resolve_from_plan(
                ctx,
                parameter_name="dataset_id",
                value=resolved_dataset_id,
                inherited_plan=inherited_plan,
                inherited_attribute="dataset_id",
            )
            resolved_source_split = resolve_from_plan(
                ctx,
                parameter_name="source_split",
                value=resolved_source_split,
                inherited_plan=inherited_plan,
                inherited_attribute="source_split",
            )
            resolved_train_ratio = resolve_from_plan(
                ctx,
                parameter_name="train_ratio",
                value=resolved_train_ratio,
                inherited_plan=inherited_plan,
                inherited_attribute="train_ratio",
            )
            resolved_split_seed = resolve_from_plan(
                ctx,
                parameter_name="split_seed",
                value=resolved_split_seed,
                inherited_plan=inherited_plan,
                inherited_attribute="split_seed",
            )
            split_plan = build_cli_split_plan(
                corpus_definition,
                corpus=corpus,
                dataset_id=resolved_dataset_id,
                source_split=resolved_source_split,
                train_ratio=resolved_train_ratio,
                split_seed=resolved_split_seed,
            )
        model_options = {
            "corpus": corpus,
            "model_path": staged_model_path,
            "top_k": top_k,
        }
        if model_definition.validate_evaluation_options is not None:
            model_definition.validate_evaluation_options(model_options)

        task_id = clearml_run.task_id
        task_url = clearml_run.task_url
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.evaluate",
                    "artifact_store": "clearml",
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": resolved_dataset_id,
                    "source_split": source_split_label(resolved_source_split),
                    "evaluation_partition": evaluation_partition,
                    "text_column": resolved_text_column,
                    "streaming": streaming,
                    "limit": limit,
                },
                "Model": {
                    "model": model_definition.name,
                    "model_task_id": model_task_id,
                },
                "Evaluation": {
                    "top_k": top_k,
                },
                **split_plan_parameter_sections(split_plan),
                "Artifacts": {
                    "model_artifact": "trained-model-json",
                    "tokenizer_artifact": "input-tokenizer-model",
                    "model_artifact_file": staged_model_path.name,
                },
            }
        )

        texts = load_partition_texts(
            corpus_definition,
            dataset_id=resolved_dataset_id,
            plan=split_plan,
            partition=evaluation_partition,
            streaming=streaming,
            text_column=resolved_text_column,
            limit=limit,
        )

        summary = model_definition.evaluate(texts, model_options)

        clearml_run.log_metrics(
            "Evaluation",
            evaluation_metrics_for_partition(summary, partition=evaluation_partition),
        )
        upload_split_plan_artifact(
            clearml_run,
            staging_dir=Path(staging_root),
            plan=split_plan,
            metadata={"model": model_definition.name, "corpus": corpus, "stage": "evaluation"},
        )
        clearml_run.upload_artifact(
            "evaluation-summary",
            {
                **evaluation_payload(summary),
                "evaluation_partition": evaluation_partition,
                "data_split": split_plan.to_payload(),
            },
            metadata={
                "model": model_definition.name,
                "corpus": corpus,
                "evaluation_partition": evaluation_partition,
                "split_id": split_plan.split_id,
            },
        )
        clearml_run.upload_artifact(
            "evaluated-model",
            summary.model_path,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "tokenizer-model",
            summary.tokenizer_model,
            metadata={"model": model_definition.name, "corpus": corpus},
        )

    click.echo(f"Model: {model_definition.name}")
    click.echo(f"Corpus: {corpus}")
    click.echo(f"Dataset: {resolved_dataset_id}")
    click.echo(f"Source split: {source_split_label(resolved_source_split)}")
    click.echo(f"Evaluation partition: {evaluation_partition}")
    click.echo(f"Split ratio train/validation: {split_ratio_label(split_plan)}")
    click.echo(f"Split seed: {split_plan.split_seed}")
    click.echo(f"Split ID: {split_plan.split_id}")
    split_note = split_note_for(
        corpus_definition,
        dataset_id_override=dataset_id,
    )
    if split_note is not None:
        click.echo(f"Split note: {split_note}")
    click.echo(f"Text column: {resolved_text_column}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    for label, value in model_definition.evaluation_items(summary):
        click.echo(f"{label}: {value}")
    click.echo(f"ClearML task ID: {task_id}")
    if task_url is not None:
        click.echo(f"ClearML task URL: {task_url}")
    click.echo("Data split artifact: data-split-plan-json")
    click.echo("Evaluation artifact: evaluation-summary")


def stage_model_artifacts(
    *,
    model_task_id: str | None,
    model_path: Path | None,
    staging_dir: Path,
) -> Path:
    validate_model_source(model_task_id=model_task_id, model_path=model_path)

    if model_task_id is not None:
        staged_model_path = download_task_artifact(
            task_id=model_task_id,
            artifact_name="trained-model-json",
            destination_dir=staging_dir,
        )
        download_task_artifact(
            task_id=model_task_id,
            artifact_name="input-tokenizer-model",
            destination_dir=staging_dir,
            filename=stored_tokenizer_filename(staged_model_path),
        )
        return staged_model_path

    return model_path


def validate_model_source(
    *,
    model_task_id: str | None,
    model_path: Path | None,
) -> None:
    if model_task_id is not None and model_path is not None:
        raise click.ClickException("Pass either --model-task-id or --model-path, not both.")

    if model_task_id is None and model_path is None:
        raise click.ClickException(
            "Evaluation now uses ClearML as the artifact store. Pass --model-task-id "
            "from src.cli.train, or pass --model-path to evaluate a local model file."
        )


def evaluation_metrics(summary: object) -> dict[str, object]:
    return {
        "sequence_count": getattr(summary, "sequence_count", None),
        "token_count": getattr(summary, "token_count", None),
        "transition_count": getattr(summary, "transition_count", None),
        "correct_next_token_count": getattr(summary, "correct_next_token_count", None),
        "top_k_correct_next_token_count": getattr(
            summary,
            "top_k_correct_next_token_count",
            None,
        ),
        "next_token_accuracy": getattr(summary, "next_token_accuracy", None),
        "top_k_accuracy": getattr(summary, "top_k_accuracy", None),
        "average_negative_log_likelihood": getattr(
            summary,
            "average_negative_log_likelihood",
            None,
        ),
        "cross_entropy_bits": getattr(summary, "cross_entropy_bits", None),
        "perplexity": getattr(summary, "perplexity", None),
        "zero_probability_count": getattr(summary, "zero_probability_count", None),
        "top_k": getattr(summary, "top_k", None),
        "discount": getattr(summary, "discount", None),
        "unigram_weight": getattr(summary, "unigram_weight", None),
        "bigram_weight": getattr(summary, "bigram_weight", None),
        "trigram_weight": getattr(summary, "trigram_weight", None),
    }


def evaluation_metrics_for_partition(
    summary: object,
    *,
    partition: str,
) -> dict[str, object]:
    return partitioned_metric_names(
        evaluation_metrics(summary),
        partition=partition,
    )


def evaluation_payload(summary: object) -> dict[str, object]:
    return {
        "model_artifact_file": artifact_file(getattr(summary, "model_path", None)),
        "tokenizer_artifact_file": artifact_file(getattr(summary, "tokenizer_model", None)),
        "text_normalization": getattr(summary, "text_normalization", None),
        **evaluation_metrics(summary),
    }


def artifact_file(path: object) -> str | None:
    if path is None:
        return None
    return Path(path).name


def stored_tokenizer_filename(model_path: Path) -> str | None:
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None

    tokenizer_model = payload.get("tokenizer_model")
    if tokenizer_model is None:
        return None
    return Path(str(tokenizer_model)).name


if __name__ == "__main__":
    main()
