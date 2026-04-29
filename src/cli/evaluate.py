"""Generic Click CLI for evaluating registered language models."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.cli.config import configured_command
from src.cli.output import stage_title
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.text import iter_text_column
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import (
    clearml_options,
    clearml_settings,
    download_task_artifact,
    start_clearml_run,
)


@configured_command(
    "evaluate",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Evaluate a registered language model on a registered corpus.",
)
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
@click.option("--split", default=None, help="Override the registered dataset split.")
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
    "--model-task-id",
    default=None,
    help=(
        "ClearML task ID produced by src.cli.train. Downloads its trained-model-json "
        "and input-tokenizer-model artifacts for evaluation."
    ),
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Local trained model file to evaluate.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="K value for top-k next-token accuracy.",
)
@clearml_options
def main(
    model_name: str,
    corpus: str,
    dataset_id: str | None,
    split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
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
    resolved_split = split or corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    validate_model_source(model_task_id=model_task_id, model_path=model_path)

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
        model_options = {
            "corpus": corpus,
            "model_path": staged_model_path,
            "top_k": top_k,
        }
        if model_definition.validate_evaluation_options is not None:
            model_definition.validate_evaluation_options(model_options)

        task_id = clearml_run.task_id
        task_url = clearml_run.task_url
        clearml_run.connect_parameters(
            {
                "command": "src.cli.evaluate",
                "artifact_store": "clearml",
                "model": model_definition.name,
                "corpus": corpus,
                "dataset_id": resolved_dataset_id,
                "split": resolved_split,
                "text_column": resolved_text_column,
                "streaming": streaming,
                "limit": limit,
                "model_task_id": model_task_id,
                "model_artifact": "trained-model-json",
                "tokenizer_artifact": "input-tokenizer-model",
                "model_artifact_file": staged_model_path.name,
                "top_k": top_k,
            }
        )

        dataset = corpus_definition.load(
            dataset_id=resolved_dataset_id,
            split=resolved_split,
            streaming=streaming,
        )
        texts = iter_text_column(
            dataset,
            text_column=resolved_text_column,
            limit=limit,
        )

        summary = model_definition.evaluate(texts, model_options)

        clearml_run.log_metrics("Evaluation", evaluation_metrics(summary))
        clearml_run.upload_artifact(
            "evaluation-summary",
            evaluation_payload(summary),
            metadata={"model": model_definition.name, "corpus": corpus},
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
    click.echo(f"Split: {resolved_split}")
    click.echo(f"Text column: {resolved_text_column}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    for label, value in model_definition.evaluation_items(summary):
        click.echo(f"{label}: {value}")
    click.echo(f"ClearML task ID: {task_id}")
    if task_url is not None:
        click.echo(f"ClearML task URL: {task_url}")
    click.echo("Evaluation artifact: evaluation-summary")


def stage_model_artifacts(
    *,
    model_task_id: str | None,
    model_path: Path | None,
    staging_dir: Path,
) -> Path:
    validate_model_source(model_task_id=model_task_id, model_path=model_path)

    if model_task_id is not None:
        download_task_artifact(
            task_id=model_task_id,
            artifact_name="input-tokenizer-model",
            destination_dir=staging_dir,
        )
        return download_task_artifact(
            task_id=model_task_id,
            artifact_name="trained-model-json",
            destination_dir=staging_dir,
        )

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


if __name__ == "__main__":
    main()
