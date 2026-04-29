"""Generic Click CLI for training registered language models."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.cli.config import configured_command
from src.cli.data_splits import (
    build_cli_split_plan,
    inherited_split_plan_from_task,
    resolve_from_plan,
    split_plan_parameter_sections,
    upload_split_plan_artifact,
)
from src.cli.output import stage_title
from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    TRAIN_PARTITION,
    attach_split_plan_to_json_model,
    load_partition_texts,
    source_split_label,
    split_ratio_label,
)
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import (
    clearml_options,
    clearml_settings,
    download_task_artifact,
    start_clearml_run,
)


@configured_command(
    "train",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Train a registered language model from a registered corpus.",
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_names()),
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Registered model to train.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus to train on.",
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
    help="Train on only the first N rows. Useful for smoke tests.",
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
    "--tokenizer-task-id",
    default=None,
    help=(
        "ClearML task ID produced by src.cli.train_sentencepiece. "
        "Downloads its sentencepiece-model artifact for training."
    ),
)
@click.option(
    "--tokenizer-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Local SentencePiece .model file to stage into ClearML-backed training.",
)
@click.option(
    "--smoothing",
    type=click.FloatRange(min=0.0),
    default=0.1,
    show_default=True,
    help="Add-k smoothing value for models that use it.",
)
@click.option(
    "--unigram-weight",
    type=click.FloatRange(min=0.0),
    default=0.1,
    show_default=True,
    help="Interpolation weight for unigram probabilities in models that use it.",
)
@click.option(
    "--bigram-weight",
    type=click.FloatRange(min=0.0),
    default=0.3,
    show_default=True,
    help="Interpolation weight for bigram probabilities in models that use it.",
)
@click.option(
    "--trigram-weight",
    type=click.FloatRange(min=0.0),
    default=0.6,
    show_default=True,
    help="Interpolation weight for trigram probabilities in models that use it.",
)
@click.option(
    "--discount",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.75,
    show_default=True,
    help="Absolute discount value for models that use it.",
)
@click.option(
    "--text-normalization",
    type=click.Choice(TEXT_NORMALIZATION_MODES),
    default=DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before model training.",
)
@clearml_options
@click.pass_context
def main(
    ctx: click.Context,
    model_name: str,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    tokenizer_task_id: str | None,
    tokenizer_model: Path | None,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    discount: float,
    text_normalization: str,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = get_corpus(corpus)
    model_definition = get_model(model_name)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_train_ratio = train_ratio
    resolved_split_seed = split_seed
    resolved_text_column = text_column or corpus_definition.text_column
    validate_tokenizer_source(
        tokenizer_task_id=tokenizer_task_id,
        tokenizer_model=tokenizer_model,
    )

    click.echo(stage_title(1, 1, "Model training"), color=True)
    task_id: str | None = None
    task_url: str | None = None
    with (
        TemporaryDirectory(prefix="lme-model-") as staging_root,
        start_clearml_run(
            clearml_settings(
                project_name=clearml_project,
                task_name=clearml_task_name,
                config_file=clearml_config_file,
                connectivity_check=clearml_connectivity_check,
                output_uri=clearml_output_uri,
                tags=clearml_tags,
            ),
            default_task_name=f"train {model_definition.name} {corpus}",
            task_type="training",
        ) as clearml_run,
    ):
        staging_dir = Path(staging_root)
        staged_tokenizer_model = stage_tokenizer_model(
            tokenizer_task_id=tokenizer_task_id,
            tokenizer_model=tokenizer_model,
            staging_dir=staging_dir,
        )
        output_path = staging_dir / f"{corpus}-sentencepiece-{model_definition.name}.json"
        inherited_plan = inherited_split_plan_from_task(
            task_id=tokenizer_task_id,
            staging_dir=staging_dir,
        )
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
            "tokenizer_model": staged_tokenizer_model,
            "output": output_path,
            "stored_tokenizer_model": Path(staged_tokenizer_model.name),
            "smoothing": smoothing,
            "unigram_weight": unigram_weight,
            "bigram_weight": bigram_weight,
            "trigram_weight": trigram_weight,
            "discount": discount,
            "text_normalization": text_normalization,
        }
        model_definition.validate_options(model_options)
        task_id = clearml_run.task_id
        task_url = clearml_run.task_url
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.train",
                    "artifact_store": "clearml",
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": resolved_dataset_id,
                    "source_split": source_split_label(resolved_source_split),
                    "training_partition": TRAIN_PARTITION,
                    "text_column": resolved_text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Model": {
                    "model": model_definition.name,
                    "smoothing": smoothing,
                    "unigram_weight": unigram_weight,
                    "bigram_weight": bigram_weight,
                    "trigram_weight": trigram_weight,
                    "discount": discount,
                },
                "Tokenizer": {
                    "tokenizer_task_id": tokenizer_task_id,
                },
                **split_plan_parameter_sections(split_plan),
                "Artifacts": {
                    "tokenizer_artifact": "sentencepiece-model",
                    "tokenizer_artifact_file": staged_tokenizer_model.name,
                    "output_artifact_file": output_path.name,
                },
            }
        )

        texts = load_partition_texts(
            corpus_definition,
            dataset_id=resolved_dataset_id,
            plan=split_plan,
            partition=TRAIN_PARTITION,
            streaming=streaming,
            text_column=resolved_text_column,
            limit=limit,
        )

        summary = model_definition.train(texts, model_options)
        attach_split_plan_to_json_model(summary.output_path, split_plan)

        clearml_run.log_metrics(
            "Model training",
            training_summary_metrics(summary),
        )
        upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"model": model_definition.name, "corpus": corpus, "stage": "model-training"},
        )
        clearml_run.upload_artifact(
            "input-tokenizer-model",
            summary.tokenizer_model,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "trained-model-json",
            summary.output_path,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.register_model(
            name=summary.output_path.stem,
            model_path=summary.output_path,
            framework="custom",
            tags=("language-model", model_definition.name, corpus),
            comment="Token n-gram language model JSON.",
        )

    click.echo(f"Model: {model_definition.name}")
    click.echo(f"Corpus: {corpus}")
    click.echo(f"Dataset: {resolved_dataset_id}")
    click.echo(f"Source split: {source_split_label(resolved_source_split)}")
    click.echo(f"Training partition: {TRAIN_PARTITION}")
    click.echo(f"Split ratio train/validation: {split_ratio_label(split_plan)}")
    click.echo(f"Split seed: {split_plan.split_seed}")
    click.echo(f"Split ID: {split_plan.split_id}")
    click.echo(f"Text column: {resolved_text_column}")
    click.echo(f"Text normalization: {text_normalization}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    for label, value in model_definition.summary_items(summary):
        click.echo(f"{label}: {value}")
    click.echo(f"ClearML task ID: {task_id}")
    if task_url is not None:
        click.echo(f"ClearML task URL: {task_url}")
    click.echo("Data split artifact: data-split-plan-json")
    click.echo("Model artifact: trained-model-json")
    click.echo("Tokenizer artifact: input-tokenizer-model")


def stage_tokenizer_model(
    *,
    tokenizer_task_id: str | None,
    tokenizer_model: Path | None,
    staging_dir: Path,
) -> Path:
    validate_tokenizer_source(
        tokenizer_task_id=tokenizer_task_id,
        tokenizer_model=tokenizer_model,
    )

    if tokenizer_task_id is not None:
        return download_task_artifact(
            task_id=tokenizer_task_id,
            artifact_name="sentencepiece-model",
            destination_dir=staging_dir,
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    destination = staging_dir / tokenizer_model.name
    if tokenizer_model.resolve() != destination.resolve():
        shutil.copy2(tokenizer_model, destination)
    return destination


def validate_tokenizer_source(
    *,
    tokenizer_task_id: str | None,
    tokenizer_model: Path | None,
) -> None:
    if tokenizer_task_id is not None and tokenizer_model is not None:
        raise click.ClickException(
            "Pass either --tokenizer-task-id or --tokenizer-model, not both."
        )

    if tokenizer_task_id is None and tokenizer_model is None:
        raise click.ClickException(
            "Language model training now uses ClearML as the artifact store. "
            "Pass --tokenizer-task-id from src.cli.train_sentencepiece, or pass "
            "--tokenizer-model to stage a local tokenizer file."
        )


def training_summary_metrics(summary: object) -> dict[str, object]:
    return {
        "vocab_size": getattr(summary, "vocab_size", None),
        "sequence_count": getattr(summary, "sequence_count", None),
        "token_count": getattr(summary, "token_count", None),
        "transition_count": getattr(summary, "transition_count", None),
        "unigram_count": getattr(summary, "unigram_count", None),
        "bigram_transition_count": getattr(summary, "bigram_transition_count", None),
        "trigram_transition_count": getattr(summary, "trigram_transition_count", None),
        "continuation_unigram_count": getattr(summary, "continuation_unigram_count", None),
        "continuation_bigram_type_count": getattr(
            summary,
            "continuation_bigram_type_count",
            None,
        ),
        "smoothing": getattr(summary, "smoothing", None),
        "discount": getattr(summary, "discount", None),
        "unigram_weight": getattr(summary, "unigram_weight", None),
        "bigram_weight": getattr(summary, "bigram_weight", None),
        "trigram_weight": getattr(summary, "trigram_weight", None),
    }


if __name__ == "__main__":
    main()
