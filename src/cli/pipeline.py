"""End-to-end ClearML-backed tokenizer, model, evaluation, and query pipeline."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import click

from src.cli.data_splits import (
    build_cli_split_plan,
    split_plan_parameter_sections,
    upload_split_plan_artifact,
)
from src.cli.config import configured_command
from src.cli.output import highlight_stage_title, stage_title
from src.cli.evaluate import (
    evaluation_metrics,
    evaluation_metrics_for_partition,
    evaluation_payload,
)
from src.cli.query import query_metrics, query_payload
from src.cli.train import training_summary_metrics
from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import (
    DEFAULT_CORPUS_NAME,
    corpus_names,
    get_corpus,
    split_note_for,
)
from src.corpora.splits import (
    DataSplitPlan,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    PROJECT_PARTITIONS,
    TRAIN_PARTITION,
    VALIDATION_PARTITION,
    attach_split_plan_to_json_model,
    load_partition_texts,
    source_split_label,
    split_ratio_label,
)
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import clearml_options, clearml_settings, start_clearml_run
from src.tokenizers.sentencepiece_training import train_sentencepiece


@configured_command(
    "pipeline",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run tokenizer training, model training, evaluation, and query in one ClearML task.",
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_names()),
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Registered model to train, evaluate, and query.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus to use.",
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
    help="Apply the same row limit to tokenizer training, model training, and evaluation.",
)
@click.option(
    "--tokenizer-limit",
    type=click.IntRange(min=0),
    default=None,
    help="Train the tokenizer on only the first N rows. Overrides --limit for this stage.",
)
@click.option(
    "--training-limit",
    type=click.IntRange(min=0),
    default=None,
    help="Train the language model on only the first N rows. Overrides --limit for this stage.",
)
@click.option(
    "--evaluation-limit",
    type=click.IntRange(min=0),
    default=None,
    help="Evaluate on only the first N rows. Overrides --limit for this stage.",
)
@click.option(
    "--vocab-size",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="SentencePiece vocabulary size.",
)
@click.option(
    "--artifact-name",
    default=None,
    help="Base name for the tokenizer artifacts stored in ClearML.",
)
@click.option(
    "--model-type",
    type=click.Choice(("unigram", "bpe", "char", "word")),
    default="unigram",
    show_default=True,
    help="SentencePiece model type.",
)
@click.option(
    "--character-coverage",
    type=float,
    default=1.0,
    show_default=True,
    help="Fraction of characters covered by the tokenizer.",
)
@click.option(
    "--hard-vocab-limit/--no-hard-vocab-limit",
    default=True,
    show_default=True,
    help="Require SentencePiece to produce exactly vocab-size pieces.",
)
@click.option(
    "--max-sentence-length",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum sentence length passed to SentencePiece.",
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
    "--top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="K value for top-k next-token accuracy.",
)
@click.option(
    "--query-prompt",
    default="Once upon",
    show_default=True,
    help="Text prefix for the final query stage.",
)
@click.option(
    "--query-max-tokens",
    type=click.IntRange(min=0),
    default=80,
    show_default=True,
    help="Maximum number of new tokens to generate in the final query stage.",
)
@click.option(
    "--query-top-k",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of likely next tokens to store for the query prompt.",
)
@click.option(
    "--query-decoding",
    type=click.Choice(("sample", "most-probable")),
    default="sample",
    show_default=True,
    help="Generate the final query by sampling or by choosing the most probable next token.",
)
@click.option(
    "--query-temperature",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Sampling temperature for the final query. Ignored for most-probable decoding.",
)
@click.option(
    "--query-seed",
    type=int,
    default=1,
    show_default=True,
    help="Random seed for the final query sampling stage.",
)
@click.option(
    "--text-normalization",
    type=click.Choice(TEXT_NORMALIZATION_MODES),
    default=DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before tokenizer and model training.",
)
@clearml_options
def main(
    model_name: str,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    tokenizer_limit: int | None,
    training_limit: int | None,
    evaluation_limit: int | None,
    vocab_size: int,
    artifact_name: str | None,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    discount: float,
    top_k: int,
    query_prompt: str,
    query_max_tokens: int,
    query_top_k: int,
    query_decoding: str,
    query_temperature: float,
    query_seed: int | None,
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
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")

    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_artifact_name = artifact_name or f"{corpus}-sentencepiece-{vocab_size}"
    resolved_tokenizer_limit = tokenizer_limit if tokenizer_limit is not None else limit
    resolved_training_limit = training_limit if training_limit is not None else limit
    resolved_evaluation_limit = evaluation_limit if evaluation_limit is not None else limit
    split_plan = build_cli_split_plan(
        corpus_definition,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )

    task_id: str | None = None
    task_url: str | None = None
    click.echo(stage_title(1, 5, "ClearML setup"), color=True)
    with (
        TemporaryDirectory(prefix="lme-pipeline-") as staging_root,
        start_clearml_run(
            clearml_settings(
                project_name=clearml_project,
                task_name=clearml_task_name,
                config_file=clearml_config_file,
                connectivity_check=clearml_connectivity_check,
                output_uri=clearml_output_uri,
                tags=clearml_tags,
            ),
            default_task_name=f"pipeline {model_definition.name} {corpus}",
            task_type="training",
        ) as clearml_run,
    ):
        staging_dir = Path(staging_root)
        tokenizer_output_prefix = staging_dir / resolved_artifact_name
        model_output_path = staging_dir / f"{corpus}-sentencepiece-{model_definition.name}.json"
        task_id = clearml_run.task_id
        task_url = clearml_run.task_url

        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.pipeline",
                    "artifact_store": "clearml",
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": resolved_dataset_id,
                    "source_split": source_split_label(resolved_source_split),
                    "training_partition": TRAIN_PARTITION,
                    "evaluation_partition": evaluation_partition,
                    "text_column": resolved_text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "tokenizer_limit": resolved_tokenizer_limit,
                    "training_limit": resolved_training_limit,
                    "evaluation_limit": resolved_evaluation_limit,
                    "text_normalization": text_normalization,
                },
                **split_plan_parameter_sections(split_plan),
                "Tokenizer": {
                    "vocab_size": vocab_size,
                    "artifact_name": resolved_artifact_name,
                    "model_type": model_type,
                    "character_coverage": character_coverage,
                    "hard_vocab_limit": hard_vocab_limit,
                    "max_sentence_length": max_sentence_length,
                },
                "Model": {
                    "model": model_definition.name,
                    "smoothing": smoothing,
                    "unigram_weight": unigram_weight,
                    "bigram_weight": bigram_weight,
                    "trigram_weight": trigram_weight,
                    "discount": discount,
                },
                "Evaluation": {
                    "top_k": top_k,
                },
                "Query": {
                    "query_prompt": query_prompt,
                    "query_max_tokens": query_max_tokens,
                    "query_top_k": query_top_k,
                    "query_decoding": query_decoding,
                    "query_temperature": query_temperature,
                    "query_seed": query_seed,
                },
                "Artifacts": {
                    "tokenizer_artifact": "sentencepiece-model",
                    "model_artifact": "trained-model-json",
                    "evaluation_artifact": "evaluation-summary",
                    "query_artifact": "query-result",
                },
            }
        )

        click.echo(stage_title(2, 5, "Tokenizer training"), color=True)
        tokenizer_model, tokenizer_vocab = train_sentencepiece(
            load_partition_texts(
                corpus_definition,
                dataset_id=resolved_dataset_id,
                plan=split_plan,
                partition=TRAIN_PARTITION,
                streaming=streaming,
                text_column=resolved_text_column,
                limit=resolved_tokenizer_limit,
            ),
            output_prefix=tokenizer_output_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            hard_vocab_limit=hard_vocab_limit,
            max_sentence_length=max_sentence_length,
            text_normalization=text_normalization,
        )
        clearml_run.log_metrics(
            "Tokenizer training",
            {
                "vocab_size": vocab_size,
                "character_coverage": character_coverage,
                "hard_vocab_limit": hard_vocab_limit,
                "limit": resolved_tokenizer_limit,
            },
        )
        upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"corpus": corpus, "stage": "pipeline"},
        )
        clearml_run.upload_artifact(
            "sentencepiece-model",
            tokenizer_model,
            metadata={"corpus": corpus, "vocab_size": vocab_size},
        )
        clearml_run.upload_artifact(
            "sentencepiece-vocabulary",
            tokenizer_vocab,
            metadata={"corpus": corpus, "vocab_size": vocab_size},
        )
        clearml_run.upload_artifact(
            "input-tokenizer-model",
            tokenizer_model,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.register_model(
            name=tokenizer_model.stem,
            model_path=tokenizer_model,
            framework="custom",
            tags=("tokenizer", corpus),
            comment="SentencePiece tokenizer model.",
        )

        click.echo(stage_title(3, 5, "Model training"), color=True)
        model_options = {
            "corpus": corpus,
            "tokenizer_model": tokenizer_model,
            "output": model_output_path,
            "stored_tokenizer_model": Path(tokenizer_model.name),
            "smoothing": smoothing,
            "unigram_weight": unigram_weight,
            "bigram_weight": bigram_weight,
            "trigram_weight": trigram_weight,
            "discount": discount,
            "text_normalization": text_normalization,
        }
        model_definition.validate_options(model_options)
        training_summary = model_definition.train(
            load_partition_texts(
                corpus_definition,
                dataset_id=resolved_dataset_id,
                plan=split_plan,
                partition=TRAIN_PARTITION,
                streaming=streaming,
                text_column=resolved_text_column,
                limit=resolved_training_limit,
            ),
            model_options,
        )
        attach_split_plan_to_json_model(training_summary.output_path, split_plan)
        clearml_run.log_metrics("Model training", training_summary_metrics(training_summary))
        clearml_run.upload_artifact(
            "trained-model-json",
            training_summary.output_path,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.register_model(
            name=training_summary.output_path.stem,
            model_path=training_summary.output_path,
            framework="custom",
            tags=("language-model", model_definition.name, corpus),
            comment="Token n-gram language model JSON.",
        )

        click.echo(stage_title(4, 5, "Evaluation"), color=True)
        evaluation_options = {
            "corpus": corpus,
            "model_path": training_summary.output_path,
            "top_k": top_k,
        }
        if model_definition.validate_evaluation_options is not None:
            model_definition.validate_evaluation_options(evaluation_options)
        evaluation_summary = model_definition.evaluate(
            load_partition_texts(
                corpus_definition,
                dataset_id=resolved_dataset_id,
                plan=split_plan,
                partition=evaluation_partition,
                streaming=streaming,
                text_column=resolved_text_column,
                limit=resolved_evaluation_limit,
            ),
            evaluation_options,
        )
        clearml_run.log_metrics(
            "Evaluation",
            evaluation_metrics_for_partition(
                evaluation_summary,
                partition=evaluation_partition,
            ),
        )
        clearml_run.upload_artifact(
            "evaluation-summary",
            pipeline_evaluation_payload(
                evaluation_summary,
                evaluation_partition=evaluation_partition,
                evaluation_limit=resolved_evaluation_limit,
                split_plan=split_plan,
            ),
            metadata={
                "model": model_definition.name,
                "corpus": corpus,
                "evaluation_partition": evaluation_partition,
                "split_id": split_plan.split_id,
            },
        )

        click.echo(stage_title(5, 5, "Query"), color=True)
        query_options = {
            "corpus": corpus,
            "model_path": training_summary.output_path,
            "prompt": query_prompt,
            "max_tokens": query_max_tokens,
            "top_k": query_top_k,
            "decoding": query_decoding,
            "temperature": query_temperature,
            "seed": query_seed,
        }
        if model_definition.validate_query_options is not None:
            model_definition.validate_query_options(query_options)
        query_result = model_definition.query(query_options)
        clearml_run.log_metrics("Query", query_metrics(query_result))
        clearml_run.upload_artifact(
            "query-result",
            pipeline_query_payload(query_result),
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "pipeline-summary",
            pipeline_payload(
                model_name=model_definition.name,
                corpus=corpus,
                dataset_id=resolved_dataset_id,
                source_split=resolved_source_split,
                training_partition=TRAIN_PARTITION,
                evaluation_partition=evaluation_partition,
                text_column=resolved_text_column,
                split_plan=split_plan,
                tokenizer_model=tokenizer_model,
                tokenizer_vocab=tokenizer_vocab,
                training_summary=training_summary,
                evaluation_summary=evaluation_summary,
                query_result=query_result,
            ),
            metadata={"model": model_definition.name, "corpus": corpus},
        )

    click.echo(f"Pipeline: {model_definition.name} on {corpus}")
    click.echo(f"Dataset: {resolved_dataset_id}")
    click.echo(f"Source split: {source_split_label(resolved_source_split)}")
    click.echo(f"Training partition: {TRAIN_PARTITION}")
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
    click.echo(f"Text normalization: {text_normalization}")
    echo_limit("Tokenizer limit", resolved_tokenizer_limit)
    echo_limit("Training limit", resolved_training_limit)
    echo_limit("Evaluation limit", resolved_evaluation_limit)
    click.echo("")
    click.echo(highlight_stage_title("Model training:"), color=True)
    for label, value in model_definition.summary_items(training_summary):
        click.echo(f"{label}: {value}")
    click.echo("")
    click.echo(highlight_stage_title("Evaluation:"), color=True)
    for label, value in model_definition.evaluation_items(evaluation_summary):
        click.echo(f"{label}: {value}")
    click.echo("")
    click.echo(highlight_stage_title("Query:"), color=True)
    for line in model_definition.query_lines(query_result):
        click.echo(line)
    click.echo("")
    click.echo(f"ClearML task ID: {task_id}")
    if task_url is not None:
        click.echo(f"ClearML task URL: {task_url}")
    click.echo("Data split artifact: data-split-plan-json")
    click.echo("Tokenizer artifact: sentencepiece-model")
    click.echo("Model artifact: trained-model-json")
    click.echo("Evaluation artifact: evaluation-summary")
    click.echo("Query artifact: query-result")


def pipeline_evaluation_payload(
    summary: object,
    *,
    evaluation_partition: str,
    evaluation_limit: int | None,
    split_plan: DataSplitPlan,
) -> dict[str, object]:
    return {
        **evaluation_payload(summary),
        "evaluation_partition": evaluation_partition,
        "evaluation_limit": evaluation_limit,
        "data_split": split_plan.to_payload(),
    }


def pipeline_query_payload(result: object) -> dict[str, object]:
    return query_payload(result)


def pipeline_payload(
    *,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    training_partition: str,
    evaluation_partition: str,
    text_column: str,
    split_plan: DataSplitPlan,
    tokenizer_model: Path,
    tokenizer_vocab: Path,
    training_summary: object,
    evaluation_summary: object,
    query_result: object,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "corpus": corpus,
        "dataset_id": dataset_id,
        "source_split": source_split_label(source_split),
        "training_partition": training_partition,
        "evaluation_partition": evaluation_partition,
        "text_column": text_column,
        "data_split": split_plan.to_payload(),
        "artifacts": {
            "sentencepiece_model": tokenizer_model.name,
            "sentencepiece_vocabulary": tokenizer_vocab.name,
            "input_tokenizer_model": tokenizer_model.name,
            "trained_model_json": getattr(training_summary, "output_path").name,
            "evaluation_summary": "evaluation-summary",
            "query_result": "query-result",
        },
        "training_metrics": training_summary_metrics(training_summary),
        "evaluation_metrics": evaluation_metrics(evaluation_summary),
        "query_metrics": query_metrics(query_result),
    }


def echo_limit(label: str, limit: int | None) -> None:
    if limit is None:
        click.echo(f"{label}: none")
        return
    click.echo(f"{label}: first {limit:,} rows")


if __name__ == "__main__":
    main()
