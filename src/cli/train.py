"""Generic Click CLI for training registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.text import iter_text_column
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import clearml_options, clearml_settings, start_clearml_run


@click.command(
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
    help="Train on only the first N rows. Useful for smoke tests.",
)
@click.option(
    "--tokenizer-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="SentencePiece .model file for token-based models.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for the trained model.",
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
def main(
    model_name: str,
    corpus: str,
    dataset_id: str | None,
    split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    tokenizer_model: Path | None,
    output: Path | None,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    discount: float,
    text_normalization: str,
    clearml: bool,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = get_corpus(corpus)
    model_definition = get_model(model_name)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_split = split or corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column

    model_options = {
        "corpus": corpus,
        "tokenizer_model": tokenizer_model,
        "output": output,
        "smoothing": smoothing,
        "unigram_weight": unigram_weight,
        "bigram_weight": bigram_weight,
        "trigram_weight": trigram_weight,
        "discount": discount,
        "text_normalization": text_normalization,
    }
    model_definition.validate_options(model_options)

    with start_clearml_run(
        clearml_settings(
            enabled=clearml,
            project_name=clearml_project,
            task_name=clearml_task_name,
            output_uri=clearml_output_uri,
            tags=clearml_tags,
        ),
        default_task_name=f"train {model_definition.name} {corpus}",
        task_type="training",
    ) as clearml_run:
        clearml_run.connect_parameters(
            {
                "command": "src.cli.train",
                "model": model_definition.name,
                "corpus": corpus,
                "dataset_id": resolved_dataset_id,
                "split": resolved_split,
                "text_column": resolved_text_column,
                "streaming": streaming,
                "limit": limit,
                **model_options,
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

        summary = model_definition.train(texts, model_options)

        clearml_run.log_metrics(
            "Model training",
            training_summary_metrics(summary),
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
    click.echo(f"Split: {resolved_split}")
    click.echo(f"Text column: {resolved_text_column}")
    click.echo(f"Text normalization: {text_normalization}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    for label, value in model_definition.summary_items(summary):
        click.echo(f"{label}: {value}")


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
