"""Generic Click CLI for registered corpus statistics."""

from __future__ import annotations

import click

from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import (
    DEFAULT_CORPUS_NAME,
    corpus_names,
    get_corpus,
)
from src.corpora.stats import distribution_metrics, print_corpus_report, scan_text_column
from src.tracking.clearml import clearml_options, clearml_settings, start_clearml_run


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Load a registered corpus and print row, character, "
        "and simple whitespace-token statistics."
    ),
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus to scan.",
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
    help="Scan only the first N rows. Useful for quick smoke tests.",
)
@click.option(
    "--top-n-lengths",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="Show the N longest rows by character count. Use 0 to hide examples.",
)
@click.option(
    "--preview-chars",
    type=click.IntRange(min=0),
    default=120,
    show_default=True,
    help="Characters to show from each longest-row preview.",
)
@click.option(
    "--text-normalization",
    type=click.Choice(TEXT_NORMALIZATION_MODES),
    default=DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before computing stats.",
)
@clearml_options
def main(
    corpus: str,
    dataset_id: str | None,
    split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    top_n_lengths: int,
    preview_chars: int,
    text_normalization: str,
    clearml: bool,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = get_corpus(corpus)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_split = split or corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column

    with start_clearml_run(
        clearml_settings(
            enabled=clearml,
            project_name=clearml_project,
            task_name=clearml_task_name,
            output_uri=clearml_output_uri,
            tags=clearml_tags,
        ),
        default_task_name=f"corpus stats {corpus}",
        task_type="data_processing",
    ) as clearml_run:
        clearml_run.connect_parameters(
            {
                "command": "src.cli.corpus_stats",
                "corpus": corpus,
                "dataset_id": resolved_dataset_id,
                "split": resolved_split,
                "text_column": resolved_text_column,
                "streaming": streaming,
                "limit": limit,
                "top_n_lengths": top_n_lengths,
                "preview_chars": preview_chars,
                "text_normalization": text_normalization,
            }
        )

        dataset = corpus_definition.load(
            dataset_id=resolved_dataset_id,
            split=resolved_split,
            streaming=streaming,
        )

        stats = scan_text_column(
            dataset,
            text_column=resolved_text_column,
            limit=limit,
            top_n_lengths=top_n_lengths,
            preview_chars=preview_chars,
            text_normalization=text_normalization,
        )

        clearml_run.log_metrics("Corpus stats", corpus_stats_metrics(stats))
        clearml_run.upload_artifact(
            "corpus-stats",
            corpus_stats_payload(stats),
            metadata={"corpus": corpus, "split": resolved_split},
        )

    print_corpus_report(
        dataset_label=resolved_dataset_id,
        split=resolved_split,
        mode="streaming" if streaming else "download/cache",
        limit=limit,
        reported_rows=getattr(dataset, "num_rows", None),
        features=getattr(dataset, "features", None),
        stats=stats,
    )
    click.echo(f"Text normalization: {text_normalization}")


def corpus_stats_metrics(stats: object) -> dict[str, object]:
    metrics = {
        "rows": getattr(stats, "rows", None),
        "nonempty_rows": getattr(stats, "nonempty_rows", None),
        "empty_rows": getattr(stats, "rows", 0) - getattr(stats, "nonempty_rows", 0),
        "total_chars": getattr(stats, "total_chars", None),
        "total_newlines": getattr(stats, "total_newlines", None),
        "total_whitespace_tokens": getattr(stats, "total_whitespace_tokens", None),
    }
    metrics.update(
        prefixed_distribution_metrics(
            "chars",
            getattr(stats, "char_lengths", []),
            getattr(stats, "total_chars", 0),
        )
    )
    metrics.update(
        prefixed_distribution_metrics(
            "whitespace_tokens",
            getattr(stats, "token_lengths", []),
            getattr(stats, "total_whitespace_tokens", 0),
        )
    )
    return metrics


def prefixed_distribution_metrics(
    prefix: str,
    values: list[int],
    total: int,
) -> dict[str, float]:
    return {
        f"{prefix}_{metric}": value
        for metric, value, _format_spec in distribution_metrics(values, total)
    }


def corpus_stats_payload(stats: object) -> dict[str, object]:
    return {
        "metrics": corpus_stats_metrics(stats),
        "longest_examples": [
            {
                "row_number": example.row_number,
                "char_count": example.char_count,
                "token_count": example.token_count,
                "preview": example.preview,
            }
            for example in sorted(getattr(stats, "longest_examples", []), reverse=True)
        ],
    }


if __name__ == "__main__":
    main()
