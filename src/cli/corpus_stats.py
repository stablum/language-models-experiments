"""Generic Click CLI for registered corpus statistics."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli import corpus_source
from src.cli import options as cli_options
from src.corpora import stats as corpus_stats
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.ml_core.cli import output as cli_output
from src.ml_core.data import split_artifacts
from src.ml_core.data import splits as data_splits


@cli_config.configured_command(
    "corpus_stats",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Load a registered corpus and print row, character, "
        "and simple whitespace-token statistics."
    ),
)
@cli_options.corpus_data_options
@cli_options.limit_option("Scan only the first N rows. Useful for quick smoke tests.")
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
@cli_options.text_normalization_option("Text normalization applied before computing stats.")
@tracking.clearml_options
def main(
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    top_n_lengths: int,
    preview_chars: int,
    text_normalization: str,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    source = corpus_source.resolve(
        corpus=corpus,
        dataset_id=dataset_id,
        source_split=source_split,
        text_column=text_column,
    )
    split_plan = split_artifacts.build_cli_split_plan(
        source.definition,
        corpus=corpus,
        dataset_id=source.dataset_id,
        source_split=source.source_split,
        train_ratio=data_splits.DEFAULT_TRAIN_RATIO,
        split_seed=data_splits.DEFAULT_SPLIT_SEED,
    )

    click.echo(cli_output.stage_title(1, 1, "Corpus stats"), color=True)
    with tracking.start_clearml_run(
        tracking.clearml_settings(
            project_name=clearml_project,
            task_name=clearml_task_name,
            config_file=clearml_config_file,
            connectivity_check=clearml_connectivity_check,
            output_uri=clearml_output_uri,
            tags=clearml_tags,
        ),
        default_task_name=f"corpus stats {corpus}",
        task_type="data_processing",
    ) as clearml_run:
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.corpus_stats",
                    "artifact_store": "clearml",
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": source.dataset_id,
                    "dataset_revision": split_plan.dataset_revision or "",
                    "source_split": data_splits.source_split_label(source.source_split),
                    "text_column": source.text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Reporting": {
                    "top_n_lengths": top_n_lengths,
                    "preview_chars": preview_chars,
                },
            }
        )

        dataset = source.definition.load(
            dataset_id=source.dataset_id,
            revision=split_plan.dataset_revision,
            split=source.source_split,
            streaming=streaming,
        )
        rows = (
            row
            for _, _, row in data_splits.iter_merged_source_rows(
                dataset,
                plan=split_plan,
            )
        )

        stats = corpus_stats.scan_text_column(
            rows,
            text_column=source.text_column,
            limit=limit,
            top_n_lengths=top_n_lengths,
            preview_chars=preview_chars,
            text_normalization=text_normalization,
        )

        clearml_run.log_metrics("Corpus stats", corpus_stats_metrics(stats))
        clearml_run.upload_artifact(
            "corpus-stats",
            corpus_stats_payload(stats),
            metadata={
                "corpus": corpus,
                "source_split": data_splits.source_split_label(source.source_split),
            },
        )

    corpus_stats.print_corpus_report(
        dataset_label=source.dataset_id,
        split=data_splits.source_split_label(source.source_split),
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
        for metric, value, _format_spec in corpus_stats.distribution_metrics(values, total)
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
