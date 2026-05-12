"""ClearML PipelineController DAG for reusable tokenizer training."""

from __future__ import annotations

from pathlib import Path

import click

from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.ml_core.data import splits as data_splits
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import tokenizer_training as tokenizer_pipeline
from src.corpora import normalization
from src.corpora import registry as corpora_registry
from src.tokenizers import registry as tokenizer_registry


TOKENIZER_TRAINING_CONFIG_SECTION = "tokenizer-training"


def load_tokenizer_training_command_defaults(_config_section: str) -> dict[str, object]:
    return cli_config.load_defaults_from_sections(
        ("defaults", "clearml", TOKENIZER_TRAINING_CONFIG_SECTION)
    )


@cli_config.configured_command(
    "tokenizer-training",
    default_loader=load_tokenizer_training_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run reusable tokenizer training as a ClearML Pipeline DAG.",
)
@core_pipeline.pipeline_resume_option
@core_pipeline.pipeline_options(
    default_name=tokenizer_pipeline.TOKENIZER_TRAINING_PIPELINE.default_name
)
@click.option(
    "--corpus",
    type=click.Choice(corpora_registry.corpus_names()),
    default=corpora_registry.default_corpus_name(),
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
    default=data_splits.DEFAULT_TRAIN_RATIO,
    show_default=True,
    help="Fraction of merged source rows assigned to the reusable training partition.",
)
@click.option(
    "--split-seed",
    type=int,
    default=data_splits.DEFAULT_SPLIT_SEED,
    show_default=True,
    help="Seed for the reusable deterministic train/validation partition.",
)
@click.option(
    "--vocab-size",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="Tokenizer vocabulary size.",
)
@click.option(
    "--artifact-name",
    default=None,
    help="Base name for the tokenizer model and vocabulary outputs stored in ClearML.",
)
@click.option(
    "--tokenizer-algo",
    type=click.Choice(tokenizer_registry.tokenizer_algo_names()),
    default=tokenizer_registry.DEFAULT_TOKENIZER_ALGO,
    show_default=True,
    help="Tokenizer training algorithm.",
)
@click.option(
    "--sentencepiece-model-type",
    "--model-type",
    "sentencepiece_model_type",
    type=click.Choice(("unigram", "bpe", "char", "word")),
    default="unigram",
    show_default=True,
    help="SentencePiece model type.",
)
@click.option(
    "--sentencepiece-character-coverage",
    "--character-coverage",
    "sentencepiece_character_coverage",
    type=float,
    default=1.0,
    show_default=True,
    help="Fraction of characters covered by the model.",
)
@click.option(
    "--sentencepiece-hard-vocab-limit/--no-sentencepiece-hard-vocab-limit",
    "--hard-vocab-limit/--no-hard-vocab-limit",
    "sentencepiece_hard_vocab_limit",
    default=True,
    show_default=True,
    help="Require SentencePiece to produce exactly vocab-size pieces.",
)
@click.option(
    "--sentencepiece-max-sentence-length",
    "--max-sentence-length",
    "sentencepiece_max_sentence_length",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum sentence length passed to SentencePiece.",
)
@click.option(
    "--text-normalization",
    type=click.Choice(normalization.TEXT_NORMALIZATION_MODES),
    default=normalization.DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before tokenizer training.",
)
@tracking.clearml_options
def main(
    pipeline_name: str,
    pipeline_version: str,
    pipeline_local: bool,
    controller_queue: str,
    execution_queue: str | None,
    wait: bool,
    add_run_number: bool,
    pipeline_controller_id: str | None,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    vocab_size: int,
    artifact_name: str | None,
    tokenizer_algo: str,
    sentencepiece_model_type: str,
    sentencepiece_character_coverage: float,
    sentencepiece_hard_vocab_limit: bool,
    sentencepiece_max_sentence_length: int | None,
    text_normalization: str,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    if pipeline_local and not wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")

    corpus_definition = corpora_registry.get_corpus(corpus)
    resolved_pipeline_name = clearml_task_name or pipeline_name
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_artifact_name = artifact_name or tokenizer_registry.default_artifact_name(
        corpus=corpus,
        tokenizer_algo=tokenizer_algo,
        vocab_size=vocab_size,
    )

    parameter_filters = {
        "corpus": corpus,
        "tokenizer_model_name": resolved_artifact_name,
        "tokenizer_algo": tokenizer_algo,
    }
    if pipeline_controller_id is not None:
        if pipeline_local:
            raise click.ClickException(
                "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
                "Use --pipeline-queued with --pipeline-controller-id."
            )
        core_pipeline.resume_pipeline_controller_stage(
            stage_name=lm_def.TOKENIZER_STAGE,
            pipeline_controller_id=pipeline_controller_id,
            pipeline_name=resolved_pipeline_name,
            pipeline_version=pipeline_version,
            controller_queue=controller_queue,
            wait=wait,
            clearml_project=clearml_project,
            clearml_task_name=clearml_task_name,
            clearml_config_file=clearml_config_file,
            clearml_connectivity_check=clearml_connectivity_check,
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            parameter_filters=parameter_filters,
            stage_dependencies=(
                tokenizer_pipeline.TOKENIZER_TRAINING_PIPELINE.stage_dependencies
            ),
            stage_names=tokenizer_pipeline.TOKENIZER_TRAINING_PIPELINE.stages,
        )
        return

    settings = tracking.clearml_settings(
        project_name=clearml_project,
        task_name=resolved_pipeline_name,
        config_file=clearml_config_file,
        connectivity_check=clearml_connectivity_check,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
    )
    resolved_config_file = tracking.configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        tracking.assert_clearml_endpoints_reachable(
            resolved_config_file,
            settings.output_uri,
        )

    pipeline = core_pipeline.build_pipeline_controller(
        pipeline_name=resolved_pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=settings.project_name,
        clearml_tags=settings.tags,
        clearml_output_uri=settings.output_uri,
        add_run_number=add_run_number,
    )
    lm_def.configure_pipeline_control(
        pipeline.task,
        run_stage=None,
        run_until_stage=None,
        updated_by="tokenizer-pipeline-cli",
    )
    core_pipeline.connect_controller_experiment_parameters(
        pipeline.task,
        {
            "corpus": corpus,
            "tokenizer_model_name": resolved_artifact_name,
            "dataset_id": resolved_dataset_id,
            "source_split": resolved_source_split or "",
            "text_column": resolved_text_column,
            "vocab_size": vocab_size,
            "tokenizer_algo": tokenizer_algo,
            "sentencepiece_model_type": sentencepiece_model_type,
            "text_normalization": text_normalization,
        },
    )
    tokenizer_pipeline.add_pipeline_steps(
        pipeline,
        clearml_project=settings.project_name,
        clearml_output_uri=settings.output_uri,
        clearml_tags=settings.tags,
        clearml_config_file=resolved_config_file if pipeline_local else None,
        execution_queue=None if pipeline_local else execution_queue,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        text_column=resolved_text_column,
        streaming=streaming,
        limit=limit,
        train_ratio=train_ratio,
        split_seed=split_seed,
        vocab_size=vocab_size,
        artifact_name=resolved_artifact_name,
        tokenizer_algo=tokenizer_algo,
        sentencepiece_model_type=sentencepiece_model_type,
        sentencepiece_character_coverage=sentencepiece_character_coverage,
        sentencepiece_hard_vocab_limit=sentencepiece_hard_vocab_limit,
        sentencepiece_max_sentence_length=sentencepiece_max_sentence_length,
        text_normalization=text_normalization,
    )

    click.echo(f"ClearML tokenizer-training pipeline: {settings.project_name}/{resolved_pipeline_name}")
    click.echo(f"Pipeline version: {pipeline_version}")
    click.echo(f"Tokenizer algorithm: {tokenizer_algo}")
    click.echo(f"Tokenizer model name: {resolved_artifact_name}")
    click.echo(f"Pipeline controller task ID: {pipeline.task.id}")
    task_url = pipeline.task.get_output_log_web_page()
    if task_url:
        click.echo(f"Pipeline controller URL: {task_url}")
    click.echo(f"Stage tasks: {lm_def.TOKENIZER_STAGE}")

    if pipeline_local:
        click.echo("Execution mode: local ClearML PipelineController")
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        click.echo(f"Execution mode: queued controller on {controller_queue}")
        if execution_queue is not None:
            click.echo(f"Step execution queue: {execution_queue}")
        pipeline.start(queue=controller_queue, wait=wait)

    click.echo("ClearML tokenizer-training pipeline submitted.")
    if wait:
        core_pipeline.assert_pipeline_finished_successfully(pipeline)
        core_pipeline.print_stage_task_ids(
            pipeline.task.id,
            tokenizer_pipeline.TOKENIZER_TRAINING_PIPELINE.stages,
            stage_names=tokenizer_pipeline.TOKENIZER_TRAINING_PIPELINE.stages,
        )
        click.echo("ClearML tokenizer-training pipeline run completed.")


if __name__ == "__main__":
    main()
