"""Generic Click CLI for evaluating registered language models."""

from __future__ import annotations

import click

from src.cli import corpus_source
from src.cli import options as cli_options
from src.cli import stage_resume
from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_training as model_pipeline
from src.models.core import registry as model_registry


def load_evaluate_command_defaults(_config_section: str) -> dict[str, object]:
    return stage_resume.load_stage_command_defaults("evaluate")


@cli_config.configured_command(
    "evaluate",
    default_loader=load_evaluate_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Evaluate a registered language model on a registered corpus.",
)
@core_pipeline.pipeline_resume_option
@core_pipeline.pipeline_options(
    default_name=model_pipeline.MODEL_TRAINING_PIPELINE.default_name,
    default_local=False,
    default_wait=False,
)
@cli_options.model_option("Registered model to evaluate.")
@cli_options.tokenizer_model_name_option
@cli_options.corpus_data_options
@cli_options.limit_option("Evaluate on only the first N rows. Useful for smoke tests.")
@cli_options.split_plan_options
@cli_options.evaluation_partition_option
@cli_options.top_k_option("K value for top-k next-token accuracy.")
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
    model_name: str,
    tokenizer_model_name: str | None,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    top_k: int,
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
    model_definition = model_registry.get_model(model_name)
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")

    resolved_tokenizer_model_name = stage_resume.require_tokenizer_model_name(
        tokenizer_model_name,
        action="Evaluation",
    )
    stage_resume.reject_pipeline_local(pipeline_local)
    stage_resume.resume_model_training_stage(
        stage_name=lm_def.EVALUATION_STAGE,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        controller_queue=controller_queue,
        wait=wait,
        pipeline_controller_id=pipeline_controller_id,
        clearml_project=clearml_project,
        clearml_task_name=clearml_task_name,
        clearml_config_file=clearml_config_file,
        clearml_connectivity_check=clearml_connectivity_check,
        clearml_output_uri=clearml_output_uri,
        clearml_tags=clearml_tags,
        parameter_filters={
            "model": model_definition.name,
            "tokenizer_model_name": resolved_tokenizer_model_name,
            "corpus": corpus,
            "dataset_id": source.dataset_id,
            "source_split": source.source_split or "",
            "text_column": source.text_column,
            "streaming": streaming,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "evaluation_partition": evaluation_partition,
            "evaluation_limit": limit,
            "top_k": top_k,
        },
    )


if __name__ == "__main__":
    main()
