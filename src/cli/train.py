"""Generic Click CLI for training registered language models."""

from __future__ import annotations

import click

from src.cli import options as cli_options
from src.cli import stage_resume
from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import model_training as model_pipeline


@cli_config.configured_command(
    "train",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Train a registered language model from a registered corpus.",
)
@core_pipeline.pipeline_resume_option
@core_pipeline.pipeline_options(
    default_name=model_pipeline.MODEL_TRAINING_PIPELINE.default_name,
    default_local=False,
    default_wait=False,
)
@cli_options.model_option("Registered model to train.")
@cli_options.tokenizer_model_name_option
@cli_options.corpus_data_options
@cli_options.limit_option("Train on only the first N rows. Useful for smoke tests.")
@cli_options.split_plan_options
@cli_options.model_hyperparameter_options
@cli_options.text_normalization_option("Text normalization applied before model training.")
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
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    beta_2: float | None,
    beta_3: float | None,
    discount: float,
    text_normalization: str,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    filter_resolution = stage_resume.resolve_model_training_stage_filters(
        stage_resume.ModelTrainingStageFilterCfg(
            model_name=model_name,
            tokenizer_model_name=tokenizer_model_name,
            action="Model training",
            corpus=stage_resume.CorpusFilterCfg(
                corpus=corpus,
                dataset_id=dataset_id,
                source_split=source_split,
                text_column=text_column,
                streaming=streaming,
                train_ratio=train_ratio,
                split_seed=split_seed,
            ),
            limit_param="training_limit",
            limit=limit,
        ),
        extra_filters={
            **lm_model_options.model_hyperparameters_from(locals()),
            "text_normalization": text_normalization,
        },
    )
    stage_resume.reject_pipeline_local(pipeline_local)
    stage_resume.resume_model_training_stage(
        stage_name=lm_def.MODEL_STAGE,
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
        parameter_filters=filter_resolution.filters,
    )


if __name__ == "__main__":
    main()
