"""Click entry point for model training experiments."""

from __future__ import annotations

import click

from src.cli import model_training_defaults
from src.cli import model_training_flow
from src.cli import options as cli_options
from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_training as model_pipeline


@cli_config.configured_command(
    "model-training",
    default_loader=model_training_defaults.load_model_training_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run model training, evaluation, and query as a ClearML Pipeline DAG.",
)
@core_pipeline.pipeline_resume_option
@click.option(
    "--run-stage",
    type=click.Choice(model_pipeline.MODEL_TRAINING_PIPELINE.stages),
    default=None,
    help=(
        "Resume an existing controller and run only this stage. "
        "If --pipeline-controller-id is omitted, the newest eligible run is selected."
    ),
)
@click.option(
    "--run-until-stage",
    type=click.Choice(model_pipeline.MODEL_TRAINING_PIPELINE.stages),
    default=None,
    help="Create a new controller run and stop after this stage has run.",
)
@core_pipeline.pipeline_options(
    default_name=model_pipeline.MODEL_TRAINING_PIPELINE.default_name
)
@cli_options.model_option("Registered model to train, evaluate, and query.")
@cli_options.tokenizer_model_name_option
@click.option(
    "--tokenizer-training-name",
    default=lm_def.DEFAULT_TOKENIZER_TRAINING_NAME,
    show_default=True,
    help="ClearML tokenizer-training pipeline name to search for reusable tokenizer models.",
)
@cli_options.corpus_data_options
@cli_options.split_plan_options
@cli_options.evaluation_partition_option
@click.option(
    "--limit",
    type=click.IntRange(min=0),
    default=None,
    help="Apply the same row limit to model training and evaluation.",
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
@cli_options.model_hyperparameter_options
@cli_options.top_k_option("K value for top-k next-token accuracy.")
@cli_options.pipeline_query_options
@cli_options.text_normalization_option("Text normalization applied before model training.")
@cli_options.optuna_options
@tracking.clearml_options
def main(**kwargs: object) -> None:
    model_training_flow.run(model_training_flow.CliArgs(**kwargs))


if __name__ == "__main__":
    main()
