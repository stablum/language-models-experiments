"""Click entry point for model training experiments."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli import model_training_defaults
from src.cli import model_training_flow
from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.ml_core.data import splits as data_splits
from src.models.core import registry as model_registry
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_training as model_pipeline
from src.pipelines.language_model import optuna as lm_optuna
from src.corpora import normalization
from src.corpora import registry as corpora_registry


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
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_registry.model_names()),
    default=model_registry.default_model_name(),
    show_default=True,
    help="Registered model to train, evaluate, and query.",
)
@click.option(
    "--tokenizer-model-name",
    default=None,
    help="Registered tokenizer model name to resolve from tokenizer-training runs.",
)
@click.option(
    "--tokenizer-training-name",
    default=lm_def.DEFAULT_TOKENIZER_TRAINING_NAME,
    show_default=True,
    help="ClearML tokenizer-training pipeline name to search for reusable tokenizer models.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpora_registry.corpus_names()),
    default=corpora_registry.default_corpus_name(),
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
    "--evaluation-partition",
    "--evaluation-split",
    type=click.Choice(data_splits.PROJECT_PARTITIONS),
    default=data_splits.VALIDATION_PARTITION,
    show_default=True,
    help=(
        "Primary project partition for unpartitioned summary metrics and Optuna "
        "objectives. The evaluation stage evaluates all project partitions."
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
    "--beta-2",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "Recursive interpolation beta for the bigram-vs-unigram branch. "
        "Set with --beta-3 to derive interpolation weights."
    ),
)
@click.option(
    "--beta-3",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "Recursive interpolation beta for the trigram-vs-lower-order branch. "
        "Set with --beta-2 to derive interpolation weights."
    ),
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
    type=click.Choice(normalization.TEXT_NORMALIZATION_MODES),
    default=normalization.DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before model training.",
)
@click.option(
    "--optuna-trials",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Run this many Optuna trials. Zero disables hyperparameter optimization.",
)
@click.option(
    "--optuna-search",
    "optuna_search",
    multiple=True,
    help=(
        "Hyperparameter search spec. Repeatable. Examples: "
        "smoothing=float:1e-4:1.0:log, discount=float:0.1:0.95, "
        "beta_2=float:0:1.0, top_k=int:1:10, "
        "model=categorical:bigram,trigram-add-k."
    ),
)
@click.option(
    "--optuna-metric",
    default=lm_optuna.DEFAULT_OPTUNA_METRIC,
    show_default=True,
    help="Evaluation summary metric used as the Optuna objective.",
)
@click.option(
    "--optuna-direction",
    type=click.Choice(("minimize", "maximize")),
    default=lm_optuna.DEFAULT_OPTUNA_DIRECTION,
    show_default=True,
    help="Whether Optuna should minimize or maximize the objective metric.",
)
@click.option(
    "--optuna-study-name",
    default=None,
    help="Optional Optuna study name. Required by some persistent storage backends.",
)
@click.option(
    "--optuna-storage",
    default=None,
    help="Optional Optuna storage URL, for example sqlite:///optuna.db.",
)
@click.option(
    "--optuna-load-if-exists/--optuna-no-load-if-exists",
    default=True,
    show_default=True,
    help="Reuse an existing named Optuna study when storage is configured.",
)
@click.option(
    "--optuna-timeout-seconds",
    type=click.IntRange(min=1),
    default=None,
    help="Optional maximum wall-clock time for the Optuna study.",
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
    run_until_stage: str | None,
    run_stage: str | None,
    pipeline_controller_id: str | None,
    model_name: str,
    tokenizer_model_name: str | None,
    tokenizer_training_name: str,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    training_limit: int | None,
    evaluation_limit: int | None,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    beta_2: float | None,
    beta_3: float | None,
    discount: float,
    top_k: int,
    query_prompt: str,
    query_max_tokens: int,
    query_top_k: int,
    query_decoding: str,
    query_temperature: float,
    query_seed: int | None,
    text_normalization: str,
    optuna_trials: int,
    optuna_search: tuple[str, ...],
    optuna_metric: str,
    optuna_direction: str,
    optuna_study_name: str | None,
    optuna_storage: str | None,
    optuna_load_if_exists: bool,
    optuna_timeout_seconds: int | None,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    model_training_flow.run(model_training_flow.CliArgs(**locals()))


if __name__ == "__main__":
    main()
