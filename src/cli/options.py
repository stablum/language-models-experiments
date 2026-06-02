"""Reusable Click option groups for project CLIs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click

from src.corpora import normalization
from src.corpora import registry as corpora_registry
from src.ml_core.data import splits as data_splits
from src.models.core import registry as model_registry
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import optuna as lm_optuna


Command = Callable[..., object]
CommandDecorator = Callable[[Command], Command]
DECODING_MODES = ("sample", "most-probable")


def apply_options(*decorators: CommandDecorator) -> CommandDecorator:
    """Apply option decorators while preserving normal stacked-decorator order."""

    def decorate(command: Command) -> Command:
        for decorator in reversed(decorators):
            command = decorator(command)
        return command

    return decorate


def model_option(help_text: str = "Registered model to use.") -> CommandDecorator:
    return click.option(
        "--model",
        "model_name",
        type=click.Choice(model_registry.model_names()),
        default=model_registry.default_model_name(),
        show_default=True,
        help=help_text,
    )


def tokenizer_model_name_option(command: Command) -> Command:
    return click.option(
        "--tokenizer-model-name",
        default=None,
        help="Registered tokenizer model name used by model training.",
    )(command)


def corpus_option(command: Command) -> Command:
    return click.option(
        "--corpus",
        type=click.Choice(corpora_registry.corpus_names()),
        default=corpora_registry.default_corpus_name(),
        show_default=True,
        help="Registered corpus to use.",
    )(command)


def dataset_id_option(command: Command) -> Command:
    return click.option(
        "--dataset-id",
        default=None,
        help="Override the registered Hugging Face dataset ID.",
    )(command)


def source_split_option(command: Command) -> Command:
    return click.option(
        "--source-split",
        "source_split",
        default=None,
        help=(
            "Restrict the source dataset to one named split before downstream "
            "processing. Omit to merge all source splits."
        ),
    )(command)


def text_column_option(command: Command) -> Command:
    return click.option(
        "--text-column",
        default=None,
        help="Override the registered text column.",
    )(command)


def streaming_option(command: Command) -> Command:
    return click.option(
        "--streaming",
        is_flag=True,
        help="Stream rows instead of downloading the full dataset first.",
    )(command)


def corpus_data_options(command: Command) -> Command:
    return apply_options(
        corpus_option,
        dataset_id_option,
        source_split_option,
        text_column_option,
        streaming_option,
    )(command)


def train_ratio_option(command: Command) -> Command:
    return click.option(
        "--train-ratio",
        type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
        default=data_splits.DEFAULT_TRAIN_RATIO,
        show_default=True,
        help="Fraction of merged source rows assigned to the reusable training partition.",
    )(command)


def split_seed_option(command: Command) -> Command:
    return click.option(
        "--split-seed",
        type=int,
        default=data_splits.DEFAULT_SPLIT_SEED,
        show_default=True,
        help="Seed for the reusable deterministic train/validation partition.",
    )(command)


def split_plan_options(command: Command) -> Command:
    return apply_options(train_ratio_option, split_seed_option)(command)


def limit_option(help_text: str) -> CommandDecorator:
    return click.option(
        "--limit",
        type=click.IntRange(min=0),
        default=None,
        help=help_text,
    )


def evaluation_partition_option(command: Command) -> Command:
    return click.option(
        "--evaluation-partition",
        type=click.Choice(data_splits.PROJECT_PARTITIONS),
        default=data_splits.VALIDATION_PARTITION,
        show_default=True,
        help=(
            "Primary project partition for unpartitioned summary metrics and Optuna "
            "objectives. The evaluation stage evaluates all project partitions."
        ),
    )(command)


def top_k_option(help_text: str) -> CommandDecorator:
    return click.option(
        "--top-k",
        type=click.IntRange(min=1),
        default=5,
        show_default=True,
        help=help_text,
    )


def text_normalization_option(help_text: str) -> CommandDecorator:
    return click.option(
        "--text-normalization",
        type=click.Choice(normalization.TEXT_NORMALIZATION_MODES),
        default=normalization.DEFAULT_TEXT_NORMALIZATION,
        show_default=True,
        help=help_text,
    )


def model_hyperparameter_options(command: Command) -> Command:
    return apply_options(
        *(
            _model_hyperparameter_option(name)
            for name in lm_model_options.MODEL_HYPERPARAMETER_NAMES
        )
    )(command)


def _model_hyperparameter_option(name: str) -> CommandDecorator:
    default = lm_model_options.MODEL_HYPERPARAMETER_DEFAULTS[name]
    return click.option(
        f"--{name.replace('_', '-')}",
        type=_MODEL_HYPERPARAMETER_TYPES[name],
        default=default,
        show_default=default is not None,
        help=lm_model_options.MODEL_HYPERPARAMETER_DESCRIPTIONS[name],
    )


def query_generation_options(command: Command) -> Command:
    return apply_options(
        click.option(
            "--prompt",
            default="",
            show_default=True,
            help="Text prefix to condition on.",
        ),
        click.option(
            "--max-tokens",
            type=click.IntRange(min=0),
            default=80,
            show_default=True,
            help="Maximum number of new tokens to generate.",
        ),
        click.option(
            "--top-k",
            type=click.IntRange(min=1),
            default=10,
            show_default=True,
            help="Number of likely next tokens to print for the prompt.",
        ),
        click.option(
            "--decoding",
            type=click.Choice(DECODING_MODES),
            default="sample",
            show_default=True,
            help="Generate by sampling or by choosing the most probable next token.",
        ),
        click.option(
            "--temperature",
            type=click.FloatRange(min=0.0),
            default=1.0,
            show_default=True,
            help="Sampling temperature. Ignored for most-probable decoding.",
        ),
        click.option(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducible sampling.",
        ),
    )(command)


def pipeline_query_options(command: Command) -> Command:
    return apply_options(
        click.option(
            "--query-prompt",
            default="Once upon",
            show_default=True,
            help="Text prefix for the final query stage.",
        ),
        click.option(
            "--query-max-tokens",
            type=click.IntRange(min=0),
            default=80,
            show_default=True,
            help="Maximum number of new tokens to generate in the final query stage.",
        ),
        click.option(
            "--query-top-k",
            type=click.IntRange(min=1),
            default=10,
            show_default=True,
            help="Number of likely next tokens to store for the query prompt.",
        ),
        click.option(
            "--query-decoding",
            type=click.Choice(DECODING_MODES),
            default="sample",
            show_default=True,
            help=(
                "Generate the final query by sampling or by choosing the most "
                "probable next token."
            ),
        ),
        click.option(
            "--query-temperature",
            type=click.FloatRange(min=0.0),
            default=1.0,
            show_default=True,
            help="Sampling temperature for the final query. Ignored for most-probable decoding.",
        ),
        click.option(
            "--query-seed",
            type=int,
            default=1,
            show_default=True,
            help="Random seed for the final query sampling stage.",
        ),
    )(command)


def model_path_option(command: Command) -> Command:
    return click.option(
        "--model-path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Local trained model JSON to query. Overrides pipeline lookup.",
    )(command)


def optuna_options(command: Command) -> Command:
    return apply_options(
        click.option(
            "--optuna-trials",
            type=click.IntRange(min=0),
            default=0,
            show_default=True,
            help="Run this many Optuna trials. Zero disables hyperparameter optimization.",
        ),
        click.option(
            "--optuna-search",
            "optuna_search",
            multiple=True,
            help=(
                "Hyperparameter search spec. Repeatable. Examples: "
                "smoothing=float:1e-4:1.0:log, discount=float:0.1:0.95, "
                "beta_2=float:0:1.0, top_k=int:1:10, "
                "model_name=categorical:bigram,trigram-add-k."
            ),
        ),
        click.option(
            "--optuna-metric",
            default=lm_optuna.DEFAULT_OPTUNA_METRIC,
            show_default=True,
            help="Evaluation summary metric used as the Optuna objective.",
        ),
        click.option(
            "--optuna-direction",
            type=click.Choice(("minimize", "maximize")),
            default=lm_optuna.DEFAULT_OPTUNA_DIRECTION,
            show_default=True,
            help="Whether Optuna should minimize or maximize the objective metric.",
        ),
        click.option(
            "--optuna-study-name",
            default=None,
            help="Optional Optuna study name. Required by some persistent storage backends.",
        ),
        click.option(
            "--optuna-storage",
            default=None,
            help="Optional Optuna storage URL, for example sqlite:///optuna.db.",
        ),
        click.option(
            "--optuna-load-if-exists/--optuna-no-load-if-exists",
            default=True,
            show_default=True,
            help="Reuse an existing named Optuna study when storage is configured.",
        ),
        click.option(
            "--optuna-timeout-seconds",
            type=click.IntRange(min=1),
            default=None,
            help="Optional maximum wall-clock time for the Optuna study.",
        ),
    )(command)


_MODEL_HYPERPARAMETER_TYPES = {
    "smoothing": click.FloatRange(min=0.0),
    "unigram_weight": click.FloatRange(min=0.0),
    "bigram_weight": click.FloatRange(min=0.0),
    "trigram_weight": click.FloatRange(min=0.0),
    "beta_2": click.FloatRange(min=0.0, max=1.0),
    "beta_3": click.FloatRange(min=0.0, max=1.0),
    "discount": click.FloatRange(min=0.0, max=1.0),
}
