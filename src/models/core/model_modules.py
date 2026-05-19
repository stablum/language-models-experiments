"""Adapters from concrete model modules to the shared registry contract."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TypeVar

from src.ml_core.models import definition as model_def
from src.models.core import formatting, ngram
from src.tokenizers import core as tok_core


LoadedModel = TypeVar("LoadedModel")
TrainingSummary = TypeVar("TrainingSummary")
QueryResult = TypeVar("QueryResult")
EvaluationSummary = TypeVar("EvaluationSummary")

_TRAINING_INFRA_OPTION_NAMES = frozenset(
    (
        "tokenizer_model",
        "output_path",
        "stored_tokenizer_model",
        "text_normalization",
    )
)
_INTERPOLATION_OPTION_NAMES = frozenset(
    ("unigram_weight", "bigram_weight", "trigram_weight", "beta_2", "beta_3")
)


def model_definition_from_module(module: ModuleType) -> model_def.ModelDefinition | None:
    train_model = get_module_callable(module, "train")
    load_model = get_module_callable(module, "load")
    summary_items = get_module_callable(module, "format_summary")
    if train_model is None or load_model is None or summary_items is None:
        return None

    training_option_names = infer_training_option_names(train_model)
    validate_training_options = get_module_callable(
        module,
        "validate_training_options",
    )

    return model_definition(
        module_name=module.__name__,
        train_model=train_model,
        load_model=load_model,
        summary_items=summary_items,
        training_option_names=training_option_names,
        query_lines=get_module_callable(module, "format_query"),
        evaluation_items=get_module_callable(module, "format_evaluation"),
        validate_training_options=(
            validate_training_options
            or inferred_training_options_validator(training_option_names)
        ),
    )


def get_module_callable(module: ModuleType, name: str) -> Callable[..., Any] | None:
    fn = getattr(module, name, None)
    if fn is not None and not callable(fn):
        raise TypeError(f"{module.__name__}.{name} must be callable")
    return fn


def infer_training_option_names(train_model: Callable[..., Any]) -> tuple[str, ...]:
    signature = inspect.signature(train_model)
    return tuple(
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in _TRAINING_INFRA_OPTION_NAMES
    )


def inferred_training_options_validator(
    option_names: Sequence[str],
) -> model_def.ModelOptionValidator | None:
    if _INTERPOLATION_OPTION_NAMES <= set(option_names):
        from src.models.core import trigram_interpolation as interp

        return interp.validate_options
    return None


def model_definition(
    *,
    module_name: str,
    train_model: Callable[..., TrainingSummary],
    load_model: Callable[[Path], LoadedModel],
    summary_items: model_def.SummaryFormatter,
    training_option_names: Sequence[str] = (),
    query_lines: model_def.QueryFormatter | None = None,
    evaluation_items: model_def.SummaryFormatter | None = None,
    validate_training_options: model_def.ModelOptionValidator | None = None,
) -> model_def.ModelDefinition:
    name = model_name_from_module(module_name)
    model_label = model_label_from_name(name)

    def train(
        texts: Iterable[str],
        options: model_def.ModelOptions,
    ) -> TrainingSummary:
        stored_tokenizer_model = options.get("stored_tokenizer_model")
        training_options = {
            option_name: options[option_name]
            for option_name in training_option_names
            if option_name in options
        }
        return train_model(
            texts,
            tokenizer_model=resolve_tokenizer_model(options),
            output_path=resolve_output(options, model_suffix=name),
            stored_tokenizer_model=(
                Path(stored_tokenizer_model) if stored_tokenizer_model else None
            ),
            text_normalization=options["text_normalization"],
            **training_options,
        )

    def validate_options(options: model_def.ModelOptions) -> None:
        validate_tokenizer_model(options)
        if validate_training_options is not None:
            validate_training_options(options)

    def validate_query_options(options: model_def.ModelOptions) -> None:
        validate_model_path(options, model_suffix=name, label=model_label)

    def query(options: model_def.ModelOptions) -> QueryResult:
        model = load_model(resolve_model(options, model_suffix=name))
        return model.query(
            prompt=options["prompt"],
            max_tokens=options["max_tokens"],
            top_k=options["top_k"],
            decoding=options["decoding"],
            temperature=options["temperature"],
            seed=options["seed"],
        )

    def evaluate(
        texts: Iterable[str],
        options: model_def.ModelOptions,
    ) -> EvaluationSummary:
        model = load_model(resolve_model(options, model_suffix=name))
        return model.evaluate(texts, top_k=options["top_k"])

    return model_def.ModelDefinition(
        name=name,
        train=train,
        validate_options=validate_options,
        summary_items=summary_items,
        query=query,
        validate_query_options=validate_query_options,
        query_lines=query_lines or formatting.format_ngram_query,
        evaluate=evaluate,
        validate_evaluation_options=validate_query_options,
        evaluation_items=evaluation_items or standard_evaluation_items,
    )


def model_name_from_module(module_name: str) -> str:
    return module_name.rsplit(".", maxsplit=1)[-1].replace("_", "-")


def model_label_from_name(name: str) -> str:
    return name.replace("-", " ").capitalize()


def default_tokenizer_model(corpus: str) -> Path:
    return Path(
        "artifacts",
        "tokenizers",
        f"{corpus}-{tok_core.SENTENCEPIECE_ALGO}-1000.model",
    )


def default_ngram_output(
    corpus: str,
    model_suffix: str,
    tokenizer_model: object = None,
) -> Path:
    tokenizer_stem = (
        Path(tokenizer_model).stem
        if tokenizer_model
        else f"{corpus}-{tok_core.SENTENCEPIECE_ALGO}-1000"
    )
    return Path("artifacts", "models", f"{tokenizer_stem}-{model_suffix}.json")


def resolve_tokenizer_model(options: model_def.ModelOptions) -> Path:
    tokenizer_model = options.get("tokenizer_model")
    if tokenizer_model:
        return Path(tokenizer_model)
    return default_tokenizer_model(str(options["corpus"]))


def resolve_output(options: model_def.ModelOptions, *, model_suffix: str) -> Path:
    output = options.get("output")
    if output:
        return Path(output)
    return default_ngram_output(
        str(options["corpus"]),
        model_suffix,
        tokenizer_model=options.get("tokenizer_model"),
    )


def resolve_model(options: model_def.ModelOptions, *, model_suffix: str) -> Path:
    model_path = options.get("model_path")
    if model_path:
        return Path(model_path)
    return default_ngram_output(
        str(options["corpus"]),
        model_suffix,
        tokenizer_model=options.get("tokenizer_model"),
    )


def validate_tokenizer_model(options: model_def.ModelOptions) -> None:
    tokenizer_model = resolve_tokenizer_model(options)
    if not tokenizer_model.exists():
        raise model_def.ModelOptionError(
            f"Tokenizer model not found: {tokenizer_model}. "
            "Train it first with src.cli.tokenizer_training."
        )


def validate_model_path(
    options: model_def.ModelOptions,
    *,
    model_suffix: str,
    label: str,
) -> None:
    model_path = resolve_model(options, model_suffix=model_suffix)
    if not model_path.exists():
        raise model_def.ModelOptionError(
            f"{label} model not found: {model_path}. "
            "Train it first with src.cli.train."
        )


def standard_evaluation_items(
    summary: ngram.NgramEvaluationSummary,
) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        *evaluation_param_items(summary),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


def evaluation_param_items(
    summary: ngram.NgramEvaluationSummary,
) -> list[tuple[str, str]]:
    if has_interpolation_params(summary):
        return interpolation_items(summary)
    if hasattr(summary, "discount"):
        return [("Discount", f"{float(getattr(summary, 'discount')):.3f}")]
    return []


def has_interpolation_params(summary: ngram.NgramEvaluationSummary) -> bool:
    return all(
        hasattr(summary, name)
        for name in ("unigram_weight", "bigram_weight", "trigram_weight")
    )


def interpolation_items(
    summary: ngram.NgramEvaluationSummary,
) -> list[tuple[str, str]]:
    unigram_weight = float(getattr(summary, "unigram_weight"))
    bigram_weight = float(getattr(summary, "bigram_weight"))
    trigram_weight = float(getattr(summary, "trigram_weight"))
    beta_2 = getattr(summary, "beta_2", None)
    beta_3 = getattr(summary, "beta_3", None)
    if beta_2 is None or beta_3 is None:
        beta_2, beta_3 = betas_from_interpolation_weights(
            unigram_weight=unigram_weight,
            bigram_weight=bigram_weight,
            trigram_weight=trigram_weight,
        )

    return [
        (
            "Interpolation weights",
            formatting.format_interpolation_weights(
                unigram_weight=unigram_weight,
                bigram_weight=bigram_weight,
                trigram_weight=trigram_weight,
            ),
        ),
        (
            "Interpolation betas",
            f"beta_2={float(beta_2):.3f}, beta_3={float(beta_3):.3f}",
        ),
    ]


def betas_from_interpolation_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> tuple[float, float]:
    lower_weight = unigram_weight + bigram_weight  # lambda_1 + lambda_2.
    beta_2 = bigram_weight / lower_weight if lower_weight > 0 else 0.0
    return beta_2, trigram_weight
