"""Runtime operations for registered language-model modules."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from src.corpora import normalization
from src.ml_core.models import definition as model_def
from src.models.core import ngram
from src.models.core import model_modules
from src.tokenizers import core as tok_core


TrainSummaryT = TypeVar(
    "TrainSummaryT",
    bound=ngram.NgramTrainingSummary,
)  # Preserve each model family's concrete summary type.


def fit(
    model: model_modules.RegisteredModel,
    data: model_def.ModelFitData,
    opts: model_def.ModelOptions,
) -> ngram.NgramTrainingSummary:
    """Call module-level ``fit`` and persist its payload, not a ``Model`` dump.

    Example: training writes JSON first; later ``load`` hydrates a runtime model.
    """
    stored_tok_model = opts.get("stored_tokenizer_model")
    tok_model = resolve_tokenizer_model(opts)
    out_path = resolve_output(opts, model_suffix=model.name)
    tokenizer = tok_core.load_tokenizer(tok_model)
    fit_opts = {
        opt_name: opts[opt_name]
        for opt_name in model.fit_option_names
        if opt_name in opts
    }
    fit_kwargs: dict[str, object] = {
        "tokenizer": tokenizer,
        "text_normalization": opts["text_normalization"],
        **fit_opts,
    }
    if model.uses_validation_data:
        fit_kwargs["validation_texts"] = data.validation_items

    result = model.fit_fn(data.train_items, **fit_kwargs)
    return save_training_result(
        result,
        module_name=model.module.__name__,
        output_path=out_path,
        tokenizer_model=tok_model,
        stored_tokenizer_model=Path(stored_tok_model) if stored_tok_model else None,
        tokenizer=tokenizer,
        text_normalization=opts["text_normalization"],
    )


def query(
    model: model_modules.RegisteredModel,
    opts: model_def.ModelOptions,
) -> Any:
    """Query a persisted registered model artifact with generation options."""
    loaded_model = model.load(resolve_model(opts, model_suffix=model.name))
    return loaded_model.query(
        ngram.NgramQueryCfg(
            prompt=opts["prompt"],
            max_tokens=opts["max_tokens"],
            top_k=opts["top_k"],
            decoding=opts["decoding"],
            temperature=opts["temperature"],
            seed=opts["seed"],
        ),
    )


def evaluate(
    model: model_modules.RegisteredModel,
    texts: Iterable[str],
    opts: model_def.ModelOptions,
) -> Any:
    """Evaluate a persisted registered model artifact over text rows."""
    loaded_model = model.load(resolve_model(opts, model_suffix=model.name))
    return loaded_model.evaluate(texts, top_k=opts["top_k"])


def validate_fit_options(
    model: model_modules.RegisteredModel,
    opts: model_def.ModelOptions,
) -> None:
    """Validate tokenizer existence and model-owned fitting hyperparameters."""
    validate_tokenizer_model(opts)
    if model.fit_options_validator is not None:
        model.fit_options_validator(opts)


def validate_query_options(
    model: model_modules.RegisteredModel,
    opts: model_def.ModelOptions,
) -> None:
    """Validate the model artifact path used by query runs."""
    validate_model_path(opts, model_suffix=model.name, label=model.label)


def validate_evaluation_options(
    model: model_modules.RegisteredModel,
    opts: model_def.ModelOptions,
) -> None:
    """Validate the model artifact path used by evaluation runs."""
    validate_query_options(model, opts)


def query_lines(
    model: model_modules.RegisteredModel,
    result: Any,
) -> tuple[str, ...]:
    """Format query result lines through the registered model module."""
    if not model.supports_query:
        return ()
    return tuple(model.query_formatter(result))


def save_training_result(
    result: ngram.TrainingResult[TrainSummaryT],
    *,
    module_name: str,
    output_path: Path,
    tokenizer_model: Path,
    stored_tokenizer_model: Path | None,
    tokenizer: tok_core.TokenizerCodec,
    text_normalization: normalization.TextNormalization,
) -> TrainSummaryT:
    """Wrap model-owned fields in the standard artifact envelope.

    Example: merges schema/tokenizer metadata; not faithful object serialization.
    """
    schema_payload = ngram.model_schema_payload(module_name)
    tok_payload = ngram.tokenizer_model_payload(
        tokenizer,
        tokenizer_model=tokenizer_model,
        stored_tokenizer_model=stored_tokenizer_model,
        text_normalization=text_normalization,
    )
    owned_fields = frozenset((*schema_payload, *tok_payload))
    overlap_fields = owned_fields & result.payload.keys()
    if overlap_fields:
        fields = ", ".join(sorted(overlap_fields))
        raise ValueError(f"Model payload defines runtime-owned fields: {fields}")

    payload = {
        **schema_payload,
        **tok_payload,
        **result.payload,
    }
    ngram.write_json_model_payload(output_path, payload)

    summary = result.summary
    summary.output_path = output_path
    summary.tokenizer_model = tokenizer_model
    summary.vocab_size = tokenizer.vocab_size
    summary.text_normalization = text_normalization
    return summary


def default_tokenizer_model(corpus: str) -> Path:
    """Return the conventional tokenizer artifact path for a corpus."""
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
    """Return the conventional n-gram model artifact path."""
    tok_stem = (
        Path(tokenizer_model).stem
        if tokenizer_model
        else f"{corpus}-{tok_core.SENTENCEPIECE_ALGO}-1000"
    )
    return Path("artifacts", "models", f"{tok_stem}-{model_suffix}.json")


def resolve_tokenizer_model(opts: model_def.ModelOptions) -> Path:
    """Resolve the tokenizer model path from options or corpus defaults."""
    tok_model = opts.get("tokenizer_model")
    if tok_model:
        return Path(tok_model)
    return default_tokenizer_model(str(opts["corpus"]))


def resolve_output(opts: model_def.ModelOptions, *, model_suffix: str) -> Path:
    """Resolve the model output artifact path from options or defaults."""
    out = opts.get("output")
    if out:
        return Path(out)
    return default_ngram_output(
        str(opts["corpus"]),
        model_suffix,
        tokenizer_model=opts.get("tokenizer_model"),
    )


def resolve_model(opts: model_def.ModelOptions, *, model_suffix: str) -> Path:
    """Resolve an existing model artifact path from options or defaults."""
    model_path = opts.get("model_path")
    if model_path:
        return Path(model_path)
    return default_ngram_output(
        str(opts["corpus"]),
        model_suffix,
        tokenizer_model=opts.get("tokenizer_model"),
    )


def validate_tokenizer_model(opts: model_def.ModelOptions) -> None:
    """Reject fitting options that point at a missing tokenizer artifact."""
    tok_model = resolve_tokenizer_model(opts)
    if not tok_model.exists():
        raise model_def.ModelOptionError(
            f"Tokenizer model not found: {tok_model}. "
            "Train it first with src.cli.tokenizer_training."
        )


def validate_model_path(
    opts: model_def.ModelOptions,
    *,
    model_suffix: str,
    label: str,
) -> None:
    """Reject query/evaluation options that point at a missing model artifact."""
    model_path = resolve_model(opts, model_suffix=model_suffix)
    if not model_path.exists():
        raise model_def.ModelOptionError(
            f"{label} model not found: {model_path}. "
            "Train it first with src.cli.train."
        )
