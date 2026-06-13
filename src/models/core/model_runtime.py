"""Runtime operations for registered language-model modules."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from src.corpora import normalization
from src.ml_core.models import definition as model_def
from src.models.core import ngram
from src.models.core import model_modules
from src.models.core import token_sequences
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
    tok_space = ngram.token_space_from_tokenizer(tokenizer)
    train_corpus = model_token_corpus(
        data.train_items,
        tokenizer,
        model=model,
        text_normalization=opts["text_normalization"],
    )
    fit_opts = {
        opt_name: opts[opt_name]
        for opt_name in model.fit_option_names
        if opt_name in opts
    }
    fit_kwargs: dict[str, object] = {
        **fit_opts,
    }
    if model.uses_validation_corpus:
        fit_kwargs["validation_corpus"] = optional_model_token_corpus(
            data.validation_items,
            tokenizer,
            model=model,
            text_normalization=opts["text_normalization"],
        )

    result = model.fit_fn(train_corpus, **fit_kwargs)
    return save_training_result(
        result,
        module_name=model.module.__name__,
        output_path=out_path,
        tokenizer_model=tok_model,
        stored_tokenizer_model=Path(stored_tok_model) if stored_tok_model else None,
        tokenizer=tokenizer,
        token_space=tok_space,
        text_normalization=opts["text_normalization"],
    )


def query(
    model: model_modules.RegisteredModel,
    opts: model_def.ModelOptions,
) -> Any:
    """Query a persisted registered model artifact with generation options."""
    loaded_model = model.load(resolve_model(opts, model_suffix=model.name))
    tokenizer = load_model_tokenizer(loaded_model)
    return query_token_model(
        loaded_model,
        tokenizer,
        cfg=ngram.NgramQueryCfg(
            prompt=str(opts["prompt"]),
            max_tokens=int(opts["max_tokens"]),
            top_k=int(opts["top_k"]),
            decoding=opts["decoding"],
            temperature=float(opts["temperature"]),
            seed=opts["seed"],
        )
    )


def evaluate(
    model: model_modules.RegisteredModel,
    texts: Iterable[str],
    opts: model_def.ModelOptions,
) -> Any:
    """Evaluate a persisted registered model artifact over text rows."""
    loaded_model = model.load(resolve_model(opts, model_suffix=model.name))
    tokenizer = load_model_tokenizer(loaded_model)
    corpus = model_token_corpus(
        texts,
        tokenizer,
        model=model,
        text_normalization=loaded_model.text_normalization,
    )
    return loaded_model.evaluate_token_corpus(corpus, top_k=opts["top_k"])


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
    token_space: ngram.TokenSpace,
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
    summary.vocab_size = token_space.vocab_size
    summary.text_normalization = text_normalization
    return summary


def model_token_corpus(
    texts: Iterable[str],
    tokenizer: tok_core.TokenizerCodec,
    *,
    model: model_modules.RegisteredModel,
    text_normalization: normalization.TextNormalization,
) -> token_sequences.TokenCorpus:
    """Adapt raw text rows into a token-space corpus for one model family."""
    seqs = tok_core.iter_token_sequences(
        texts,
        tokenizer,
        bos_count=model.context_length,
        min_length=model.context_length + 1,
        text_normalization=text_normalization,
    )
    return token_sequences.TokenCorpus(
        seqs,
        vocab_size=tokenizer.vocab_size,
    )


def optional_model_token_corpus(
    texts: Iterable[str] | None,
    tokenizer: tok_core.TokenizerCodec,
    *,
    model: model_modules.RegisteredModel,
    text_normalization: normalization.TextNormalization,
) -> token_sequences.TokenCorpus | None:
    """Tokenize an optional validation stream only when the model requests it."""
    if texts is None:
        return None
    return model_token_corpus(
        texts,
        tokenizer,
        model=model,
        text_normalization=text_normalization,
    )


def load_model_tokenizer(model: ngram.BaseNgramModel) -> tok_core.TokenizerCodec:
    """Load the text adapter stored beside a persisted token-space model."""
    return tok_core.load_tokenizer(
        model.tokenizer_model,
        tokenizer_algo=model.tokenizer_algo,
    )


def query_token_model(
    model: ngram.BaseNgramModel,
    tokenizer: tok_core.TokenizerCodec,
    *,
    cfg: ngram.NgramQueryCfg,
) -> ngram.NgramQueryResult:
    """Run text query adaptation around a token-space next-token model."""
    prompt_ids = tok_core.encode_prompt(
        tokenizer,
        cfg.prompt,
        text_normalization=model.text_normalization,
    )
    context = model.context_for_tokens(prompt_ids)
    next_preds = model.next_token_predictions(
        context,
        top_k=cfg.top_k,
    )
    gen_top_k = ngram.generation_prediction_top_k(
        decoding=cfg.decoding,
        temperature=cfg.temperature,
    )
    rng = random.Random(cfg.seed)  # nosec B311
    all_ids = list(prompt_ids)  # ids = prompt plus generated token IDs.
    gen_ids: list[int] = []  # gen = generated continuation token IDs.

    for _ in range(cfg.max_tokens):
        next_id = ngram.select_next_token(
            model.next_token_predictions(context, top_k=gen_top_k),
            eos_id=model.eos_id,
            decoding=cfg.decoding,
            rng=rng,
            temperature=cfg.temperature,
        )
        if next_id == model.eos_id:
            break

        gen_ids.append(next_id)
        all_ids.append(next_id)
        context = model.advance_context(context, next_id)

    prompt_text = tokenizer.decode(prompt_ids)
    generated_text = tokenizer.decode(all_ids)
    continuation_text = tok_core.decode_continuation(
        tokenizer,
        generated_text=generated_text,
        prompt_text=prompt_text,
        generated_token_ids=gen_ids,
    )

    return ngram.NgramQueryResult(
        model_path=model.model_path,
        tokenizer_model=model.tokenizer_model,
        decoding=cfg.decoding,
        bos_id=model.bos_id,
        eos_id=model.eos_id,
        unk_id=model.unk_id,
        prompt=cfg.prompt,
        prompt_token_ids=prompt_ids,
        continuation_text=continuation_text,
        generated_text=generated_text,
        generated_token_ids=gen_ids,
        token_ids=all_ids,
        next_token_predictions=next_preds,
        text_normalization=model.text_normalization,
    )


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
