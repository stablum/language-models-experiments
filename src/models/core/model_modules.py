"""Adapters from concrete model modules to the shared registry contract."""

from __future__ import annotations

import io
import inspect
import token
import tokenize
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TypeGuard, TypeVar

from src.corpora import normalization
from src.ml_core.models import definition as model_def
from src.models.core import formatting
from src.models.core import naming
from src.models.core import ngram
from src.models.core import trigram_interpolation as interp
from src.tokenizers import core as tok_core


LoadedT = TypeVar("LoadedT")
QueryT = TypeVar("QueryT")
EvalT = TypeVar("EvalT")
TrainSummaryT = TypeVar("TrainSummaryT", bound=ngram.NgramTrainingSummary)

REGISTRY_FLAG = "REGISTER_MODEL"
FIT_FN_NAME = "fit"

_FIT_INFRA_OPTION_NAMES = frozenset(
    (
        "tokenizer",
        "tokenizer_model",
        "output_path",
        "stored_tokenizer_model",
        "text_normalization",
        "validation_texts",
    )
)
_INTERPOLATION_OPTION_NAMES = frozenset(
    ("unigram_weight", "bigram_weight", "trigram_weight", "beta_2", "beta_3")
)


def registry_enabled(module_path: Path, *, module_name: str) -> bool:
    return registry_enabled_from_source(
        module_path.read_text(encoding="utf-8"),
        module_name=module_name,
    )


def registry_enabled_from_source(source: str, *, module_name: str) -> bool:
    depth = 0
    stmt: list[tokenize.TokenInfo] = []
    stream = tokenize.generate_tokens(io.StringIO(source).readline)

    try:
        for item in stream:
            if item.type == tokenize.INDENT:
                depth += 1
                continue
            if item.type == tokenize.DEDENT:
                depth -= 1
                continue
            if depth != 0 or item.type in (tokenize.COMMENT, tokenize.NL):
                continue
            if item.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                enabled = registry_flag_value(stmt, module_name=module_name)
                if enabled is not None:
                    return enabled
                stmt.clear()
                continue

            stmt.append(item)
    except tokenize.TokenError:
        # Let importlib surface syntax errors unless the opt-out flag was already seen.
        return True

    return True


def registry_flag_value(
    stmt: Sequence[tokenize.TokenInfo],
    *,
    module_name: str,
) -> bool | None:
    if (
        not stmt
        or stmt[0].type != tokenize.NAME
        or stmt[0].string != REGISTRY_FLAG
    ):
        return None

    eq_idx = next(
        (
            idx
            for idx, item in enumerate(stmt[1:], start=1)
            if item.exact_type == token.EQUAL
        ),
        None,
    )
    if eq_idx is None:
        return None

    value_toks = stmt[eq_idx + 1 :]
    if (
        len(value_toks) == 1
        and value_toks[0].type == tokenize.NAME
        and value_toks[0].string in ("False", "True")
    ):
        return value_toks[0].string == "True"

    raise TypeError(
        f"{module_name}.{REGISTRY_FLAG} must be assigned a literal bool."
    )


def model_definition_from_module(module: ModuleType) -> model_def.ModelDefinition | None:
    fit_model = get_module_callable(module, FIT_FN_NAME)
    load_model = get_module_callable(module, "load")
    summary_items = get_module_callable(module, "format_summary")
    if fit_model is None or load_model is None or summary_items is None:
        return None

    fit_opt_names = infer_fit_option_names(fit_model)
    validate_fit_opts = get_module_callable(module, "validate_fit_options")

    return model_definition(
        module_name=module.__name__,
        fit_model=fit_model,
        load_model=load_model,
        summary_items=summary_items,
        fit_option_names=fit_opt_names,
        uses_validation_data=accepts_keyword(fit_model, "validation_texts"),
        query_lines=get_module_callable(module, "format_query"),
        evaluation_items=get_module_callable(module, "format_evaluation"),
        validate_fit_options=(
            validate_fit_opts
            or inferred_fit_options_validator(fit_opt_names)
        ),
    )


def get_module_callable(module: ModuleType, name: str) -> Callable[..., Any] | None:
    fn = getattr(module, name, None)
    if fn is not None and not callable(fn):
        raise TypeError(f"{module.__name__}.{name} must be callable")
    return fn


def infer_fit_option_names(fit_model: Callable[..., Any]) -> tuple[str, ...]:
    sig = inspect.signature(fit_model)
    return tuple(
        name
        for name, param in sig.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in _FIT_INFRA_OPTION_NAMES
    )


def accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Return whether a callable can receive a named keyword argument."""
    param = inspect.signature(fn).parameters.get(name)
    if param is None:
        return False
    return param.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def inferred_fit_options_validator(
    opt_names: Sequence[str],
) -> model_def.ModelOptionValidator | None:
    if _INTERPOLATION_OPTION_NAMES <= set(opt_names):
        return interp.validate_options
    return None


def model_definition(
    *,
    module_name: str,
    fit_model: Callable[..., ngram.TrainingResult[Any]],
    load_model: Callable[[Path], LoadedT],
    summary_items: model_def.SummaryFormatter,
    fit_option_names: Sequence[str] = (),
    uses_validation_data: bool = False,
    query_lines: model_def.QueryFormatter | None = None,
    evaluation_items: model_def.SummaryFormatter | None = None,
    validate_fit_options: model_def.ModelOptionValidator | None = None,
) -> model_def.ModelDefinition:
    name = model_name_from_module(module_name)
    model_label = model_label_from_name(name)

    def fit(
        data: model_def.ModelFitData,
        opts: model_def.ModelOptions,
    ) -> ngram.NgramTrainingSummary:
        """Fit a concrete model and persist its learned artifact payload."""
        stored_tok_model = opts.get("stored_tokenizer_model")
        tok_model = resolve_tokenizer_model(opts)
        out_path = resolve_output(opts, model_suffix=name)
        tokenizer = tok_core.load_tokenizer(tok_model)
        fit_opts = {
            opt_name: opts[opt_name]
            for opt_name in fit_option_names
            if opt_name in opts
        }
        fit_kwargs: dict[str, object] = {
            "tokenizer": tokenizer,
            "text_normalization": opts["text_normalization"],
            **fit_opts,
        }
        if uses_validation_data:
            fit_kwargs["validation_texts"] = data.validation_items

        result = fit_model(data.train_items, **fit_kwargs)
        return save_training_result(
            result,
            module_name=module_name,
            output_path=out_path,
            tokenizer_model=tok_model,
            stored_tokenizer_model=(
                Path(stored_tok_model) if stored_tok_model else None
            ),
            tokenizer=tokenizer,
            text_normalization=opts["text_normalization"],
        )

    def validate_options(opts: model_def.ModelOptions) -> None:
        validate_tokenizer_model(opts)
        if validate_fit_options is not None:
            validate_fit_options(opts)

    def validate_query_options(opts: model_def.ModelOptions) -> None:
        validate_model_path(opts, model_suffix=name, label=model_label)

    def query(opts: model_def.ModelOptions) -> QueryT:
        model = load_model(resolve_model(opts, model_suffix=name))
        return model.query(
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
        texts: Iterable[str],
        opts: model_def.ModelOptions,
    ) -> EvalT:
        model = load_model(resolve_model(opts, model_suffix=name))
        return model.evaluate(texts, top_k=opts["top_k"])

    return model_def.ModelDefinition(
        name=name,
        fit=fit,
        uses_validation_data=uses_validation_data,
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
    return naming.registered_name_from_module(module_name)


def model_label_from_name(name: str) -> str:
    return naming.label_from_registered_name(name)


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
        raise ValueError(f"Model payload defines adapter-owned fields: {fields}")

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
    tok_stem = (
        Path(tokenizer_model).stem
        if tokenizer_model
        else f"{corpus}-{tok_core.SENTENCEPIECE_ALGO}-1000"
    )
    return Path("artifacts", "models", f"{tok_stem}-{model_suffix}.json")


def resolve_tokenizer_model(opts: model_def.ModelOptions) -> Path:
    tok_model = opts.get("tokenizer_model")
    if tok_model:
        return Path(tok_model)
    return default_tokenizer_model(str(opts["corpus"]))


def resolve_output(opts: model_def.ModelOptions, *, model_suffix: str) -> Path:
    out = opts.get("output")
    if out:
        return Path(out)
    return default_ngram_output(
        str(opts["corpus"]),
        model_suffix,
        tokenizer_model=opts.get("tokenizer_model"),
    )


def resolve_model(opts: model_def.ModelOptions, *, model_suffix: str) -> Path:
    model_path = opts.get("model_path")
    if model_path:
        return Path(model_path)
    return default_ngram_output(
        str(opts["corpus"]),
        model_suffix,
        tokenizer_model=opts.get("tokenizer_model"),
    )


def validate_tokenizer_model(opts: model_def.ModelOptions) -> None:
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
    model_path = resolve_model(opts, model_suffix=model_suffix)
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
        return interp.items(summary)
    if hasattr(summary, "discount"):
        return [("Discount", f"{float(getattr(summary, 'discount')):.3f}")]
    return []


def has_interpolation_params(
    summary: ngram.NgramEvaluationSummary,
) -> TypeGuard[interp.InterpolationSummary]:
    return all(
        hasattr(summary, name)
        for name in ("unigram_weight", "bigram_weight", "trigram_weight")
    )
