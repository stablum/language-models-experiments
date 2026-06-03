"""Discovery-time model-module conformity and registered-model metadata."""

from __future__ import annotations

import inspect
import io
import token
import tokenize
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, TypeGuard

from src.ml_core import cfg as core_cfg
from src.ml_core.models import definition as model_def
from src.models.core import formatting
from src.models.core import naming
from src.models.core import ngram
from src.models.core import trigram_interpolation as interp


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


class RegisteredModel(core_cfg.BaseCfg):
    """Represent one discovered model module through dynamic registry metadata."""

    module: ModuleType

    @property
    def name(self) -> str:
        """Return the CLI registry name derived from the module path."""
        return naming.registered_name_from_module(self.module.__name__)

    @property
    def label(self) -> str:
        """Return a human label derived from the registered model name."""
        return naming.label_from_registered_name(self.name)

    @property
    def fit_fn(self) -> Callable[..., Any]:
        """Return the concrete module fitting strategy."""
        return required_module_callable(self.module, FIT_FN_NAME)

    @property
    def load_fn(self) -> Callable[[Path], Any]:
        """Return the concrete module artifact loader."""
        return required_module_callable(self.module, "load")

    @property
    def summary_formatter(self) -> model_def.SummaryFormatter:
        """Return the concrete module training-summary formatter."""
        return required_module_callable(self.module, "format_summary")

    @property
    def query_formatter(self) -> model_def.QueryFormatter:
        """Return the module query formatter, falling back to n-gram output."""
        return get_module_callable(
            self.module,
            "format_query",
        ) or formatting.format_ngram_query

    @property
    def evaluation_formatter(self) -> model_def.SummaryFormatter:
        """Return the module evaluation formatter, falling back to n-gram metrics."""
        return get_module_callable(
            self.module,
            "format_evaluation",
        ) or standard_evaluation_items

    @property
    def fit_option_names(self) -> tuple[str, ...]:
        """Infer model hyperparameter names from the module fit signature."""
        return infer_fit_option_names(self.fit_fn)

    @property
    def fit_options_validator(self) -> model_def.ModelOptionValidator | None:
        """Return the module or inferred validator for model hyperparameters."""
        module_validator = get_module_callable(self.module, "validate_fit_options")
        return module_validator or inferred_fit_options_validator(
            self.fit_option_names,
        )

    @property
    def uses_validation_data(self) -> bool:
        """Return whether fit accepts the validation-text partition."""
        return accepts_keyword(self.fit_fn, "validation_texts")

    @property
    def supports_query(self) -> bool:
        """Return whether the module model class exposes query behavior."""
        return self.model_has_method("query")

    @property
    def supports_evaluation(self) -> bool:
        """Return whether the module model class exposes evaluation behavior."""
        return self.model_has_method("evaluate")

    def model_has_method(self, method_name: str) -> bool:
        """Check a method on the declared module Model class."""
        model_cls = getattr(self.module, "Model", None)
        return isinstance(model_cls, type) and callable(
            getattr(model_cls, method_name, None)
        )

    def load(self, model_path: Path) -> Any:
        """Hydrate a persisted model artifact through the module loader."""
        return self.load_fn(model_path)


def registry_enabled(module_path: Path, *, module_name: str) -> bool:
    """Return whether a model module source file opts into discovery."""
    return registry_enabled_from_source(
        module_path.read_text(encoding="utf-8"),
        module_name=module_name,
    )


def registry_enabled_from_source(source: str, *, module_name: str) -> bool:
    """Read a top-level REGISTER_MODEL bool without importing the module."""
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
    """Parse one top-level statement as a REGISTER_MODEL bool assignment."""
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


def registered_model_from_module(module: ModuleType) -> RegisteredModel | None:
    """Adapt a conforming concrete model module into a registered model."""
    if (
        get_module_callable(module, FIT_FN_NAME) is None
        or get_module_callable(module, "load") is None
        or get_module_callable(module, "format_summary") is None
    ):
        return None

    return RegisteredModel(module=module)


def get_module_callable(module: ModuleType, name: str) -> Callable[..., Any] | None:
    """Return an optional callable exported by a model module."""
    fn = getattr(module, name, None)
    if fn is not None and not callable(fn):
        raise TypeError(f"{module.__name__}.{name} must be callable")
    return fn


def required_module_callable(module: ModuleType, name: str) -> Callable[..., Any]:
    """Return a required callable exported by a conforming model module."""
    fn = get_module_callable(module, name)
    if fn is None:
        raise TypeError(f"{module.__name__}.{name} is required for registration")
    return fn


def infer_fit_option_names(fit_fn: Callable[..., Any]) -> tuple[str, ...]:
    """Infer model-owned keyword-only hyperparameter names from fit(...)."""
    sig = inspect.signature(fit_fn)
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
    """Infer a validator from known coupled hyperparameter sets."""
    if _INTERPOLATION_OPTION_NAMES <= set(opt_names):
        return interp.validate_options
    return None


def standard_evaluation_items(
    summary: ngram.NgramEvaluationSummary,
) -> list[tuple[str, str]]:
    """Format standard n-gram evaluation artifact and metric rows."""
    return [
        *ngram.base_evaluation_items(summary),
        *evaluation_param_items(summary),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


def evaluation_param_items(
    summary: ngram.NgramEvaluationSummary,
) -> list[tuple[str, str]]:
    """Format model-family hyperparameters stored on evaluation summaries."""
    if has_interpolation_params(summary):
        return interp.items(summary)
    if hasattr(summary, "discount"):
        return [("Discount", f"{float(getattr(summary, 'discount')):.3f}")]
    return []


def has_interpolation_params(
    summary: ngram.NgramEvaluationSummary,
) -> TypeGuard[interp.InterpolationSummary]:
    """Return whether a summary carries lambda/beta interpolation fields."""
    return all(
        hasattr(summary, name)
        for name in ("unigram_weight", "bigram_weight", "trigram_weight")
    )
