"""Shared model registry contract."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pydantic

from src.ml_core import cfg as core_cfg


ModelOptions = Mapping[str, Any]


class ModelFitData(core_cfg.BaseCfg):
    """Bundle split-aware data streams supplied to a model fitting entrypoint."""

    train_items: pydantic.SkipValidation[Iterable[Any]]
    validation_items: pydantic.SkipValidation[Iterable[Any] | None] = None


ModelFitter = Callable[[ModelFitData, ModelOptions], Any]
ModelQuery = Callable[[ModelOptions], Any]
ModelEvaluator = Callable[[Iterable[Any], ModelOptions], Any]
ModelOptionValidator = Callable[[ModelOptions], None]
SummaryFormatter = Callable[[Any], list[tuple[str, str]]]
QueryFormatter = Callable[[Any], list[str]]


class ModelOptionError(ValueError):
    """Raised when model-specific options are invalid."""


class ModelDefinition(core_cfg.BaseCfg):
    """Expose one imported model module through the shared pipeline contract."""

    name: str
    fit: ModelFitter
    uses_validation_data: bool = False
    validate_options: ModelOptionValidator
    summary_items: SummaryFormatter
    query: ModelQuery | None = None
    validate_query_options: ModelOptionValidator | None = None
    query_lines: QueryFormatter | None = None
    evaluate: ModelEvaluator | None = None
    validate_evaluation_options: ModelOptionValidator | None = None
    evaluation_items: SummaryFormatter | None = None
