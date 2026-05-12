"""Shared model registry contract."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from src.ml_core import cfg as core_cfg


ModelOptions = Mapping[str, Any]
ModelTrainer = Callable[[Iterable[Any], ModelOptions], Any]
ModelQuery = Callable[[ModelOptions], Any]
ModelEvaluator = Callable[[Iterable[Any], ModelOptions], Any]
ModelOptionValidator = Callable[[ModelOptions], None]
SummaryFormatter = Callable[[Any], list[tuple[str, str]]]
QueryFormatter = Callable[[Any], list[str]]


class ModelOptionError(ValueError):
    """Raised when model-specific options are invalid."""


class ModelDefinition(core_cfg.BaseCfg):
    name: str
    train: ModelTrainer
    validate_options: ModelOptionValidator
    summary_items: SummaryFormatter
    query: ModelQuery | None = None
    validate_query_options: ModelOptionValidator | None = None
    query_lines: QueryFormatter | None = None
    evaluate: ModelEvaluator | None = None
    validate_evaluation_options: ModelOptionValidator | None = None
    evaluation_items: SummaryFormatter | None = None
