"""Shared model registry contract."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


ModelOptions = Mapping[str, Any]
ModelTrainer = Callable[[Iterable[str], ModelOptions], Any]
ModelQuery = Callable[[ModelOptions], Any]
ModelEvaluator = Callable[[Iterable[str], ModelOptions], Any]
ModelOptionValidator = Callable[[ModelOptions], None]
SummaryFormatter = Callable[[Any], list[tuple[str, str]]]
QueryFormatter = Callable[[Any], list[str]]


class ModelOptionError(ValueError):
    """Raised when model-specific options are invalid."""


@dataclass(frozen=True)
class ModelDefinition:
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
