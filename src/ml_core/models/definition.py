"""Shared model runtime contracts."""

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


ModelOptionValidator = Callable[[ModelOptions], None]
SummaryFormatter = Callable[[Any], list[tuple[str, str]]]
QueryFormatter = Callable[[Any], list[str]]


class ModelOptionError(ValueError):
    """Raised when model-specific options are invalid."""
