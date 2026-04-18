"""Registry for corpus-specific dataset loaders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from language_models_experiments.corpora.babylm_2026_strict_small import (
    DATASET_ID as BABYLM_DATASET_ID,
    DEFAULT_SPLIT as BABYLM_DEFAULT_SPLIT,
    TEXT_COLUMN as BABYLM_TEXT_COLUMN,
    load_babylm_dataset,
)


CorpusLoader = Callable[..., Any]


@dataclass(frozen=True)
class CorpusDefinition:
    name: str
    dataset_id: str
    split: str
    text_column: str
    load: CorpusLoader


DEFAULT_CORPUS_NAME = "babylm-2026-strict-small"

CORPORA = {
    DEFAULT_CORPUS_NAME: CorpusDefinition(
        name=DEFAULT_CORPUS_NAME,
        dataset_id=BABYLM_DATASET_ID,
        split=BABYLM_DEFAULT_SPLIT,
        text_column=BABYLM_TEXT_COLUMN,
        load=load_babylm_dataset,
    )
}


def corpus_names() -> tuple[str, ...]:
    return tuple(CORPORA)


def get_corpus(name: str) -> CorpusDefinition:
    return CORPORA[name]
