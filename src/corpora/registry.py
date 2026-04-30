"""Registry for corpus-specific dataset loaders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.corpora.babylm_2026_strict_small import (
    AVAILABLE_SPLITS as BABYLM_AVAILABLE_SPLITS,
    DATASET_ID as BABYLM_DATASET_ID,
    DEFAULT_SPLIT as BABYLM_DEFAULT_SPLIT,
    SPLIT_NOTE as BABYLM_SPLIT_NOTE,
    TEXT_COLUMN as BABYLM_TEXT_COLUMN,
    load_babylm_dataset,
)
from src.corpora.tinystories import (
    AVAILABLE_SPLITS as TINYSTORIES_AVAILABLE_SPLITS,
    DATASET_ID as TINYSTORIES_DATASET_ID,
    DEFAULT_SPLIT as TINYSTORIES_DEFAULT_SPLIT,
    SPLIT_NOTE as TINYSTORIES_SPLIT_NOTE,
    TEXT_COLUMN as TINYSTORIES_TEXT_COLUMN,
    load_tinystories_dataset,
)


CorpusLoader = Callable[..., Any]


@dataclass(frozen=True)
class CorpusDefinition:
    name: str
    dataset_id: str
    split: str | None
    text_column: str
    load: CorpusLoader
    available_splits: tuple[str, ...] = ()
    split_note: str | None = None


DEFAULT_CORPUS_NAME = "babylm-2026-strict-small"
TINYSTORIES_CORPUS_NAME = "tinystories"

CORPORA = {
    DEFAULT_CORPUS_NAME: CorpusDefinition(
        name=DEFAULT_CORPUS_NAME,
        dataset_id=BABYLM_DATASET_ID,
        split=BABYLM_DEFAULT_SPLIT,
        text_column=BABYLM_TEXT_COLUMN,
        load=load_babylm_dataset,
        available_splits=BABYLM_AVAILABLE_SPLITS,
        split_note=BABYLM_SPLIT_NOTE,
    ),
    TINYSTORIES_CORPUS_NAME: CorpusDefinition(
        name=TINYSTORIES_CORPUS_NAME,
        dataset_id=TINYSTORIES_DATASET_ID,
        split=TINYSTORIES_DEFAULT_SPLIT,
        text_column=TINYSTORIES_TEXT_COLUMN,
        load=load_tinystories_dataset,
        available_splits=TINYSTORIES_AVAILABLE_SPLITS,
        split_note=TINYSTORIES_SPLIT_NOTE,
    ),
}


def corpus_names() -> tuple[str, ...]:
    return tuple(CORPORA)


def get_corpus(name: str) -> CorpusDefinition:
    return CORPORA[name]


def split_note_for(
    corpus_definition: CorpusDefinition,
    *,
    dataset_id_override: str | None,
) -> str | None:
    if dataset_id_override is not None:
        return None
    if corpus_definition.split_note is None:
        return None
    return corpus_definition.split_note
