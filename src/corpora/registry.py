"""Registry for corpus-specific dataset loaders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.corpora import babylm, europarl, tinystories


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


CORPORA = {
    "babylm-2026-strict-small": CorpusDefinition(
        name="babylm-2026-strict-small",
        dataset_id=babylm.DATASET_ID,
        split=babylm.DEFAULT_SPLIT,
        text_column=babylm.TEXT_COLUMN,
        load=babylm.load_dataset,
        available_splits=babylm.AVAILABLE_SPLITS,
        split_note=babylm.SPLIT_NOTE,
    ),
    "europarl": CorpusDefinition(
        name="europarl",
        dataset_id=europarl.DATASET_ID,
        split=europarl.DEFAULT_SPLIT,
        text_column=europarl.TEXT_COLUMN,
        load=europarl.load_dataset,
        available_splits=europarl.AVAILABLE_SPLITS,
        split_note=europarl.SPLIT_NOTE,
    ),
    "tinystories": CorpusDefinition(
        name="tinystories",
        dataset_id=tinystories.DATASET_ID,
        split=tinystories.DEFAULT_SPLIT,
        text_column=tinystories.TEXT_COLUMN,
        load=tinystories.load_dataset,
        available_splits=tinystories.AVAILABLE_SPLITS,
        split_note=tinystories.SPLIT_NOTE,
    ),
}


def default_corpus_name() -> str:
    return next(iter(CORPORA))


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
