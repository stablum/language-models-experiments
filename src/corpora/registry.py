"""Registry for corpus-specific dataset loaders."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

from src.corpora import babylm, europarl, tinystories
from src.ml_core import cfg as core_cfg


CorpusLoader = Callable[..., Any]


class CorpusDefinition(core_cfg.FrozenBaseCfg):
    name: str
    dataset_id: str
    split: str | None
    text_column: str
    load: CorpusLoader
    available_splits: tuple[str, ...] = ()
    split_note: str | None = None


def _definition(name: str, module: ModuleType) -> CorpusDefinition:
    return CorpusDefinition(
        name=name,
        dataset_id=module.DATASET_ID,
        split=module.DEFAULT_SPLIT,
        text_column=module.TEXT_COLUMN,
        load=module.load_dataset,
        available_splits=module.AVAILABLE_SPLITS,
        split_note=module.SPLIT_NOTE,
    )


CORPORA = {
    name: _definition(name, module)
    for name, module in (
        ("babylm-2026-strict-small", babylm),
        ("europarl", europarl),
        ("tinystories", tinystories),
    )
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
