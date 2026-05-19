"""Resolve corpus CLI overrides against registered corpus defaults."""

from __future__ import annotations

from src.corpora import registry as corpora_registry
from src.ml_core import cfg as core_cfg


class CorpusSourceCfg(core_cfg.FrozenBaseCfg):
    """Resolved corpus source values shared by CLI entry points."""

    corpus: str
    definition: corpora_registry.CorpusDefinition
    dataset_id: str
    source_split: str | None
    text_column: str


def resolve(
    *,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    text_column: str | None,
) -> CorpusSourceCfg:
    definition = corpora_registry.get_corpus(corpus)
    return CorpusSourceCfg(
        corpus=corpus,
        definition=definition,
        dataset_id=dataset_id or definition.dataset_id,
        source_split=source_split if source_split is not None else definition.split,
        text_column=text_column or definition.text_column,
    )
