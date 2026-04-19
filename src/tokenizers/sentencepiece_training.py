"""SentencePiece tokenizer training helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import sentencepiece as spm


def iter_sentencepiece_sentences(texts: Iterable[str]) -> Iterator[str]:
    for text in texts:
        for line in text.splitlines():
            sentence = line.strip()
            if sentence:
                yield sentence


def train_sentencepiece(
    texts: Iterable[str],
    *,
    output_prefix: Path,
    vocab_size: int = 1000,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    hard_vocab_limit: bool = True,
    max_sentence_length: int | None = None,
) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    trainer_options = {
        "sentence_iterator": iter_sentencepiece_sentences(texts),
        "model_prefix": str(output_prefix),
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "hard_vocab_limit": hard_vocab_limit,
    }
    if max_sentence_length is not None:
        trainer_options["max_sentence_length"] = max_sentence_length

    spm.SentencePieceTrainer.train(**trainer_options)

    return output_prefix.with_suffix(".model"), output_prefix.with_suffix(".vocab")
