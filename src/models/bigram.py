"""Very small token-level autoregressive bigram model training."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


@dataclass(frozen=True)
class BigramTrainingSummary:
    output_path: Path
    vocab_size: int
    sequence_count: int
    token_count: int
    transition_count: int


def iter_token_sequences(
    texts: Iterable[str],
    processor: spm.SentencePieceProcessor,
) -> Iterator[list[int]]:
    bos_id = processor.bos_id()
    eos_id = processor.eos_id()

    for text in texts:
        for line in text.splitlines():
            sentence = line.strip()
            if not sentence:
                continue

            token_ids = processor.encode(sentence, out_type=int)
            if bos_id >= 0:
                token_ids = [bos_id, *token_ids]
            if eos_id >= 0:
                token_ids.append(eos_id)

            if len(token_ids) >= 2:
                yield token_ids


def train_bigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    smoothing: float = 0.1,
) -> BigramTrainingSummary:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    sequence_count = 0
    token_count = 0
    transition_count = 0

    for token_ids in iter_token_sequences(texts, processor):
        sequence_count += 1
        token_count += len(token_ids)

        for previous_id, next_id in zip(token_ids, token_ids[1:]):
            transitions[previous_id][next_id] += 1
            transition_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "schema_version": 1,
        "model_type": "autoregressive_bigram",
        "tokenizer_model": str(tokenizer_model),
        "vocab_size": processor.get_piece_size(),
        "smoothing": smoothing,
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "unk_id": processor.unk_id(),
        "pieces": [processor.id_to_piece(index) for index in range(processor.get_piece_size())],
        "sequence_count": sequence_count,
        "token_count": token_count,
        "transition_count": transition_count,
        "transitions": {
            str(previous_id): sorted(next_counts.items())
            for previous_id, next_counts in sorted(transitions.items())
        },
    }
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    return BigramTrainingSummary(
        output_path=output_path,
        vocab_size=processor.get_piece_size(),
        sequence_count=sequence_count,
        token_count=token_count,
        transition_count=transition_count,
    )
