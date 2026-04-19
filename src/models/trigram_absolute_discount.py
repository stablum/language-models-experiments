"""Absolute-discount token-level autoregressive trigram model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

from src.models.ngram import NgramEvaluationSummary, candidate_token_count, load_pieces
from src.models.ngram import resolve_stored_path
from src.models.trigram_common import (
    BaseTrigramModel,
    Context,
    TrigramEvaluationRow,
    TrigramQueryResult,
    collect_trigram_counts,
    parse_bigram_transitions,
    parse_trigram_transitions,
    parse_unigram_counts,
    trigram_counts_payload,
)


@dataclass(frozen=True)
class AbsoluteDiscountTrigramTrainingSummary:
    output_path: Path
    tokenizer_model: Path
    vocab_size: int
    sequence_count: int
    token_count: int
    unigram_count: int
    bigram_transition_count: int
    trigram_transition_count: int
    discount: float


@dataclass(frozen=True)
class AbsoluteDiscountTrigramEvaluationSummary(NgramEvaluationSummary):
    discount: float


@dataclass(frozen=True)
class AbsoluteDiscountTrigramModel(BaseTrigramModel):
    model_path: Path
    tokenizer_model: Path
    processor: spm.SentencePieceProcessor
    vocab_size: int
    smoothing: float
    discount: float
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    unigram_counts: dict[int, int]
    unigram_total: int
    bigram_transitions: dict[int, tuple[tuple[int, int], ...]]
    trigram_transitions: dict[Context, tuple[tuple[int, int], ...]]

    def evaluation_summary(self, **kwargs: Any) -> AbsoluteDiscountTrigramEvaluationSummary:
        return AbsoluteDiscountTrigramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            discount=self.discount,
            **kwargs,
        )

    def transition_probability(
        self,
        next_id: int,
        context: Context,
        *,
        row: TrigramEvaluationRow | None = None,
        bigram_counts: dict[int, int] | None = None,
        trigram_counts: dict[int, int] | None = None,
        bigram_total: int | None = None,
        trigram_total: int | None = None,
    ) -> float:
        if next_id == self.bos_id:
            return 0.0

        previous_id = context[1]
        if row is not None:
            bigram_counts = row.bigram_counts
            trigram_counts = row.trigram_counts
            bigram_total = row.bigram_total
            trigram_total = row.trigram_total
        else:
            if bigram_counts is None:
                bigram_counts = dict(self.bigram_transitions.get(previous_id, ()))
            if trigram_counts is None:
                trigram_counts = dict(self.trigram_transitions.get(context, ()))
            if bigram_total is None:
                bigram_total = sum(bigram_counts.values())
            if trigram_total is None:
                trigram_total = sum(trigram_counts.values())

        return self.trigram_probability(
            next_id,
            previous_id=previous_id,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )

    def trigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        bigram_counts: dict[int, int],
        trigram_counts: dict[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> float:
        lower_order_probability = self.bigram_probability(
            token_id,
            previous_id=previous_id,
            counts=bigram_counts,
            total=bigram_total,
        )
        if trigram_total <= 0:
            return lower_order_probability

        observed_count = trigram_counts.get(token_id, 0)
        discounted_probability = max(observed_count - self.discount, 0.0) / trigram_total
        backoff_weight = self.discount * len(trigram_counts) / trigram_total
        return discounted_probability + backoff_weight * lower_order_probability

    def bigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        counts: dict[int, int] | None = None,
        total: int | None = None,
    ) -> float:
        if counts is None:
            counts = dict(self.bigram_transitions.get(previous_id, ()))
        if total is None:
            total = sum(counts.values())

        lower_order_probability = self.unigram_probability(token_id)
        if total <= 0:
            return lower_order_probability

        observed_count = counts.get(token_id, 0)
        discounted_probability = max(observed_count - self.discount, 0.0) / total
        backoff_weight = self.discount * len(counts) / total
        return discounted_probability + backoff_weight * lower_order_probability

    def unigram_probability(self, token_id: int) -> float:
        denominator = (
            self.unigram_total
            + self.smoothing * candidate_token_count(self.vocab_size, self.bos_id)
        )
        if denominator <= 0:
            return 0.0
        return (self.unigram_counts.get(token_id, 0) + self.smoothing) / denominator


def load_absolute_discount_trigram_model(model_path: Path) -> AbsoluteDiscountTrigramModel:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    if data.get("model_type") != "absolute_discount_trigram":
        raise ValueError(f"Not an absolute-discount trigram model: {model_path}")

    tokenizer_model = resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    vocab_size = int(data["vocab_size"])

    return AbsoluteDiscountTrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        vocab_size=vocab_size,
        smoothing=float(data["smoothing"]),
        discount=float(data["discount"]),
        bos_id=int(data["bos_id"]),
        eos_id=int(data["eos_id"]),
        unk_id=int(data["unk_id"]),
        pieces=load_pieces(data, processor, vocab_size),
        unigram_counts=parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=parse_bigram_transitions(data),
        trigram_transitions=parse_trigram_transitions(data),
    )


def train_absolute_discount_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    smoothing: float = 0.1,
    discount: float = 0.75,
) -> AbsoluteDiscountTrigramTrainingSummary:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    counts = collect_trigram_counts(texts, processor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "schema_version": 1,
        "model_type": "absolute_discount_trigram",
        "tokenizer_model": str(tokenizer_model),
        "vocab_size": processor.get_piece_size(),
        "smoothing": smoothing,
        "discount": discount,
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "unk_id": processor.unk_id(),
        "pieces": [processor.id_to_piece(index) for index in range(processor.get_piece_size())],
        **trigram_counts_payload(counts),
    }
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return AbsoluteDiscountTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        sequence_count=counts.sequence_count,
        token_count=counts.token_count,
        unigram_count=counts.unigram_count,
        bigram_transition_count=counts.bigram_transition_count,
        trigram_transition_count=counts.trigram_transition_count,
        discount=discount,
    )
