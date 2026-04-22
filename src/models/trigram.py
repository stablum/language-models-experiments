"""Interpolated token-level autoregressive trigram model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TextNormalization
from src.models import ngram
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
class TrigramTrainingSummary:
    output_path: Path
    tokenizer_model: Path
    vocab_size: int
    sequence_count: int
    token_count: int
    unigram_count: int
    bigram_transition_count: int
    trigram_transition_count: int
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float
    text_normalization: str


@dataclass(frozen=True)
class TrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float


@dataclass(frozen=True)
class TrigramModel(BaseTrigramModel):
    model_path: Path
    tokenizer_model: Path
    processor: spm.SentencePieceProcessor
    vocab_size: int
    smoothing: float
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    unigram_counts: dict[int, int]
    unigram_total: int
    bigram_transitions: dict[int, tuple[tuple[int, int], ...]]
    trigram_transitions: dict[Context, tuple[tuple[int, int], ...]]
    text_normalization: str = "none"

    def evaluation_summary(self, **kwargs: Any) -> TrigramEvaluationSummary:
        return TrigramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            unigram_weight=self.unigram_weight,
            bigram_weight=self.bigram_weight,
            trigram_weight=self.trigram_weight,
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

        return (
            self.unigram_weight * self.unigram_probability(next_id)
            + self.bigram_weight * self.conditional_probability(
                next_id,
                counts=bigram_counts,
                total=bigram_total,
            )
            + self.trigram_weight * self.conditional_probability(
                next_id,
                counts=trigram_counts,
                total=trigram_total,
            )
        )

    def unigram_probability(self, token_id: int) -> float:
        denominator = (
            self.unigram_total
            + self.smoothing * ngram.candidate_token_count(self.vocab_size, self.bos_id)
        )
        if denominator <= 0:
            return 0.0
        return (self.unigram_counts.get(token_id, 0) + self.smoothing) / denominator

    def conditional_probability(
        self,
        token_id: int,
        *,
        counts: dict[int, int],
        total: int,
    ) -> float:
        denominator = total + self.smoothing * ngram.candidate_token_count(
            self.vocab_size,
            self.bos_id,
        )
        if denominator <= 0:
            return 0.0
        return (counts.get(token_id, 0) + self.smoothing) / denominator


def normalize_interpolation_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> tuple[float, float, float]:
    total = unigram_weight + bigram_weight + trigram_weight
    if total <= 0:
        raise ValueError("At least one interpolation weight must be positive.")
    return unigram_weight / total, bigram_weight / total, trigram_weight / total


def load_trigram_model(model_path: Path) -> TrigramModel:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    if data.get("model_type") != "interpolated_trigram":
        raise ValueError(f"Not an interpolated trigram model: {model_path}")

    tokenizer_model = ngram.resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    vocab_size = int(data["vocab_size"])
    weights = data["interpolation_weights"]

    return TrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        vocab_size=vocab_size,
        smoothing=float(data["smoothing"]),
        unigram_weight=float(weights["unigram"]),
        bigram_weight=float(weights["bigram"]),
        trigram_weight=float(weights["trigram"]),
        bos_id=int(data["bos_id"]),
        eos_id=int(data["eos_id"]),
        unk_id=int(data["unk_id"]),
        pieces=ngram.load_pieces(data, processor, vocab_size),
        unigram_counts=parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=parse_bigram_transitions(data),
        trigram_transitions=parse_trigram_transitions(data),
        text_normalization=str(data.get("text_normalization", "none")),
    )


def train_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    smoothing: float = 0.1,
    unigram_weight: float = 0.1,
    bigram_weight: float = 0.3,
    trigram_weight: float = 0.6,
    text_normalization: TextNormalization = DEFAULT_TEXT_NORMALIZATION,
) -> TrigramTrainingSummary:
    normalized_weights = normalize_interpolation_weights(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
    )
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    counts = collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "schema_version": 1,
        "model_type": "interpolated_trigram",
        "tokenizer_model": str(stored_tokenizer_model or tokenizer_model),
        "vocab_size": processor.get_piece_size(),
        "smoothing": smoothing,
        "text_normalization": text_normalization,
        "interpolation_weights": {
            "unigram": normalized_weights[0],
            "bigram": normalized_weights[1],
            "trigram": normalized_weights[2],
        },
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

    return TrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        sequence_count=counts.sequence_count,
        token_count=counts.token_count,
        unigram_count=counts.unigram_count,
        bigram_transition_count=counts.bigram_transition_count,
        trigram_transition_count=counts.trigram_transition_count,
        unigram_weight=normalized_weights[0],
        bigram_weight=normalized_weights[1],
        trigram_weight=normalized_weights[2],
        text_normalization=text_normalization,
    )
