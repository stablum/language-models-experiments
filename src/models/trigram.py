"""Interpolated token-level autoregressive trigram model."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

from src.models.ngram import (
    DecodingMode,
    NgramEvaluationSummary,
    NgramPrediction,
    NgramQueryResult,
    candidate_token_count,
    candidate_token_ids,
    decode_continuation,
    encode_prompt,
    iter_sentencepiece_token_sequences,
    load_pieces,
    resolve_stored_path,
    seeded_rng,
    select_next_token,
)


Context = tuple[int, int]


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


TrigramPrediction = NgramPrediction
TrigramQueryResult = NgramQueryResult


@dataclass(frozen=True)
class TrigramEvaluationSummary(NgramEvaluationSummary):
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float


@dataclass(frozen=True)
class TrigramEvaluationRow:
    bigram_counts: dict[int, int]
    trigram_counts: dict[int, int]
    bigram_total: int
    trigram_total: int
    greedy_token_id: int
    top_k_token_ids: frozenset[int]


@dataclass(frozen=True)
class TrigramModel:
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

    def encode_prompt(self, prompt: str) -> list[int]:
        return encode_prompt(self.processor, prompt)

    def next_token_predictions(
        self,
        context: Context,
        *,
        top_k: int,
    ) -> list[TrigramPrediction]:
        candidate_ids = candidate_token_ids(self.vocab_size, self.bos_id)
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        predictions = [
            TrigramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=trigram_counts.get(token_id, 0),
                probability=self.transition_probability(token_id, context),
            )
            for token_id in candidate_ids
        ]
        predictions.sort(key=lambda prediction: (-prediction.probability, prediction.token_id))
        return predictions[:top_k] if top_k > 0 else predictions

    def query(
        self,
        *,
        prompt: str = "",
        max_tokens: int = 80,
        top_k: int = 10,
        decoding: DecodingMode = "sample",
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> TrigramQueryResult:
        prompt_token_ids = self.encode_prompt(prompt)
        context = self.context_for_tokens(prompt_token_ids)
        next_token_predictions = self.next_token_predictions(context, top_k=top_k)
        rng = seeded_rng(seed)
        token_ids = list(prompt_token_ids)
        generated_token_ids: list[int] = []

        for _ in range(max_tokens):
            next_id = select_next_token(
                self.next_token_predictions(context, top_k=0),
                eos_id=self.eos_id,
                decoding=decoding,
                rng=rng,
                temperature=temperature,
            )
            if next_id == self.eos_id:
                break

            generated_token_ids.append(next_id)
            token_ids.append(next_id)
            context = (context[1], next_id)

        prompt_text = self.processor.decode(prompt_token_ids)
        generated_text = self.processor.decode(token_ids)
        continuation_text = decode_continuation(
            self.processor,
            generated_text=generated_text,
            prompt_text=prompt_text,
            generated_token_ids=generated_token_ids,
        )

        return TrigramQueryResult(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            decoding=decoding,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            unk_id=self.unk_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            continuation_text=continuation_text,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            token_ids=token_ids,
            next_token_predictions=next_token_predictions,
        )

    def evaluate(
        self,
        texts: Iterable[str],
        *,
        top_k: int = 5,
    ) -> TrigramEvaluationSummary:
        row_cache: dict[Context, TrigramEvaluationRow] = {}
        sequence_count = 0
        token_count = 0
        transition_count = 0
        correct_next_token_count = 0
        top_k_correct_next_token_count = 0
        negative_log_likelihood = 0.0
        zero_probability_count = 0

        for token_ids in iter_trigram_token_sequences(texts, self.processor):
            sequence_count += 1
            token_count += len(token_ids)

            for previous_previous_id, previous_id, next_id in zip(
                token_ids,
                token_ids[1:],
                token_ids[2:],
            ):
                transition_count += 1
                context = (previous_previous_id, previous_id)
                row = row_cache.get(context)
                if row is None:
                    row = self.evaluation_row(context, top_k=top_k)
                    row_cache[context] = row

                if next_id == row.greedy_token_id:
                    correct_next_token_count += 1
                if next_id in row.top_k_token_ids:
                    top_k_correct_next_token_count += 1

                probability = self.transition_probability(next_id, context, row=row)
                if probability <= 0:
                    zero_probability_count += 1
                else:
                    negative_log_likelihood -= math.log(probability)

        return TrigramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            top_k=top_k,
            sequence_count=sequence_count,
            token_count=token_count,
            transition_count=transition_count,
            correct_next_token_count=correct_next_token_count,
            top_k_correct_next_token_count=top_k_correct_next_token_count,
            negative_log_likelihood=negative_log_likelihood,
            zero_probability_count=zero_probability_count,
            unigram_weight=self.unigram_weight,
            bigram_weight=self.bigram_weight,
            trigram_weight=self.trigram_weight,
        )

    def evaluation_row(
        self,
        context: Context,
        *,
        top_k: int,
    ) -> TrigramEvaluationRow:
        previous_id = context[1]
        bigram_counts = dict(self.bigram_transitions.get(previous_id, ()))
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        bigram_total = sum(bigram_counts.values())
        trigram_total = sum(trigram_counts.values())
        ranked_token_ids = self.ranked_token_ids(
            context,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )
        fallback_token_id = self.eos_id if self.eos_id >= 0 else 0
        greedy_token_id = ranked_token_ids[0] if ranked_token_ids else fallback_token_id
        return TrigramEvaluationRow(
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
            greedy_token_id=greedy_token_id,
            top_k_token_ids=frozenset(ranked_token_ids[:top_k]) if top_k > 0 else frozenset(),
        )

    def ranked_token_ids(
        self,
        context: Context,
        *,
        bigram_counts: dict[int, int],
        trigram_counts: dict[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> list[int]:
        return sorted(
            candidate_token_ids(self.vocab_size, self.bos_id),
            key=lambda token_id: (
                -self.transition_probability(
                    token_id,
                    context,
                    bigram_counts=bigram_counts,
                    trigram_counts=trigram_counts,
                    bigram_total=bigram_total,
                    trigram_total=trigram_total,
                ),
                token_id,
            ),
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
            + self.smoothing * candidate_token_count(self.vocab_size, self.bos_id)
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
        denominator = total + self.smoothing * candidate_token_count(
            self.vocab_size,
            self.bos_id,
        )
        if denominator <= 0:
            return 0.0
        return (counts.get(token_id, 0) + self.smoothing) / denominator

    def context_for_tokens(self, token_ids: list[int]) -> Context:
        bos_id = self.bos_id if self.bos_id >= 0 else 0
        if len(token_ids) >= 2:
            return token_ids[-2], token_ids[-1]
        if len(token_ids) == 1:
            return bos_id, token_ids[-1]
        return bos_id, bos_id


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

    tokenizer_model = resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    vocab_size = int(data["vocab_size"])
    pieces = load_pieces(data, processor, vocab_size)
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
        pieces=pieces,
        unigram_counts={
            int(token_id): int(count)
            for token_id, count in data["unigrams"]
        },
        unigram_total=int(data["unigram_count"]),
        bigram_transitions={
            int(previous_id): tuple(
                (int(next_id), int(count))
                for next_id, count in next_counts
            )
            for previous_id, next_counts in data["bigram_transitions"].items()
        },
        trigram_transitions={
            parse_context_key(context_key): tuple(
                (int(next_id), int(count))
                for next_id, count in next_counts
            )
            for context_key, next_counts in data["trigram_transitions"].items()
        },
    )


def iter_trigram_token_sequences(
    texts: Iterable[str],
    processor: spm.SentencePieceProcessor,
) -> Iterator[list[int]]:
    yield from iter_sentencepiece_token_sequences(
        texts,
        processor,
        bos_count=2,
        min_length=3,
    )


def train_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    smoothing: float = 0.1,
    unigram_weight: float = 0.1,
    bigram_weight: float = 0.3,
    trigram_weight: float = 0.6,
) -> TrigramTrainingSummary:
    normalized_weights = normalize_interpolation_weights(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
    )
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    unigram_counts: Counter[int] = Counter()
    bigram_transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    trigram_transitions: defaultdict[Context, Counter[int]] = defaultdict(Counter)
    sequence_count = 0
    token_count = 0
    bigram_transition_count = 0
    trigram_transition_count = 0

    for token_ids in iter_trigram_token_sequences(texts, processor):
        sequence_count += 1
        token_count += len(token_ids)
        unigram_counts.update(token_ids[2:])

        for previous_id, next_id in zip(token_ids[1:], token_ids[2:]):
            bigram_transitions[previous_id][next_id] += 1
            bigram_transition_count += 1

        for previous_previous_id, previous_id, next_id in zip(
            token_ids,
            token_ids[1:],
            token_ids[2:],
        ):
            trigram_transitions[(previous_previous_id, previous_id)][next_id] += 1
            trigram_transition_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "schema_version": 1,
        "model_type": "interpolated_trigram",
        "tokenizer_model": str(tokenizer_model),
        "vocab_size": processor.get_piece_size(),
        "smoothing": smoothing,
        "interpolation_weights": {
            "unigram": normalized_weights[0],
            "bigram": normalized_weights[1],
            "trigram": normalized_weights[2],
        },
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "unk_id": processor.unk_id(),
        "pieces": [processor.id_to_piece(index) for index in range(processor.get_piece_size())],
        "sequence_count": sequence_count,
        "token_count": token_count,
        "unigram_count": sum(unigram_counts.values()),
        "bigram_transition_count": bigram_transition_count,
        "trigram_transition_count": trigram_transition_count,
        "unigrams": sorted(unigram_counts.items()),
        "bigram_transitions": {
            str(previous_id): sorted(next_counts.items())
            for previous_id, next_counts in sorted(bigram_transitions.items())
        },
        "trigram_transitions": {
            context_key(previous_previous_id, previous_id): sorted(next_counts.items())
            for (
                previous_previous_id,
                previous_id,
            ), next_counts in sorted(trigram_transitions.items())
        },
    }
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return TrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        sequence_count=sequence_count,
        token_count=token_count,
        unigram_count=sum(unigram_counts.values()),
        bigram_transition_count=bigram_transition_count,
        trigram_transition_count=trigram_transition_count,
        unigram_weight=normalized_weights[0],
        bigram_weight=normalized_weights[1],
        trigram_weight=normalized_weights[2],
    )


def context_key(previous_previous_id: int, previous_id: int) -> str:
    return f"{previous_previous_id},{previous_id}"


def parse_context_key(key: str) -> Context:
    previous_previous_id, previous_id = key.split(",", maxsplit=1)
    return int(previous_previous_id), int(previous_id)
