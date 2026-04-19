"""Interpolated token-level autoregressive trigram model."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import sentencepiece as spm


DecodingMode = Literal["sample", "most-probable"]
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


@dataclass(frozen=True)
class TrigramPrediction:
    token_id: int
    piece: str
    count: int
    probability: float


@dataclass(frozen=True)
class TrigramQueryResult:
    model_path: Path
    tokenizer_model: Path
    decoding: DecodingMode
    bos_id: int
    eos_id: int
    unk_id: int
    prompt: str
    prompt_token_ids: list[int]
    continuation_text: str
    generated_text: str
    generated_token_ids: list[int]
    token_ids: list[int]
    next_token_predictions: list[TrigramPrediction]


@dataclass(frozen=True)
class TrigramEvaluationSummary:
    model_path: Path
    tokenizer_model: Path
    top_k: int
    sequence_count: int
    token_count: int
    transition_count: int
    correct_next_token_count: int
    top_k_correct_next_token_count: int
    negative_log_likelihood: float
    zero_probability_count: int
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float

    @property
    def average_negative_log_likelihood(self) -> float | None:
        if self.transition_count == 0:
            return None
        if self.zero_probability_count:
            return math.inf
        return self.negative_log_likelihood / self.transition_count

    @property
    def cross_entropy_bits(self) -> float | None:
        average_nll = self.average_negative_log_likelihood
        if average_nll is None:
            return None
        return average_nll / math.log(2)

    @property
    def perplexity(self) -> float | None:
        average_nll = self.average_negative_log_likelihood
        if average_nll is None:
            return None
        if math.isinf(average_nll):
            return math.inf
        return math.exp(average_nll)


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
        if not prompt:
            return []
        return self.processor.encode(prompt, out_type=int)

    def next_token_predictions(
        self,
        context: Context,
        *,
        top_k: int,
    ) -> list[TrigramPrediction]:
        candidate_ids = self._candidate_token_ids()
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
        rng = random.Random(seed)
        token_ids = list(prompt_token_ids)
        generated_token_ids: list[int] = []

        for _ in range(max_tokens):
            next_id = self.next_generated_token(
                context,
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
        continuation_text = self.decode_continuation(
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
            self._candidate_token_ids(),
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
        denominator = self.unigram_total + self.smoothing * self._candidate_token_count()
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
        denominator = total + self.smoothing * self._candidate_token_count()
        if denominator <= 0:
            return 0.0
        return (counts.get(token_id, 0) + self.smoothing) / denominator

    def next_generated_token(
        self,
        context: Context,
        *,
        decoding: DecodingMode,
        rng: random.Random,
        temperature: float,
    ) -> int:
        if decoding == "most-probable":
            return self.most_probable_next_token(context)
        if decoding == "sample":
            return self.sample_next_token(
                context,
                rng=rng,
                temperature=temperature,
            )
        raise ValueError(f"Unsupported decoding mode: {decoding}")

    def most_probable_next_token(self, context: Context) -> int:
        predictions = self.next_token_predictions(context, top_k=1)
        if not predictions:
            return self.eos_id if self.eos_id >= 0 else 0
        return predictions[0].token_id

    def sample_next_token(
        self,
        context: Context,
        *,
        rng: random.Random,
        temperature: float,
    ) -> int:
        predictions = self.next_token_predictions(context, top_k=0)
        if not predictions:
            return self.eos_id if self.eos_id >= 0 else 0

        if temperature == 0:
            return predictions[0].token_id
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        weights = [prediction.probability ** (1 / temperature) for prediction in predictions]
        if not any(weights):
            return predictions[0].token_id

        return rng.choices(
            [prediction.token_id for prediction in predictions],
            weights=weights,
            k=1,
        )[0]

    def context_for_tokens(self, token_ids: list[int]) -> Context:
        bos_id = self.bos_id if self.bos_id >= 0 else 0
        if len(token_ids) >= 2:
            return token_ids[-2], token_ids[-1]
        if len(token_ids) == 1:
            return bos_id, token_ids[-1]
        return bos_id, bos_id

    def decode_continuation(
        self,
        *,
        generated_text: str,
        prompt_text: str,
        generated_token_ids: list[int],
    ) -> str:
        if prompt_text and generated_text.startswith(prompt_text):
            return generated_text[len(prompt_text):]
        return self.processor.decode(generated_token_ids)

    def _candidate_token_ids(self) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id in range(self.vocab_size)
            if token_id != self.bos_id
        )

    def _candidate_token_count(self) -> int:
        return self.vocab_size - 1 if 0 <= self.bos_id < self.vocab_size else self.vocab_size


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
    stored_pieces = data.get("pieces")
    pieces = (
        tuple(str(piece) for piece in stored_pieces)
        if stored_pieces
        else tuple(processor.id_to_piece(index) for index in range(vocab_size))
    )
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


def resolve_stored_path(stored_path: Path, model_path: Path) -> Path:
    if stored_path.is_absolute() or stored_path.exists():
        return stored_path

    model_relative_path = model_path.parent / stored_path
    if model_relative_path.exists():
        return model_relative_path

    return stored_path


def iter_trigram_token_sequences(
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
                token_ids = [bos_id, bos_id, *token_ids]
            if eos_id >= 0:
                token_ids.append(eos_id)

            if len(token_ids) >= 3:
                yield token_ids


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
