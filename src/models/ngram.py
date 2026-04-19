"""Shared helpers for small token-level n-gram models."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import sentencepiece as spm


DecodingMode = Literal["sample", "most-probable"]


@dataclass(frozen=True)
class NgramPrediction:
    token_id: int
    piece: str
    count: int
    probability: float


@dataclass(frozen=True)
class NgramQueryResult:
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
    next_token_predictions: list[NgramPrediction]


@dataclass(frozen=True)
class NgramEvaluationSummary:
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

    @property
    def next_token_accuracy(self) -> float | None:
        return divide_or_none(self.correct_next_token_count, self.transition_count)

    @property
    def top_k_accuracy(self) -> float | None:
        return divide_or_none(self.top_k_correct_next_token_count, self.transition_count)

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


def divide_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def encode_prompt(processor: spm.SentencePieceProcessor, prompt: str) -> list[int]:
    if not prompt:
        return []
    return processor.encode(prompt, out_type=int)


def decode_continuation(
    processor: spm.SentencePieceProcessor,
    *,
    generated_text: str,
    prompt_text: str,
    generated_token_ids: list[int],
) -> str:
    if prompt_text and generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):]
    return processor.decode(generated_token_ids)


def candidate_token_ids(vocab_size: int, bos_id: int) -> tuple[int, ...]:
    return tuple(token_id for token_id in range(vocab_size) if token_id != bos_id)


def candidate_token_count(vocab_size: int, bos_id: int) -> int:
    return vocab_size - 1 if 0 <= bos_id < vocab_size else vocab_size


def select_next_token(
    predictions: Sequence[NgramPrediction],
    *,
    eos_id: int,
    decoding: DecodingMode,
    rng: random.Random,
    temperature: float,
) -> int:
    if decoding == "most-probable":
        return most_probable_token(predictions, eos_id=eos_id)
    if decoding == "sample":
        return sample_token(
            predictions,
            eos_id=eos_id,
            rng=rng,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported decoding mode: {decoding}")


def most_probable_token(
    predictions: Sequence[NgramPrediction],
    *,
    eos_id: int,
) -> int:
    if not predictions:
        return eos_id if eos_id >= 0 else 0
    return predictions[0].token_id


def sample_token(
    predictions: Sequence[NgramPrediction],
    *,
    eos_id: int,
    rng: random.Random,
    temperature: float,
) -> int:
    if not predictions:
        return eos_id if eos_id >= 0 else 0
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


def seeded_rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def load_pieces(
    data: dict[str, object],
    processor: spm.SentencePieceProcessor,
    vocab_size: int,
) -> tuple[str, ...]:
    stored_pieces = data.get("pieces")
    if stored_pieces:
        return tuple(str(piece) for piece in stored_pieces)
    return tuple(processor.id_to_piece(index) for index in range(vocab_size))


def resolve_stored_path(stored_path: Path, model_path: Path) -> Path:
    if stored_path.is_absolute() or stored_path.exists():
        return stored_path

    model_relative_path = model_path.parent / stored_path
    if model_relative_path.exists():
        return model_relative_path

    return stored_path


def iter_sentencepiece_token_sequences(
    texts: Iterable[str],
    processor: spm.SentencePieceProcessor,
    *,
    bos_count: int,
    min_length: int,
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
                token_ids = [bos_id] * bos_count + token_ids
            if eos_id >= 0:
                token_ids.append(eos_id)

            if len(token_ids) >= min_length:
                yield token_ids
