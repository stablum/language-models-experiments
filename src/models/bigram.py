"""Very small token-level autoregressive bigram model training and querying."""

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


@dataclass(frozen=True)
class BigramTrainingSummary:
    output_path: Path
    tokenizer_model: Path
    vocab_size: int
    sequence_count: int
    token_count: int
    transition_count: int


@dataclass(frozen=True)
class BigramPrediction:
    token_id: int
    piece: str
    count: int
    probability: float


@dataclass(frozen=True)
class BigramQueryResult:
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
    next_token_predictions: list[BigramPrediction]


@dataclass(frozen=True)
class BigramEvaluationSummary:
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


@dataclass(frozen=True)
class BigramEvaluationRow:
    counts: dict[int, int]
    denominator: float
    greedy_token_id: int
    top_k_token_ids: frozenset[int]


@dataclass(frozen=True)
class BigramModel:
    model_path: Path
    tokenizer_model: Path
    processor: spm.SentencePieceProcessor
    vocab_size: int
    smoothing: float
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    transitions: dict[int, tuple[tuple[int, int], ...]]

    def encode_prompt(self, prompt: str) -> list[int]:
        if not prompt:
            return []
        return self.processor.encode(prompt, out_type=int)

    def next_token_predictions(
        self,
        previous_id: int,
        *,
        top_k: int,
    ) -> list[BigramPrediction]:
        candidate_ids = self._candidate_token_ids()
        observed = dict(self.transitions.get(previous_id, ()))
        denominator = sum(observed.get(token_id, 0) for token_id in candidate_ids)
        denominator += self.smoothing * len(candidate_ids)

        if denominator <= 0:
            return []

        predictions = [
            BigramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=observed.get(token_id, 0),
                probability=(observed.get(token_id, 0) + self.smoothing) / denominator,
            )
            for token_id in candidate_ids
            if observed.get(token_id, 0) > 0 or self.smoothing > 0
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
    ) -> BigramQueryResult:
        prompt_token_ids = self.encode_prompt(prompt)
        previous_id = prompt_token_ids[-1] if prompt_token_ids else self.bos_id
        next_token_predictions = self.next_token_predictions(previous_id, top_k=top_k)
        rng = random.Random(seed)
        token_ids = list(prompt_token_ids)
        generated_token_ids: list[int] = []

        for _ in range(max_tokens):
            next_id = self.next_generated_token(
                previous_id,
                decoding=decoding,
                rng=rng,
                temperature=temperature,
            )
            if next_id == self.eos_id:
                break

            generated_token_ids.append(next_id)
            token_ids.append(next_id)
            previous_id = next_id

        prompt_text = self.processor.decode(prompt_token_ids)
        generated_text = self.processor.decode(token_ids)
        continuation_text = self.decode_continuation(
            generated_text=generated_text,
            prompt_text=prompt_text,
            generated_token_ids=generated_token_ids,
        )

        return BigramQueryResult(
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
    ) -> BigramEvaluationSummary:
        candidate_ids = self._candidate_token_ids()
        candidate_id_set = set(candidate_ids)
        row_cache: dict[int, BigramEvaluationRow] = {}
        sequence_count = 0
        token_count = 0
        transition_count = 0
        correct_next_token_count = 0
        top_k_correct_next_token_count = 0
        negative_log_likelihood = 0.0
        zero_probability_count = 0

        for token_ids in iter_token_sequences(texts, self.processor):
            sequence_count += 1
            token_count += len(token_ids)

            for previous_id, next_id in zip(token_ids, token_ids[1:]):
                transition_count += 1
                row = row_cache.get(previous_id)
                if row is None:
                    row = self.evaluation_row(
                        previous_id,
                        candidate_ids=candidate_ids,
                        candidate_id_set=candidate_id_set,
                        top_k=top_k,
                    )
                    row_cache[previous_id] = row

                if next_id == row.greedy_token_id:
                    correct_next_token_count += 1
                if next_id in row.top_k_token_ids:
                    top_k_correct_next_token_count += 1

                probability = self.transition_probability(
                    next_id,
                    row=row,
                    candidate_id_set=candidate_id_set,
                )
                if probability <= 0:
                    zero_probability_count += 1
                else:
                    negative_log_likelihood -= math.log(probability)

        return BigramEvaluationSummary(
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
        )

    def evaluation_row(
        self,
        previous_id: int,
        *,
        candidate_ids: tuple[int, ...],
        candidate_id_set: set[int],
        top_k: int,
    ) -> BigramEvaluationRow:
        counts = {
            token_id: count
            for token_id, count in self.transitions.get(previous_id, ())
            if token_id in candidate_id_set
        }
        denominator = sum(counts.values()) + self.smoothing * len(candidate_ids)
        ranked_token_ids = self.ranked_token_ids(
            counts=counts,
            candidate_ids=candidate_ids,
        )
        fallback_token_id = self.eos_id if self.eos_id >= 0 else 0
        greedy_token_id = ranked_token_ids[0] if ranked_token_ids else fallback_token_id
        return BigramEvaluationRow(
            counts=counts,
            denominator=denominator,
            greedy_token_id=greedy_token_id,
            top_k_token_ids=frozenset(ranked_token_ids[:top_k]) if top_k > 0 else frozenset(),
        )

    def ranked_token_ids(
        self,
        *,
        counts: dict[int, int],
        candidate_ids: tuple[int, ...],
    ) -> list[int]:
        if self.smoothing > 0:
            return sorted(
                candidate_ids,
                key=lambda token_id: (-(counts.get(token_id, 0) + self.smoothing), token_id),
            )
        return sorted(counts, key=lambda token_id: (-counts[token_id], token_id))

    def transition_probability(
        self,
        next_id: int,
        *,
        row: BigramEvaluationRow,
        candidate_id_set: set[int],
    ) -> float:
        if row.denominator <= 0 or next_id not in candidate_id_set:
            return 0.0
        return (row.counts.get(next_id, 0) + self.smoothing) / row.denominator

    def next_generated_token(
        self,
        previous_id: int,
        *,
        decoding: DecodingMode,
        rng: random.Random,
        temperature: float,
    ) -> int:
        if decoding == "most-probable":
            return self.most_probable_next_token(previous_id)
        if decoding == "sample":
            return self.sample_next_token(
                previous_id,
                rng=rng,
                temperature=temperature,
            )
        raise ValueError(f"Unsupported decoding mode: {decoding}")

    def most_probable_next_token(self, previous_id: int) -> int:
        predictions = self.next_token_predictions(previous_id, top_k=1)
        if not predictions:
            return self.eos_id if self.eos_id >= 0 else 0
        return predictions[0].token_id

    def sample_next_token(
        self,
        previous_id: int,
        *,
        rng: random.Random,
        temperature: float,
    ) -> int:
        predictions = self.next_token_predictions(previous_id, top_k=0)
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


def divide_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def load_bigram_model(model_path: Path) -> BigramModel:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    if data.get("model_type") != "autoregressive_bigram":
        raise ValueError(f"Not an autoregressive bigram model: {model_path}")

    tokenizer_model = resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    vocab_size = int(data["vocab_size"])
    stored_pieces = data.get("pieces")
    pieces = (
        tuple(str(piece) for piece in stored_pieces)
        if stored_pieces
        else tuple(processor.id_to_piece(index) for index in range(vocab_size))
    )
    transitions = {
        int(previous_id): tuple(
            (int(next_id), int(count))
            for next_id, count in next_counts
        )
        for previous_id, next_counts in data["transitions"].items()
    }

    return BigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        vocab_size=vocab_size,
        smoothing=float(data["smoothing"]),
        bos_id=int(data["bos_id"]),
        eos_id=int(data["eos_id"]),
        unk_id=int(data["unk_id"]),
        pieces=pieces,
        transitions=transitions,
    )


def resolve_stored_path(stored_path: Path, model_path: Path) -> Path:
    if stored_path.is_absolute() or stored_path.exists():
        return stored_path

    model_relative_path = model_path.parent / stored_path
    if model_relative_path.exists():
        return model_relative_path

    return stored_path


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
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return BigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        sequence_count=sequence_count,
        token_count=token_count,
        transition_count=transition_count,
    )
