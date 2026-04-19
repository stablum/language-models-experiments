"""Very small token-level autoregressive bigram model training and querying."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


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
    prompt: str
    prompt_token_ids: list[int]
    generated_text: str
    generated_token_ids: list[int]
    token_ids: list[int]
    next_token_predictions: list[BigramPrediction]


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
            next_id = self.sample_next_token(previous_id, rng=rng, temperature=temperature)
            if next_id == self.eos_id:
                break

            generated_token_ids.append(next_id)
            token_ids.append(next_id)
            previous_id = next_id

        return BigramQueryResult(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            generated_text=self.processor.decode(token_ids),
            generated_token_ids=generated_token_ids,
            token_ids=token_ids,
            next_token_predictions=next_token_predictions,
        )

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

    def _candidate_token_ids(self) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id in range(self.vocab_size)
            if token_id != self.bos_id
        )


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
