"""Shared helpers for small token-level n-gram models."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pydantic

from src.corpora import normalization
from src.ml_core import json_io
from src.models.core import formatting
from src.tokenizers import core as tok_core


DecodingMode = Literal["sample", "most-probable"]


class NgramPydanticModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class NgramQueryCfg(NgramPydanticModel):
    prompt: str = ""
    max_tokens: int = pydantic.Field(default=80, ge=0)
    top_k: int = pydantic.Field(default=10, ge=1)
    decoding: DecodingMode = "sample"
    temperature: float = pydantic.Field(default=1.0, ge=0)
    seed: int | None = None


class FrozenNgramModel(NgramPydanticModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        validate_assignment=True,
    )


class NgramPrediction(NgramPydanticModel):
    token_id: int
    piece: str
    count: int
    probability: float


class NgramQueryResult(NgramPydanticModel):
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
    text_normalization: str = "none"


class NgramTrainingSummary(NgramPydanticModel):
    output_path: Path
    tokenizer_model: Path
    vocab_size: int = 0
    sequence_count: int = 0
    token_count: int = 0
    text_normalization: str = "none"


class NgramEvaluationSummary(NgramPydanticModel):
    model_path: Path
    tokenizer_model: Path
    top_k: int
    sequence_count: int = 0
    token_count: int = 0
    transition_count: int = 0
    correct_next_token_count: int = 0
    top_k_correct_next_token_count: int = 0
    negative_log_likelihood: float = 0.0
    zero_probability_count: int = 0
    text_normalization: str = "none"

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


class BaseNgramModel(NgramPydanticModel):
    model_path: Path
    tokenizer_model: Path
    tokenizer: tok_core.TokenizerCodec
    vocab_size: int
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    text_normalization: str = "none"
    _candidate_ids: tuple[int, ...] = pydantic.PrivateAttr(default=())
    _candidate_id_set: frozenset[int] = pydantic.PrivateAttr(default_factory=frozenset)

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        if not self._candidate_ids:
            self._candidate_ids = candidate_token_ids(self.vocab_size, self.bos_id)
        return self._candidate_ids

    @property
    def candidate_id_set(self) -> frozenset[int]:
        if not self._candidate_id_set:
            self._candidate_id_set = frozenset(self.candidate_ids)
        return self._candidate_id_set

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    def encode_prompt(self, prompt: str) -> list[int]:
        return tok_core.encode_prompt(
            self.tokenizer,
            prompt,
            text_normalization=self.text_normalization,
        )

    def context_for_tokens(self, token_ids: list[int]) -> Any:
        raise NotImplementedError

    def advance_context(self, context: Any, next_id: int) -> Any:
        raise NotImplementedError

    def next_token_predictions(
        self,
        context: Any,
        *,
        top_k: int,
    ) -> list[NgramPrediction]:
        raise NotImplementedError

    def query(self, cfg: NgramQueryCfg | None = None) -> NgramQueryResult:
        resolved_cfg = cfg or NgramQueryCfg()
        prompt_token_ids = self.encode_prompt(resolved_cfg.prompt)
        context = self.context_for_tokens(prompt_token_ids)
        next_token_predictions = self.next_token_predictions(
            context,
            top_k=resolved_cfg.top_k,
        )
        generation_top_k = generation_prediction_top_k(
            decoding=resolved_cfg.decoding,
            temperature=resolved_cfg.temperature,
        )
        rng = seeded_rng(resolved_cfg.seed)
        token_ids = list(prompt_token_ids)
        generated_token_ids: list[int] = []

        for _ in range(resolved_cfg.max_tokens):
            next_id = select_next_token(
                self.next_token_predictions(context, top_k=generation_top_k),
                eos_id=self.eos_id,
                decoding=resolved_cfg.decoding,
                rng=rng,
                temperature=resolved_cfg.temperature,
            )
            if next_id == self.eos_id:
                break

            generated_token_ids.append(next_id)
            token_ids.append(next_id)
            context = self.advance_context(context, next_id)

        prompt_text = self.tokenizer.decode(prompt_token_ids)
        generated_text = self.tokenizer.decode(token_ids)
        continuation_text = tok_core.decode_continuation(
            self.tokenizer,
            generated_text=generated_text,
            prompt_text=prompt_text,
            generated_token_ids=generated_token_ids,
        )

        return NgramQueryResult(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            decoding=resolved_cfg.decoding,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            unk_id=self.unk_id,
            prompt=resolved_cfg.prompt,
            prompt_token_ids=prompt_token_ids,
            continuation_text=continuation_text,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            token_ids=token_ids,
            next_token_predictions=next_token_predictions,
            text_normalization=self.text_normalization,
        )


def divide_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def candidate_token_ids(vocab_size: int, bos_id: int) -> tuple[int, ...]:
    return tuple(token_id for token_id in range(vocab_size) if token_id != bos_id)


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
        return sample_token(predictions, eos_id=eos_id, rng=rng, temperature=temperature)
    raise ValueError(f"Unsupported decoding mode: {decoding}")


def most_probable_token(
    predictions: Sequence[NgramPrediction],
    *,
    eos_id: int,
) -> int:
    if not predictions:
        return fallback_token_id(eos_id)
    return predictions[0].token_id


def sample_token(
    predictions: Sequence[NgramPrediction],
    *,
    eos_id: int,
    rng: random.Random,
    temperature: float,
) -> int:
    if not predictions:
        return fallback_token_id(eos_id)
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
    # Deterministic text sampling RNG; not used for secrets or security choices.
    return random.Random(seed)  # nosec B311


def fallback_token_id(eos_id: int) -> int:
    return eos_id if eos_id >= 0 else 0


def generation_prediction_top_k(*, decoding: DecodingMode, temperature: float) -> int:
    # top_k=0 means "all candidates"; greedy paths only need the first row.
    if decoding == "most-probable" or temperature == 0:
        return 1
    return 0


def prediction_sort_key(prediction: NgramPrediction) -> tuple[float, int]:
    return -prediction.probability, prediction.token_id


def sorted_predictions(
    predictions: Iterable[NgramPrediction],
    *,
    top_k: int,
) -> list[NgramPrediction]:
    ranked_predictions = sorted(predictions, key=prediction_sort_key)
    return ranked_predictions[:top_k] if top_k > 0 else ranked_predictions


def greedy_token_id(ranked_token_ids: Sequence[int], *, eos_id: int) -> int:
    if ranked_token_ids:
        return ranked_token_ids[0]
    return fallback_token_id(eos_id)


def top_k_token_id_set(ranked_token_ids: Sequence[int], *, top_k: int) -> frozenset[int]:
    return frozenset(ranked_token_ids[:top_k]) if top_k > 0 else frozenset()


def load_pieces(
    data: dict[str, object],
    tokenizer: tok_core.TokenizerCodec,
    vocab_size: int,
) -> tuple[str, ...]:
    stored_pieces = data.get("pieces")
    if stored_pieces:
        return tuple(str(piece) for piece in stored_pieces)
    return tuple(tokenizer.id_to_piece(index) for index in range(vocab_size))


def resolve_stored_path(stored_path: Path, model_path: Path) -> Path:
    if stored_path.is_absolute() or stored_path.exists():
        return stored_path

    model_relative_path = model_path.parent / stored_path
    if model_relative_path.exists():
        return model_relative_path

    return stored_path


def schema_label(model_type: str) -> str:
    words = model_type.replace("_", " ")
    article = "an" if words[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {words} model"


def load_json_model_payload(
    model_path: Path,
    *,
    model_type: str,
    label: str | None = None,
) -> dict[str, Any]:
    data = json_io.read_mapping(model_path)
    if data.get("model_type") != model_type:
        raise ValueError(f"Not {label or schema_label(model_type)}: {model_path}")
    return data


def write_json_model_payload(output_path: Path, model: dict[str, object]) -> None:
    json_io.write_json(output_path, model)


def tokenizer_model_payload(
    tokenizer: tok_core.TokenizerCodec,
    *,
    tokenizer_model: Path,
    stored_tokenizer_model: Path | None,
    text_normalization: normalization.TextNormalization,
) -> dict[str, object]:
    return tok_core.tokenizer_payload(
        tokenizer,
        tokenizer_model=tokenizer_model,
        stored_tokenizer_model=stored_tokenizer_model,
        text_normalization=text_normalization,
    )


def tokenizer_model_fields(
    data: dict[str, object],
    tokenizer: tok_core.TokenizerCodec,
    vocab_size: int,
) -> dict[str, object]:
    return {
        "vocab_size": vocab_size,
        "bos_id": int(data["bos_id"]),
        "eos_id": int(data["eos_id"]),
        "unk_id": int(data["unk_id"]),
        "pieces": load_pieces(data, tokenizer, vocab_size),
        "text_normalization": str(data.get("text_normalization", "none")),
    }


def load_tokenizer_model_fields(
    data: dict[str, object],
    model_path: Path,
) -> dict[str, object]:
    tokenizer_model, tokenizer, vocab_size = load_tokenizer_from_payload(data, model_path)
    return {
        "model_path": model_path,
        "tokenizer_model": tokenizer_model,
        "tokenizer": tokenizer,
        **tokenizer_model_fields(data, tokenizer, vocab_size),
    }


def load_tokenizer_from_payload(
    data: dict[str, object],
    model_path: Path,
) -> tuple[Path, tok_core.TokenizerCodec, int]:
    tokenizer_model = resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    tokenizer = tok_core.load_tokenizer(
        tokenizer_model,
        tokenizer_algo=str(data["tokenizer_algo"]) if data.get("tokenizer_algo") else None,
    )
    return tokenizer_model, tokenizer, int(data.get("vocab_size", tokenizer.vocab_size))


def score_evaluation_transition(
    summary: NgramEvaluationSummary,
    *,
    actual_token_id: int,
    greedy_token_id: int,
    top_k_token_ids: frozenset[int],
    probability: float,
) -> None:
    if actual_token_id == greedy_token_id:
        summary.correct_next_token_count += 1
    if actual_token_id in top_k_token_ids:
        summary.top_k_correct_next_token_count += 1

    if probability <= 0:
        summary.zero_probability_count += 1
    else:
        summary.negative_log_likelihood -= math.log(probability)


def additive_smoothed_probability(
    token_id: int,
    *,
    counts: Mapping[int, int],
    total: int,
    smoothing: float,
    candidate_count: int,
) -> float:
    denominator = total + smoothing * candidate_count
    if denominator <= 0:
        return 0.0
    return (counts.get(token_id, 0) + smoothing) / denominator


def maximum_likelihood_probability(
    token_id: int,
    *,
    counts: Mapping[int, int],
    total: int,
) -> float:
    if total <= 0:
        return 0.0
    return counts.get(token_id, 0) / total


def discounted_interpolation_probability(
    token_id: int,
    *,
    counts: Mapping[int, int],
    total: int,
    discount: float,
    lower_order_probability: float,
) -> float:
    if total <= 0:
        return lower_order_probability

    observed_count = counts.get(token_id, 0)
    discounted_probability = max(observed_count - discount, 0.0) / total
    interpolation_weight = discount * len(counts) / total
    return discounted_probability + interpolation_weight * lower_order_probability


def base_training_summary_items(
    *,
    summary: NgramPydanticModel,
    artifact_label: str,
) -> list[tuple[str, str]]:
    return [
        ("Tokenizer model file", formatting.artifact_filename(summary.tokenizer_model)),
        (artifact_label, formatting.artifact_filename(summary.output_path)),
        ("Text normalization", summary.text_normalization),
        ("Vocabulary size", f"{summary.vocab_size:,}"),
        ("Sequences", f"{summary.sequence_count:,}"),
        ("Tokens", f"{summary.token_count:,}"),
    ]


def base_evaluation_items(summary: NgramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        ("Model file", formatting.artifact_filename(summary.model_path)),
        ("Tokenizer model file", formatting.artifact_filename(summary.tokenizer_model)),
        ("Text normalization", summary.text_normalization),
    ]


def parse_token_transitions(
    data: dict[str, object],
    key: str,
) -> dict[int, tuple[tuple[int, int], ...]]:
    return {
        int(previous_id): tuple(
            (int(next_id), int(count))
            for next_id, count in next_counts
        )
        for previous_id, next_counts in data[key].items()
    }


def parse_token_counts(data: dict[str, object], key: str) -> dict[int, int]:
    return {
        int(token_id): int(count)
        for token_id, count in data[key]
    }


def token_transition_payload(
    transitions: defaultdict[int, Counter[int]] | dict[int, Counter[int]],
) -> dict[str, list[tuple[int, int]]]:
    return {
        str(previous_id): sorted(next_counts.items())
        for previous_id, next_counts in sorted(transitions.items())
    }


def token_counts_payload(counts: Counter[int] | Mapping[int, int]) -> list[tuple[int, int]]:
    return sorted(counts.items())


def iter_token_sequences(
    texts: Iterable[str],
    tokenizer: tok_core.TokenizerCodec,
    *,
    bos_count: int,
    min_length: int,
    text_normalization: normalization.TextNormalization = "none",
) -> Iterator[list[int]]:
    yield from tok_core.iter_token_sequences(
        texts,
        tokenizer,
        bos_count=bos_count,
        min_length=min_length,
        text_normalization=text_normalization,
    )
