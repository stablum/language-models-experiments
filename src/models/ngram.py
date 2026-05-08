"""Shared helpers for small token-level n-gram models."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import pydantic
import sentencepiece as spm

from src.corpora import normalization
from src.ml_core.models.definition import ModelOptionError, ModelOptions
from src.models import formatting


DecodingMode = Literal["sample", "most-probable"]


class NgramPydanticModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
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


QueryResult = TypeVar("QueryResult", bound=NgramQueryResult)
EvaluationSummary = TypeVar("EvaluationSummary", bound=NgramEvaluationSummary)
LoadedModel = TypeVar("LoadedModel")


def divide_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def encode_prompt(
    processor: spm.SentencePieceProcessor,
    prompt: str,
    *,
    text_normalization: normalization.TextNormalization = "none",
) -> list[int]:
    prompt = normalization.normalize_text(prompt, text_normalization)
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


def load_json_model_payload(model_path: Path, *, model_type: str, label: str) -> dict[str, Any]:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    if data.get("model_type") != model_type:
        raise ValueError(f"Not {label}: {model_path}")
    return data


def write_json_model_payload(output_path: Path, model: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sentencepiece_model_payload(
    processor: spm.SentencePieceProcessor,
    *,
    tokenizer_model: Path,
    stored_tokenizer_model: Path | None,
    vocab_size: int,
    text_normalization: normalization.TextNormalization,
) -> dict[str, object]:
    return {
        "tokenizer_model": str(stored_tokenizer_model or tokenizer_model),
        "vocab_size": vocab_size,
        "text_normalization": text_normalization,
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "unk_id": processor.unk_id(),
        "pieces": [processor.id_to_piece(index) for index in range(vocab_size)],
    }


def load_sentencepiece_from_payload(
    data: dict[str, object],
    model_path: Path,
) -> tuple[Path, spm.SentencePieceProcessor, int]:
    tokenizer_model = resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    return tokenizer_model, processor, int(data["vocab_size"])


def default_tokenizer_model(corpus: str) -> Path:
    return Path("artifacts", "tokenizers", f"{corpus}-sentencepiece-1000.model")


def default_ngram_output(corpus: str, model_suffix: str) -> Path:
    return Path("artifacts", "models", f"{corpus}-sentencepiece-{model_suffix}.json")


def resolve_tokenizer_model(options: ModelOptions) -> Path:
    tokenizer_model = options.get("tokenizer_model")
    if tokenizer_model:
        return Path(tokenizer_model)
    return default_tokenizer_model(str(options["corpus"]))


def resolve_output(options: ModelOptions, *, model_suffix: str) -> Path:
    output = options.get("output")
    return Path(output) if output else default_ngram_output(str(options["corpus"]), model_suffix)


def resolve_model(options: ModelOptions, *, model_suffix: str) -> Path:
    model_path = options.get("model_path")
    return Path(model_path) if model_path else default_ngram_output(
        str(options["corpus"]),
        model_suffix,
    )


def validate_tokenizer_model(options: ModelOptions) -> None:
    tokenizer_model = resolve_tokenizer_model(options)
    if not tokenizer_model.exists():
        raise ModelOptionError(
            f"Tokenizer model not found: {tokenizer_model}. "
            "Train it first with src.cli.tokenizer_training."
        )


def validate_model_path(options: ModelOptions, *, model_suffix: str, label: str) -> None:
    model_path = resolve_model(options, model_suffix=model_suffix)
    if not model_path.exists():
        raise ModelOptionError(
            f"{label} model not found: {model_path}. "
            "Train it first with src.cli.train."
        )


def query_from_options(
    options: ModelOptions,
    *,
    load_model: Callable[[Path], LoadedModel],
    model_suffix: str,
) -> QueryResult:
    model = load_model(resolve_model(options, model_suffix=model_suffix))
    return model.query(
        prompt=options["prompt"],
        max_tokens=options["max_tokens"],
        top_k=options["top_k"],
        decoding=options["decoding"],
        temperature=options["temperature"],
        seed=options["seed"],
    )


def evaluate_from_options(
    texts: Iterable[str],
    options: ModelOptions,
    *,
    load_model: Callable[[Path], LoadedModel],
    model_suffix: str,
) -> EvaluationSummary:
    model = load_model(resolve_model(options, model_suffix=model_suffix))
    return model.evaluate(texts, top_k=options["top_k"])


def base_training_summary_items(
    *,
    summary: NgramPydanticModel,
    artifact_label: str,
) -> list[tuple[str, str]]:
    return [
        ("Tokenizer artifact file", formatting.artifact_filename(summary.tokenizer_model)),
        (artifact_label, formatting.artifact_filename(summary.output_path)),
        ("Text normalization", summary.text_normalization),
        ("Vocabulary size", f"{summary.vocab_size:,}"),
        ("Sequences", f"{summary.sequence_count:,}"),
        ("Tokens", f"{summary.token_count:,}"),
    ]


def base_evaluation_items(summary: NgramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        ("Model artifact file", formatting.artifact_filename(summary.model_path)),
        ("Tokenizer artifact file", formatting.artifact_filename(summary.tokenizer_model)),
        ("Text normalization", summary.text_normalization),
    ]


def iter_sentencepiece_token_sequences(
    texts: Iterable[str],
    processor: spm.SentencePieceProcessor,
    *,
    bos_count: int,
    min_length: int,
    text_normalization: normalization.TextNormalization = "none",
) -> Iterator[list[int]]:
    bos_id = processor.bos_id()
    eos_id = processor.eos_id()

    for text in texts:
        text = normalization.normalize_text(text, text_normalization)
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
