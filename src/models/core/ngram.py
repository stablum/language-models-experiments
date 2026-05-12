"""Shared helpers for small token-level n-gram models."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import pydantic
import sentencepiece as spm

from src.corpora import normalization
from src.ml_core.models import definition as model_def
from src.models.core import formatting


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
    processor: spm.SentencePieceProcessor
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

    def encode_prompt(self, prompt: str) -> list[int]:
        return encode_prompt(
            self.processor,
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

    def query(
        self,
        *,
        prompt: str = "",
        max_tokens: int = 80,
        top_k: int = 10,
        decoding: DecodingMode = "sample",
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> NgramQueryResult:
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
            context = self.advance_context(context, next_id)

        prompt_text = self.processor.decode(prompt_token_ids)
        generated_text = self.processor.decode(token_ids)
        continuation_text = decode_continuation(
            self.processor,
            generated_text=generated_text,
            prompt_text=prompt_text,
            generated_token_ids=generated_token_ids,
        )

        return NgramQueryResult(
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
            text_normalization=self.text_normalization,
        )


QueryResult = TypeVar("QueryResult", bound=NgramQueryResult)
EvaluationSummary = TypeVar("EvaluationSummary", bound=NgramEvaluationSummary)
LoadedModel = TypeVar("LoadedModel")
TrainingSummary = TypeVar("TrainingSummary")


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
    return random.Random(seed)


def fallback_token_id(eos_id: int) -> int:
    return eos_id if eos_id >= 0 else 0


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
    data = json.loads(model_path.read_text(encoding="utf-8"))
    if data.get("model_type") != model_type:
        raise ValueError(f"Not {label or schema_label(model_type)}: {model_path}")
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


def sentencepiece_model_fields(
    data: dict[str, object],
    processor: spm.SentencePieceProcessor,
    vocab_size: int,
) -> dict[str, object]:
    return {
        "vocab_size": vocab_size,
        "bos_id": int(data["bos_id"]),
        "eos_id": int(data["eos_id"]),
        "unk_id": int(data["unk_id"]),
        "pieces": load_pieces(data, processor, vocab_size),
        "text_normalization": str(data.get("text_normalization", "none")),
    }


def load_sentencepiece_model_fields(
    data: dict[str, object],
    model_path: Path,
) -> dict[str, object]:
    tokenizer_model, processor, vocab_size = load_sentencepiece_from_payload(
        data,
        model_path,
    )
    return {
        "model_path": model_path,
        "tokenizer_model": tokenizer_model,
        "processor": processor,
        **sentencepiece_model_fields(data, processor, vocab_size),
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


def resolve_tokenizer_model(options: model_def.ModelOptions) -> Path:
    tokenizer_model = options.get("tokenizer_model")
    if tokenizer_model:
        return Path(tokenizer_model)
    return default_tokenizer_model(str(options["corpus"]))


def resolve_output(options: model_def.ModelOptions, *, model_suffix: str) -> Path:
    output = options.get("output")
    return Path(output) if output else default_ngram_output(str(options["corpus"]), model_suffix)


def resolve_model(options: model_def.ModelOptions, *, model_suffix: str) -> Path:
    model_path = options.get("model_path")
    return Path(model_path) if model_path else default_ngram_output(
        str(options["corpus"]),
        model_suffix,
    )


def validate_tokenizer_model(options: model_def.ModelOptions) -> None:
    tokenizer_model = resolve_tokenizer_model(options)
    if not tokenizer_model.exists():
        raise model_def.ModelOptionError(
            f"Tokenizer model not found: {tokenizer_model}. "
            "Train it first with src.cli.tokenizer_training."
        )


def validate_model_path(
    options: model_def.ModelOptions,
    *,
    model_suffix: str,
    label: str,
) -> None:
    model_path = resolve_model(options, model_suffix=model_suffix)
    if not model_path.exists():
        raise model_def.ModelOptionError(
            f"{label} model not found: {model_path}. "
            "Train it first with src.cli.train."
        )


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


def model_name_from_module(module_name: str) -> str:
    return module_name.rsplit(".", maxsplit=1)[-1].replace("_", "-")


def model_label_from_name(name: str) -> str:
    return name.replace("-", " ").capitalize()


def model_definition(
    *,
    module_name: str,
    train_model: Callable[..., TrainingSummary],
    load_model: Callable[[Path], LoadedModel],
    summary_items: model_def.SummaryFormatter,
    training_option_names: Sequence[str] = (),
    query_lines: model_def.QueryFormatter | None = None,
    evaluation_items: model_def.SummaryFormatter | None = None,
    validate_training_options: model_def.ModelOptionValidator | None = None,
) -> model_def.ModelDefinition:
    name = model_name_from_module(module_name)
    model_label = model_label_from_name(name)

    def train(
        texts: Iterable[str],
        options: model_def.ModelOptions,
    ) -> TrainingSummary:
        stored_tokenizer_model = options.get("stored_tokenizer_model")
        training_options = {
            option_name: options[option_name]
            for option_name in training_option_names
        }
        return train_model(
            texts,
            tokenizer_model=resolve_tokenizer_model(options),
            output_path=resolve_output(options, model_suffix=name),
            stored_tokenizer_model=(
                Path(stored_tokenizer_model) if stored_tokenizer_model else None
            ),
            text_normalization=options["text_normalization"],
            **training_options,
        )

    def validate_options(options: model_def.ModelOptions) -> None:
        validate_tokenizer_model(options)
        if validate_training_options is not None:
            validate_training_options(options)

    def validate_query_options(options: model_def.ModelOptions) -> None:
        validate_model_path(options, model_suffix=name, label=model_label)

    def query(options: model_def.ModelOptions) -> QueryResult:
        model = load_model(resolve_model(options, model_suffix=name))
        return model.query(
            prompt=options["prompt"],
            max_tokens=options["max_tokens"],
            top_k=options["top_k"],
            decoding=options["decoding"],
            temperature=options["temperature"],
            seed=options["seed"],
        )

    def evaluate(
        texts: Iterable[str],
        options: model_def.ModelOptions,
    ) -> EvaluationSummary:
        model = load_model(resolve_model(options, model_suffix=name))
        return model.evaluate(texts, top_k=options["top_k"])

    return model_def.ModelDefinition(
        name=name,
        train=train,
        validate_options=validate_options,
        summary_items=summary_items,
        query=query,
        validate_query_options=validate_query_options,
        query_lines=query_lines or formatting.format_ngram_query,
        evaluate=evaluate,
        validate_evaluation_options=validate_query_options,
        evaluation_items=evaluation_items or standard_evaluation_items,
    )


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


def standard_evaluation_items(summary: NgramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        *base_evaluation_items(summary),
        *formatting.format_ngram_evaluation_metrics(summary),
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
