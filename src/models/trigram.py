"""Interpolated token-level autoregressive trigram model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

from src.corpora import normalization
from src.models import formatting, ngram, trigram_common
from src.ml_core.models.definition import ModelDefinition, ModelOptionError, ModelOptions


MODEL_NAME = "trigram"


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


@dataclass
class _TrigramTrainingSummaryDraft:
    output_path: Path
    tokenizer_model: Path
    vocab_size: int = 0
    sequence_count: int = 0
    token_count: int = 0
    unigram_count: int = 0
    bigram_transition_count: int = 0
    trigram_transition_count: int = 0
    unigram_weight: float = 0.0
    bigram_weight: float = 0.0
    trigram_weight: float = 0.0
    text_normalization: str = "none"

    def freeze(self) -> TrigramTrainingSummary:
        return TrigramTrainingSummary(
            output_path=self.output_path,
            tokenizer_model=self.tokenizer_model,
            vocab_size=self.vocab_size,
            sequence_count=self.sequence_count,
            token_count=self.token_count,
            unigram_count=self.unigram_count,
            bigram_transition_count=self.bigram_transition_count,
            trigram_transition_count=self.trigram_transition_count,
            unigram_weight=self.unigram_weight,
            bigram_weight=self.bigram_weight,
            trigram_weight=self.trigram_weight,
            text_normalization=self.text_normalization,
        )


@dataclass(frozen=True)
class TrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float


@dataclass(frozen=True)
class TrigramModel(trigram_common.BaseTrigramModel):
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
    trigram_transitions: dict[trigram_common.Context, tuple[tuple[int, int], ...]]
    text_normalization: str = "none"

    def evaluation_summary(
        self,
        summary: ngram.NgramEvaluationSummaryDraft,
    ) -> TrigramEvaluationSummary:
        return summary.freeze(
            TrigramEvaluationSummary,
            unigram_weight=self.unigram_weight,
            bigram_weight=self.bigram_weight,
            trigram_weight=self.trigram_weight,
        )

    def transition_probability(
        self,
        next_id: int,
        context: trigram_common.Context,
        *,
        row: trigram_common.TrigramEvaluationRow | None = None,
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
        unigram_counts=trigram_common.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigram_common.parse_bigram_transitions(data),
        trigram_transitions=trigram_common.parse_trigram_transitions(data),
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
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> TrigramTrainingSummary:
    normalized_weights = normalize_interpolation_weights(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
    )
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    summary = _TrigramTrainingSummaryDraft(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        unigram_weight=normalized_weights[0],
        bigram_weight=normalized_weights[1],
        trigram_weight=normalized_weights[2],
        text_normalization=text_normalization,
    )
    counts = trigram_common.collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )
    summary.sequence_count = counts.sequence_count
    summary.token_count = counts.token_count
    summary.unigram_count = counts.unigram_count
    summary.bigram_transition_count = counts.bigram_transition_count
    summary.trigram_transition_count = counts.trigram_transition_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "schema_version": 1,
        "model_type": "interpolated_trigram",
        "tokenizer_model": str(stored_tokenizer_model or tokenizer_model),
        "vocab_size": summary.vocab_size,
        "smoothing": smoothing,
        "text_normalization": text_normalization,
        "interpolation_weights": {
            "unigram": summary.unigram_weight,
            "bigram": summary.bigram_weight,
            "trigram": summary.trigram_weight,
        },
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "unk_id": processor.unk_id(),
        "pieces": [processor.id_to_piece(index) for index in range(summary.vocab_size)],
        **trigram_common.trigram_counts_payload(counts),
    }
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary.freeze()


def default_tokenizer_model(corpus: str) -> Path:
    return Path("artifacts", "tokenizers", f"{corpus}-sentencepiece-1000.model")


def default_output(corpus: str) -> Path:
    return Path("artifacts", "models", f"{corpus}-sentencepiece-trigram.json")


def default_model(corpus: str) -> Path:
    return default_output(corpus)


def resolve_tokenizer_model(options: ModelOptions) -> Path:
    tokenizer_model = options.get("tokenizer_model")
    if tokenizer_model:
        return Path(tokenizer_model)
    return default_tokenizer_model(str(options["corpus"]))


def resolve_output(options: ModelOptions) -> Path:
    output = options.get("output")
    return Path(output) if output else default_output(str(options["corpus"]))


def resolve_model(options: ModelOptions) -> Path:
    model_path = options.get("model_path")
    return Path(model_path) if model_path else default_model(str(options["corpus"]))


def validate_options(options: ModelOptions) -> None:
    tokenizer_model = resolve_tokenizer_model(options)
    if not tokenizer_model.exists():
        raise ModelOptionError(
            f"Tokenizer model not found: {tokenizer_model}. "
            "Train it first with src.cli.tokenizer_training."
        )

    try:
        normalize_interpolation_weights(
            unigram_weight=options["unigram_weight"],
            bigram_weight=options["bigram_weight"],
            trigram_weight=options["trigram_weight"],
        )
    except ValueError as error:
        raise ModelOptionError(str(error)) from error


def validate_query_options(options: ModelOptions) -> None:
    model_path = resolve_model(options)
    if not model_path.exists():
        raise ModelOptionError(
            f"Trigram model not found: {model_path}. "
            "Train it first with src.cli.train."
        )


def train_from_options(
    texts: Iterable[str],
    options: ModelOptions,
) -> TrigramTrainingSummary:
    stored_tokenizer_model = options.get("stored_tokenizer_model")
    return train_trigram_model(
        texts,
        tokenizer_model=resolve_tokenizer_model(options),
        output_path=resolve_output(options),
        stored_tokenizer_model=Path(stored_tokenizer_model) if stored_tokenizer_model else None,
        smoothing=options["smoothing"],
        unigram_weight=options["unigram_weight"],
        bigram_weight=options["bigram_weight"],
        trigram_weight=options["trigram_weight"],
        text_normalization=options["text_normalization"],
    )


def query_from_options(options: ModelOptions) -> trigram_common.TrigramQueryResult:
    model = load_trigram_model(resolve_model(options))
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
) -> TrigramEvaluationSummary:
    model = load_trigram_model(resolve_model(options))
    return model.evaluate(texts, top_k=options["top_k"])


def format_summary(summary: TrigramTrainingSummary) -> list[tuple[str, str]]:
    return [
        ("Tokenizer artifact file", formatting.artifact_filename(summary.tokenizer_model)),
        ("Trigram model artifact file", formatting.artifact_filename(summary.output_path)),
        ("Text normalization", summary.text_normalization),
        ("Vocabulary size", f"{summary.vocab_size:,}"),
        ("Sequences", f"{summary.sequence_count:,}"),
        ("Tokens", f"{summary.token_count:,}"),
        ("Unigrams", f"{summary.unigram_count:,}"),
        ("Bigram transitions", f"{summary.bigram_transition_count:,}"),
        ("Trigram transitions", f"{summary.trigram_transition_count:,}"),
        (
            "Interpolation weights",
            formatting.format_interpolation_weights(
                unigram_weight=summary.unigram_weight,
                bigram_weight=summary.bigram_weight,
                trigram_weight=summary.trigram_weight,
            ),
        ),
    ]


def format_evaluation(summary: TrigramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        ("Model artifact file", formatting.artifact_filename(summary.model_path)),
        ("Tokenizer artifact file", formatting.artifact_filename(summary.tokenizer_model)),
        ("Text normalization", summary.text_normalization),
        (
            "Interpolation weights",
            formatting.format_interpolation_weights(
                unigram_weight=summary.unigram_weight,
                bigram_weight=summary.bigram_weight,
                trigram_weight=summary.trigram_weight,
            ),
        ),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


def format_query(result: trigram_common.TrigramQueryResult) -> list[str]:
    return formatting.format_ngram_query(result)


MODEL_DEFINITION = ModelDefinition(
    name=MODEL_NAME,
    train=train_from_options,
    validate_options=validate_options,
    summary_items=format_summary,
    query=query_from_options,
    validate_query_options=validate_query_options,
    query_lines=format_query,
    evaluate=evaluate_from_options,
    validate_evaluation_options=validate_query_options,
    evaluation_items=format_evaluation,
)
