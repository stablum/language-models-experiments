"""Interpolated Kneser-Ney token-level autoregressive trigram model."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TextNormalization
from src.models import ngram
from src.models.definition import ModelDefinition, ModelOptionError, ModelOptions
from src.models.formatting import (
    artifact_filename,
    format_ngram_evaluation_metrics,
    format_ngram_query,
)
from src.models.trigram_common import (
    BaseTrigramModel,
    Context,
    TrigramEvaluationRow,
    TrigramQueryResult,
    collect_trigram_counts,
    parse_context_transitions,
    parse_token_counts,
    parse_token_transitions,
    token_counts_payload,
    token_transition_payload,
    trigram_counts_payload,
)


MODEL_NAME = "trigram-kneser-ney"


@dataclass(frozen=True)
class KneserNeyTrigramTrainingSummary:
    output_path: Path
    tokenizer_model: Path
    vocab_size: int
    sequence_count: int
    token_count: int
    unigram_count: int
    bigram_transition_count: int
    trigram_transition_count: int
    continuation_unigram_count: int
    continuation_bigram_type_count: int
    discount: float
    text_normalization: str


@dataclass(frozen=True)
class KneserNeyTrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    discount: float


@dataclass(frozen=True)
class KneserNeyContinuationCounts:
    unigram_counts: Counter[int]
    bigram_transitions: defaultdict[int, Counter[int]]

    @property
    def unigram_count(self) -> int:
        return sum(self.unigram_counts.values())

    @property
    def bigram_type_count(self) -> int:
        return sum(len(next_counts) for next_counts in self.bigram_transitions.values())


@dataclass(frozen=True)
class KneserNeyTrigramModel(BaseTrigramModel):
    model_path: Path
    tokenizer_model: Path
    processor: spm.SentencePieceProcessor
    vocab_size: int
    discount: float
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    unigram_counts: dict[int, int]
    unigram_total: int
    bigram_transitions: dict[int, tuple[tuple[int, int], ...]]
    trigram_transitions: dict[Context, tuple[tuple[int, int], ...]]
    text_normalization: str = "none"

    def evaluation_summary(self, **kwargs: Any) -> KneserNeyTrigramEvaluationSummary:
        return KneserNeyTrigramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            discount=self.discount,
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

        return self.trigram_probability(
            next_id,
            previous_id=previous_id,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )

    def trigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        bigram_counts: dict[int, int],
        trigram_counts: dict[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> float:
        lower_order_probability = self.bigram_probability(
            token_id,
            previous_id=previous_id,
            counts=bigram_counts,
            total=bigram_total,
        )
        if trigram_total <= 0:
            return lower_order_probability

        observed_count = trigram_counts.get(token_id, 0)
        discounted_probability = max(observed_count - self.discount, 0.0) / trigram_total
        interpolation_weight = self.discount * len(trigram_counts) / trigram_total
        return discounted_probability + interpolation_weight * lower_order_probability

    def bigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        counts: dict[int, int] | None = None,
        total: int | None = None,
    ) -> float:
        if counts is None:
            counts = dict(self.bigram_transitions.get(previous_id, ()))
        if total is None:
            total = sum(counts.values())

        lower_order_probability = self.unigram_probability(token_id)
        if total <= 0:
            return lower_order_probability

        observed_count = counts.get(token_id, 0)
        discounted_probability = max(observed_count - self.discount, 0.0) / total
        interpolation_weight = self.discount * len(counts) / total
        return discounted_probability + interpolation_weight * lower_order_probability

    def unigram_probability(self, token_id: int) -> float:
        candidate_count = ngram.candidate_token_count(self.vocab_size, self.bos_id)
        if candidate_count <= 0:
            return 0.0

        uniform_probability = 1 / candidate_count
        if self.unigram_total <= 0:
            return uniform_probability

        observed_count = self.unigram_counts.get(token_id, 0)
        discounted_probability = max(observed_count - self.discount, 0.0) / self.unigram_total
        interpolation_weight = self.discount * len(self.unigram_counts) / self.unigram_total
        return discounted_probability + interpolation_weight * uniform_probability


def load_kneser_ney_trigram_model(model_path: Path) -> KneserNeyTrigramModel:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    if data.get("model_type") != "interpolated_kneser_ney_trigram":
        raise ValueError(f"Not an interpolated Kneser-Ney trigram model: {model_path}")

    tokenizer_model = ngram.resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    vocab_size = int(data["vocab_size"])

    return KneserNeyTrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        vocab_size=vocab_size,
        discount=float(data["discount"]),
        bos_id=int(data["bos_id"]),
        eos_id=int(data["eos_id"]),
        unk_id=int(data["unk_id"]),
        pieces=ngram.load_pieces(data, processor, vocab_size),
        unigram_counts=parse_token_counts(data, "kneser_ney_unigrams"),
        unigram_total=int(data["kneser_ney_unigram_count"]),
        bigram_transitions=parse_token_transitions(data, "kneser_ney_bigram_transitions"),
        trigram_transitions=parse_context_transitions(data, "trigram_transitions"),
        text_normalization=str(data.get("text_normalization", "none")),
    )


def train_kneser_ney_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    discount: float = 0.75,
    text_normalization: TextNormalization = DEFAULT_TEXT_NORMALIZATION,
) -> KneserNeyTrigramTrainingSummary:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    counts = collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )
    continuation_counts = collect_kneser_ney_continuation_counts(
        counts.trigram_transitions,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "schema_version": 1,
        "model_type": "interpolated_kneser_ney_trigram",
        "tokenizer_model": str(stored_tokenizer_model or tokenizer_model),
        "vocab_size": processor.get_piece_size(),
        "discount": discount,
        "text_normalization": text_normalization,
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "unk_id": processor.unk_id(),
        "pieces": [processor.id_to_piece(index) for index in range(processor.get_piece_size())],
        **trigram_counts_payload(counts),
        "kneser_ney_unigram_count": continuation_counts.unigram_count,
        "kneser_ney_unigrams": token_counts_payload(continuation_counts.unigram_counts),
        "kneser_ney_bigram_transitions": token_transition_payload(
            continuation_counts.bigram_transitions
        ),
    }
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return KneserNeyTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        sequence_count=counts.sequence_count,
        token_count=counts.token_count,
        unigram_count=counts.unigram_count,
        bigram_transition_count=counts.bigram_transition_count,
        trigram_transition_count=counts.trigram_transition_count,
        continuation_unigram_count=continuation_counts.unigram_count,
        continuation_bigram_type_count=continuation_counts.bigram_type_count,
        discount=discount,
        text_normalization=text_normalization,
    )


def default_tokenizer_model(corpus: str) -> Path:
    return Path("artifacts", "tokenizers", f"{corpus}-sentencepiece-1000.model")


def default_output(corpus: str) -> Path:
    return Path("artifacts", "models", f"{corpus}-sentencepiece-trigram-kneser-ney.json")


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
            "Train it first with src.cli.train_sentencepiece."
        )


def validate_query_options(options: ModelOptions) -> None:
    model_path = resolve_model(options)
    if not model_path.exists():
        raise ModelOptionError(
            f"Interpolated Kneser-Ney trigram model not found: {model_path}. "
            "Train it first with src.cli.train."
        )


def train_from_options(
    texts: Iterable[str],
    options: ModelOptions,
) -> KneserNeyTrigramTrainingSummary:
    stored_tokenizer_model = options.get("stored_tokenizer_model")
    return train_kneser_ney_trigram_model(
        texts,
        tokenizer_model=resolve_tokenizer_model(options),
        output_path=resolve_output(options),
        stored_tokenizer_model=Path(stored_tokenizer_model) if stored_tokenizer_model else None,
        discount=options["discount"],
        text_normalization=options["text_normalization"],
    )


def query_from_options(options: ModelOptions) -> TrigramQueryResult:
    model = load_kneser_ney_trigram_model(resolve_model(options))
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
) -> KneserNeyTrigramEvaluationSummary:
    model = load_kneser_ney_trigram_model(resolve_model(options))
    return model.evaluate(texts, top_k=options["top_k"])


def format_summary(
    summary: KneserNeyTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        ("Tokenizer artifact file", artifact_filename(summary.tokenizer_model)),
        (
            "Interpolated Kneser-Ney trigram model artifact file",
            artifact_filename(summary.output_path),
        ),
        ("Text normalization", summary.text_normalization),
        ("Vocabulary size", f"{summary.vocab_size:,}"),
        ("Sequences", f"{summary.sequence_count:,}"),
        ("Tokens", f"{summary.token_count:,}"),
        ("Unigrams", f"{summary.unigram_count:,}"),
        ("Bigram transitions", f"{summary.bigram_transition_count:,}"),
        ("Trigram transitions", f"{summary.trigram_transition_count:,}"),
        ("Continuation unigrams", f"{summary.continuation_unigram_count:,}"),
        ("Continuation bigram types", f"{summary.continuation_bigram_type_count:,}"),
        ("Discount", f"{summary.discount:.3f}"),
    ]


def format_evaluation(
    summary: KneserNeyTrigramEvaluationSummary,
) -> list[tuple[str, str]]:
    return [
        ("Model artifact file", artifact_filename(summary.model_path)),
        ("Tokenizer artifact file", artifact_filename(summary.tokenizer_model)),
        ("Text normalization", summary.text_normalization),
        ("Discount", f"{summary.discount:.3f}"),
        *format_ngram_evaluation_metrics(summary),
    ]


def format_query(result: TrigramQueryResult) -> list[str]:
    return format_ngram_query(result)


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


def collect_kneser_ney_continuation_counts(
    trigram_transitions: defaultdict[Context, Counter[int]] | dict[Context, Counter[int]],
) -> KneserNeyContinuationCounts:
    bigram_transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    unigram_predecessors: defaultdict[int, set[int]] = defaultdict(set)

    for (_, previous_id), next_counts in trigram_transitions.items():
        for next_id, count in next_counts.items():
            if count <= 0:
                continue
            bigram_transitions[previous_id][next_id] += 1
            unigram_predecessors[next_id].add(previous_id)

    return KneserNeyContinuationCounts(
        unigram_counts=Counter(
            {
                token_id: len(predecessors)
                for token_id, predecessors in unigram_predecessors.items()
            }
        ),
        bigram_transitions=bigram_transitions,
    )
