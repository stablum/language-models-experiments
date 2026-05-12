"""Shared pieces for token-level trigram models."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from src.corpora import normalization
from src.models.core import formatting, ngram
from src.tokenizers import core as tok_core


Context = tuple[int, int]


@dataclass(frozen=True)
class TrigramCounts:
    sequence_count: int
    token_count: int
    unigram_counts: Counter[int]
    bigram_transitions: defaultdict[int, Counter[int]]
    trigram_transitions: defaultdict[Context, Counter[int]]
    bigram_transition_count: int
    trigram_transition_count: int

    @property
    def unigram_count(self) -> int:
        return sum(self.unigram_counts.values())


class TrigramTrainingSummary(ngram.NgramTrainingSummary):
    unigram_count: int = 0
    bigram_transition_count: int = 0
    trigram_transition_count: int = 0


@dataclass(frozen=True)
class TrigramEvaluationRow:
    bigram_counts: dict[int, int]
    trigram_counts: dict[int, int]
    bigram_total: int
    trigram_total: int
    greedy_token_id: int
    top_k_token_ids: frozenset[int]


@dataclass(frozen=True)
class ResolvedTrigramContextCounts:
    previous_id: int
    bigram_counts: dict[int, int]
    trigram_counts: dict[int, int]
    bigram_total: int
    trigram_total: int


class BaseTrigramModel(ngram.BaseNgramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        ngram.NgramEvaluationSummary
    )
    bigram_transitions: dict[int, tuple[tuple[int, int], ...]]
    trigram_transitions: dict[Context, tuple[tuple[int, int], ...]]

    def advance_context(self, context: Context, next_id: int) -> Context:
        return context[1], next_id

    def next_token_predictions(
        self,
        context: Context,
        *,
        top_k: int,
    ) -> list[ngram.NgramPrediction]:
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        predictions = [
            ngram.NgramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=trigram_counts.get(token_id, 0),
                probability=self.transition_probability(token_id, context),
            )
            for token_id in self.candidate_ids
        ]
        predictions.sort(key=lambda prediction: (-prediction.probability, prediction.token_id))
        return predictions[:top_k] if top_k > 0 else predictions

    def evaluate(
        self,
        texts: Iterable[str],
        *,
        top_k: int = 5,
        text_normalization: normalization.TextNormalization | None = None,
    ) -> ngram.NgramEvaluationSummary:
        row_cache: dict[Context, TrigramEvaluationRow] = {}

        resolved_text_normalization = text_normalization or self.text_normalization
        summary = self.evaluation_summary(
            top_k=top_k,
            text_normalization=resolved_text_normalization,
        )
        for token_ids in iter_trigram_token_sequences(
            texts,
            self.tokenizer,
            text_normalization=resolved_text_normalization,
        ):
            summary.sequence_count += 1
            summary.token_count += len(token_ids)

            for previous_previous_id, previous_id, next_id in zip(
                token_ids,
                token_ids[1:],
                token_ids[2:],
            ):
                summary.transition_count += 1
                context = (previous_previous_id, previous_id)
                row = row_cache.get(context)
                if row is None:
                    row = self.evaluation_row(context, top_k=top_k)
                    row_cache[context] = row

                ngram.score_evaluation_transition(
                    summary,
                    actual_token_id=next_id,
                    greedy_token_id=row.greedy_token_id,
                    top_k_token_ids=row.top_k_token_ids,
                    probability=self.transition_probability(next_id, context, row=row),
                )

        return summary

    def evaluation_summary(
        self,
        *,
        top_k: int,
        text_normalization: str,
    ) -> ngram.NgramEvaluationSummary:
        return self.evaluation_summary_type(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            top_k=top_k,
            text_normalization=text_normalization,
            **self.evaluation_summary_fields(),
        )

    def evaluation_summary_fields(self) -> dict[str, object]:
        return {}

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
        if next_id not in self.candidate_id_set:
            return 0.0

        counts = self.resolved_context_counts(
            context,
            row=row,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )
        return self.context_probability(next_id, counts)

    def context_probability(
        self,
        next_id: int,
        counts: ResolvedTrigramContextCounts,
    ) -> float:
        raise NotImplementedError

    def resolved_context_counts(
        self,
        context: Context,
        *,
        row: TrigramEvaluationRow | None = None,
        bigram_counts: dict[int, int] | None = None,
        trigram_counts: dict[int, int] | None = None,
        bigram_total: int | None = None,
        trigram_total: int | None = None,
    ) -> ResolvedTrigramContextCounts:
        previous_id = context[1]
        if row is not None:
            return ResolvedTrigramContextCounts(
                previous_id=previous_id,
                bigram_counts=row.bigram_counts,
                trigram_counts=row.trigram_counts,
                bigram_total=row.bigram_total,
                trigram_total=row.trigram_total,
            )

        if bigram_counts is None:
            bigram_counts = dict(self.bigram_transitions.get(previous_id, ()))
        if trigram_counts is None:
            trigram_counts = dict(self.trigram_transitions.get(context, ()))

        return ResolvedTrigramContextCounts(
            previous_id=previous_id,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total if bigram_total is not None else sum(bigram_counts.values()),
            trigram_total=(
                trigram_total if trigram_total is not None else sum(trigram_counts.values())
            ),
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
        fallback_token_id = ngram.fallback_token_id(self.eos_id)
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
            self.candidate_ids,
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

    def context_for_tokens(self, token_ids: list[int]) -> Context:
        bos_id = self.bos_id if self.bos_id >= 0 else 0
        if len(token_ids) >= 2:
            return token_ids[-2], token_ids[-1]
        if len(token_ids) == 1:
            return bos_id, token_ids[-1]
        return bos_id, bos_id


class DiscountedTrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    discount: float = 0.0


class DiscountedTrigramModel(BaseTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        DiscountedTrigramEvaluationSummary
    )
    discount: float

    def evaluation_summary_fields(self) -> dict[str, object]:
        return {"discount": self.discount}


def collect_trigram_counts(
    texts: Iterable[str],
    tokenizer: tok_core.TokenizerCodec,
    *,
    text_normalization: normalization.TextNormalization = (
        normalization.DEFAULT_TEXT_NORMALIZATION
    ),
) -> TrigramCounts:
    unigram_counts: Counter[int] = Counter()
    bigram_transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    trigram_transitions: defaultdict[Context, Counter[int]] = defaultdict(Counter)
    sequence_count = 0
    token_count = 0
    bigram_transition_count = 0
    trigram_transition_count = 0

    for token_ids in iter_trigram_token_sequences(
        texts,
        tokenizer,
        text_normalization=text_normalization,
    ):
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

    return TrigramCounts(
        sequence_count=sequence_count,
        token_count=token_count,
        unigram_counts=unigram_counts,
        bigram_transitions=bigram_transitions,
        trigram_transitions=trigram_transitions,
        bigram_transition_count=bigram_transition_count,
        trigram_transition_count=trigram_transition_count,
    )


def trigram_counts_payload(counts: TrigramCounts) -> dict[str, object]:
    return {
        "sequence_count": counts.sequence_count,
        "token_count": counts.token_count,
        "unigram_count": counts.unigram_count,
        "bigram_transition_count": counts.bigram_transition_count,
        "trigram_transition_count": counts.trigram_transition_count,
        "unigrams": ngram.token_counts_payload(counts.unigram_counts),
        "bigram_transitions": ngram.token_transition_payload(counts.bigram_transitions),
        "trigram_transitions": context_transition_payload(counts.trigram_transitions),
    }


def apply_trigram_counts_to_summary(
    summary: ngram.NgramPydanticModel,
    counts: TrigramCounts,
) -> None:
    summary.sequence_count = counts.sequence_count
    summary.token_count = counts.token_count
    summary.unigram_count = counts.unigram_count
    summary.bigram_transition_count = counts.bigram_transition_count
    summary.trigram_transition_count = counts.trigram_transition_count


def base_training_summary_items(
    *,
    summary: ngram.NgramPydanticModel,
    artifact_label: str,
) -> list[tuple[str, str]]:
    return [
        *ngram.base_training_summary_items(summary=summary, artifact_label=artifact_label),
        ("Unigrams", f"{summary.unigram_count:,}"),
        ("Bigram transitions", f"{summary.bigram_transition_count:,}"),
        ("Trigram transitions", f"{summary.trigram_transition_count:,}"),
    ]


def discount_item(summary: ngram.NgramPydanticModel) -> tuple[str, str]:
    return "Discount", f"{summary.discount:.3f}"


def discounted_evaluation_items(
    summary: DiscountedTrigramEvaluationSummary,
) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        discount_item(summary),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


def load_standard_trigram_model_fields(
    model_path: Path,
    *,
    model_type: str,
    label: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    data = ngram.load_json_model_payload(model_path, model_type=model_type, label=label)
    return data, ngram.load_tokenizer_model_fields(data, model_path)


def standard_trigram_model_payload(
    tokenizer: tok_core.TokenizerCodec,
    *,
    model_type: str,
    tokenizer_model: Path,
    stored_tokenizer_model: Path | None,
    text_normalization: normalization.TextNormalization,
    counts: TrigramCounts,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_type": model_type,
        **ngram.tokenizer_model_payload(
            tokenizer,
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            text_normalization=text_normalization,
        ),
        **trigram_counts_payload(counts),
    }


def parse_unigram_counts(data: dict[str, object]) -> dict[int, int]:
    return ngram.parse_token_counts(data, "unigrams")


def parse_bigram_transitions(data: dict[str, object]) -> dict[int, tuple[tuple[int, int], ...]]:
    return ngram.parse_token_transitions(data, "bigram_transitions")


def parse_trigram_transitions(
    data: dict[str, object],
) -> dict[Context, tuple[tuple[int, int], ...]]:
    return parse_context_transitions(data, "trigram_transitions")


def context_transition_payload(
    transitions: defaultdict[Context, Counter[int]] | dict[Context, Counter[int]],
) -> dict[str, list[tuple[int, int]]]:
    return {
        context_key(previous_previous_id, previous_id): sorted(next_counts.items())
        for (
            previous_previous_id,
            previous_id,
        ), next_counts in sorted(transitions.items())
    }


def parse_context_transitions(
    data: dict[str, object],
    key: str,
) -> dict[Context, tuple[tuple[int, int], ...]]:
    return {
        parse_context_key(context_key): tuple(
            (int(next_id), int(count))
            for next_id, count in next_counts
        )
        for context_key, next_counts in data[key].items()
    }


def iter_trigram_token_sequences(
    texts: Iterable[str],
    tokenizer: tok_core.TokenizerCodec,
    *,
    text_normalization: normalization.TextNormalization = "none",
) -> Iterator[list[int]]:
    yield from ngram.iter_token_sequences(
        texts,
        tokenizer,
        bos_count=2,
        min_length=3,
        text_normalization=text_normalization,
    )


def context_key(previous_previous_id: int, previous_id: int) -> str:
    return f"{previous_previous_id},{previous_id}"


def parse_context_key(key: str) -> Context:
    previous_previous_id, previous_id = key.split(",", maxsplit=1)
    return int(previous_previous_id), int(previous_id)
