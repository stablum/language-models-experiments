"""Shared pieces for token-level trigram models."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import ClassVar

from src.corpora import normalization
from src.models.core import formatting, ngram
from src.tokenizers import core as tok_core


Context = tuple[int, int]  # h = (u, v), the trigram history.


class TrigramCounts(ngram.FrozenNgramModel):
    sequence_count: int
    token_count: int
    unigram_counts: Counter[int]  # c(w), unigram event counts.
    bigram_transitions: dict[int, Counter[int]]  # v -> c(v, w).
    trigram_transitions: dict[Context, Counter[int]]  # (u, v) -> c(u, v, w).
    bigram_transition_count: int  # sum_v c(v), bigram event count.
    trigram_transition_count: int  # sum_{u,v} c(u, v), trigram event count.

    @property
    def unigram_count(self) -> int:
        return sum(self.unigram_counts.values())


class TrigramTrainingArtifacts(ngram.FrozenNgramModel):
    tokenizer: tok_core.TokenizerCodec
    counts: TrigramCounts


class TrigramTrainingSummary(ngram.NgramTrainingSummary):
    unigram_count: int = 0
    bigram_transition_count: int = 0
    trigram_transition_count: int = 0


class InterpolatedTrigramTrainingSummary(TrigramTrainingSummary):
    unigram_weight: float = 0.0  # lambda_1.
    bigram_weight: float = 0.0  # lambda_2.
    trigram_weight: float = 0.0  # lambda_3.
    beta_2: float | None = None  # beta_2, lower-order bigram share.
    beta_3: float | None = None  # beta_3, trigram share.


class InterpolatedTrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    unigram_weight: float = 0.0  # lambda_1.
    bigram_weight: float = 0.0  # lambda_2.
    trigram_weight: float = 0.0  # lambda_3.
    beta_2: float | None = None  # beta_2, lower-order bigram share.
    beta_3: float | None = None  # beta_3, trigram share.


class ResolvedTrigramContextCounts(ngram.FrozenNgramModel):
    previous_id: int  # v, the second token in h = (u, v).
    bigram_counts: dict[int, int]  # c(v, w), the lower-order row.
    trigram_counts: dict[int, int]  # c(u, v, w), the trigram row.
    bigram_total: int  # c(v) = sum_w c(v, w).
    trigram_total: int  # c(u, v) = sum_w c(u, v, w).


class TrigramEvaluationRow(ngram.FrozenNgramModel):
    counts: ResolvedTrigramContextCounts
    greedy_token_id: int
    top_k_token_ids: frozenset[int]


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
        counts = self.context_counts(context)
        return ngram.sorted_predictions(
            (
                ngram.NgramPrediction(
                    token_id=token_id,
                    piece=self.pieces[token_id],
                    count=counts.trigram_counts.get(token_id, 0),
                    probability=self.transition_probability(
                        token_id,
                        context,
                        counts=counts,
                    ),
                )
                for token_id in self.candidate_ids
            ),
            top_k=top_k,
        )

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
                    probability=self.transition_probability(
                        next_id,
                        context,
                        counts=row.counts,
                    ),
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
        counts: ResolvedTrigramContextCounts | None = None,
    ) -> float:
        if next_id not in self.candidate_id_set:
            return 0.0

        if counts is None:
            counts = self.context_counts(context)
        return self.context_probability(next_id, counts)

    def context_probability(
        self,
        next_id: int,
        counts: ResolvedTrigramContextCounts,
    ) -> float:
        raise NotImplementedError

    def context_counts(
        self,
        context: Context,
    ) -> ResolvedTrigramContextCounts:
        previous_id = context[1]
        bigram_counts = dict(self.bigram_transitions.get(previous_id, ()))
        trigram_counts = dict(self.trigram_transitions.get(context, ()))

        return ResolvedTrigramContextCounts(
            previous_id=previous_id,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=sum(bigram_counts.values()),
            trigram_total=sum(trigram_counts.values()),
        )

    def evaluation_row(
        self,
        context: Context,
        *,
        top_k: int,
    ) -> TrigramEvaluationRow:
        counts = self.context_counts(context)
        ranked_token_ids = self.ranked_token_ids(
            context,
            counts=counts,
        )
        return TrigramEvaluationRow(
            counts=counts,
            greedy_token_id=ngram.greedy_token_id(ranked_token_ids, eos_id=self.eos_id),
            top_k_token_ids=ngram.top_k_token_id_set(ranked_token_ids, top_k=top_k),
        )

    def ranked_token_ids(
        self,
        context: Context,
        *,
        counts: ResolvedTrigramContextCounts,
    ) -> list[int]:
        return sorted(
            self.candidate_ids,
            key=lambda token_id: (
                -self.transition_probability(
                    token_id,
                    context,
                    counts=counts,
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


class InterpolatedTrigramModel(BaseTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        InterpolatedTrigramEvaluationSummary
    )
    unigram_weight: float  # lambda_1.
    bigram_weight: float  # lambda_2.
    trigram_weight: float  # lambda_3.
    beta_2: float | None = None  # beta_2, lower-order bigram share.
    beta_3: float | None = None  # beta_3, trigram share.

    def evaluation_summary_fields(self) -> dict[str, object]:
        return {
            "unigram_weight": self.unigram_weight,
            "bigram_weight": self.bigram_weight,
            "trigram_weight": self.trigram_weight,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
        }

    def context_probability(
        self,
        next_id: int,
        counts: ResolvedTrigramContextCounts,
    ) -> float:
        # next_id is w. This is lambda_1 P_1(w) + lambda_2 P_2(w | v)
        # + lambda_3 P_3(w | u, v).
        return (
            self.unigram_weight * self.unigram_probability(next_id)
            + self.bigram_weight * self.bigram_probability(
                next_id,
                counts=counts.bigram_counts,
                total=counts.bigram_total,
            )
            + self.trigram_weight * self.trigram_probability(
                next_id,
                counts=counts.trigram_counts,
                total=counts.trigram_total,
            )
        )

    def unigram_probability(self, token_id: int) -> float:
        raise NotImplementedError

    def bigram_probability(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        total: int,
    ) -> float:
        return self.conditional_probability(token_id, counts=counts, total=total)

    def trigram_probability(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        total: int,
    ) -> float:
        return self.conditional_probability(token_id, counts=counts, total=total)

    def conditional_probability(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        total: int,
    ) -> float:
        raise NotImplementedError


class DiscountedTrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    discount: float = 0.0  # D, the absolute discount.


class DiscountedTrigramModel(BaseTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        DiscountedTrigramEvaluationSummary
    )
    discount: float  # D, the absolute discount.

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
    unigram_counts: Counter[int] = Counter()  # c(w).
    bigram_transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    trigram_transitions: defaultdict[Context, Counter[int]] = defaultdict(Counter)
    # bigram_transitions[v][w] is c(v, w); trigram_transitions[(u, v)][w]
    # is c(u, v, w).
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
            # previous_id is v and next_id is w in c(v, w).
            bigram_transitions[previous_id][next_id] += 1
            bigram_transition_count += 1

        for previous_previous_id, previous_id, next_id in zip(
            token_ids,
            token_ids[1:],
            token_ids[2:],
        ):
            # previous_previous_id is u, previous_id is v, and next_id is w.
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


def collect_training_artifacts(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    text_normalization: normalization.TextNormalization = (
        normalization.DEFAULT_TEXT_NORMALIZATION
    ),
) -> TrigramTrainingArtifacts:
    tokenizer = tok_core.load_tokenizer(tokenizer_model)
    counts = collect_trigram_counts(
        texts,
        tokenizer,
        text_normalization=text_normalization,
    )
    return TrigramTrainingArtifacts(tokenizer=tokenizer, counts=counts)


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
