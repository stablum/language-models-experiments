"""Interpolated Kneser-Ney token-level autoregressive trigram model."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import sentencepiece as spm

from src.corpora import normalization
from src.models.core import ngram
from src.models.core import trigrams


_SCHEMA_TYPE = "interpolated_kneser_ney_trigram"


class KneserNeyTrigramTrainingSummary(trigrams.TrigramTrainingSummary):
    continuation_unigram_count: int = 0
    continuation_bigram_type_count: int = 0
    discount: float = 0.0


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


class KneserNeyTrigramModel(trigrams.DiscountedTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        trigrams.DiscountedTrigramEvaluationSummary
    )
    unigram_counts: dict[int, int]
    unigram_total: int

    def context_probability(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        return self.trigram_probability(
            next_id,
            previous_id=counts.previous_id,
            bigram_counts=counts.bigram_counts,
            trigram_counts=counts.trigram_counts,
            bigram_total=counts.bigram_total,
            trigram_total=counts.trigram_total,
        )

    def trigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        bigram_counts: Mapping[int, int],
        trigram_counts: Mapping[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> float:
        lower_order_probability = self.bigram_probability(
            token_id,
            previous_id=previous_id,
            counts=bigram_counts,
            total=bigram_total,
        )
        return self._discounted_interpolation_probability(
            token_id,
            counts=trigram_counts,
            total=trigram_total,
            lower_order_probability=lower_order_probability,
        )

    def bigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        counts: Mapping[int, int] | None = None,
        total: int | None = None,
    ) -> float:
        if counts is None:
            counts = dict(self.bigram_transitions.get(previous_id, ()))
        if total is None:
            total = sum(counts.values())

        return self._discounted_interpolation_probability(
            token_id,
            counts=counts,
            total=total,
            lower_order_probability=self.unigram_probability(token_id),
        )

    def unigram_probability(self, token_id: int) -> float:
        candidate_count = ngram.candidate_token_count(self.vocab_size, self.bos_id)
        if candidate_count <= 0:
            return 0.0

        uniform_probability = 1 / candidate_count
        return self._discounted_interpolation_probability(
            token_id,
            counts=self.unigram_counts,
            total=self.unigram_total,
            lower_order_probability=uniform_probability,
        )

    def _discounted_interpolation_probability(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        total: int,
        lower_order_probability: float,
    ) -> float:
        if total <= 0:
            return lower_order_probability

        observed_count = counts.get(token_id, 0)
        discounted_probability = max(observed_count - self.discount, 0.0) / total
        interpolation_weight = self.discount * len(counts) / total
        return discounted_probability + interpolation_weight * lower_order_probability


def load_kneser_ney_trigram_model(model_path: Path) -> KneserNeyTrigramModel:
    data, tokenizer_model, processor, vocab_size = trigrams.load_standard_trigram_payload(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return KneserNeyTrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        **ngram.sentencepiece_model_fields(data, processor, vocab_size),
        discount=float(data["discount"]),
        unigram_counts=ngram.parse_token_counts(data, "kneser_ney_unigrams"),
        unigram_total=int(data["kneser_ney_unigram_count"]),
        bigram_transitions=ngram.parse_token_transitions(
            data,
            "kneser_ney_bigram_transitions",
        ),
        trigram_transitions=trigrams.parse_context_transitions(
            data,
            "trigram_transitions",
        ),
    )


def train_kneser_ney_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    discount: float = 0.75,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> KneserNeyTrigramTrainingSummary:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    summary = KneserNeyTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        discount=discount,
        text_normalization=text_normalization,
    )
    counts = trigrams.collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )
    continuation_counts = collect_kneser_ney_continuation_counts(
        counts.trigram_transitions,
    )
    trigrams.apply_trigram_counts_to_summary(summary, counts)
    summary.continuation_unigram_count = continuation_counts.unigram_count
    summary.continuation_bigram_type_count = continuation_counts.bigram_type_count

    model = {
        **trigrams.standard_trigram_model_payload(
            processor,
            model_type=_SCHEMA_TYPE,
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            vocab_size=summary.vocab_size,
            text_normalization=text_normalization,
            counts=counts,
        ),
        "discount": summary.discount,
        "kneser_ney_unigram_count": summary.continuation_unigram_count,
        "kneser_ney_unigrams": ngram.token_counts_payload(
            continuation_counts.unigram_counts
        ),
        "kneser_ney_bigram_transitions": ngram.token_transition_payload(
            continuation_counts.bigram_transitions
        ),
    }
    ngram.write_json_model_payload(output_path, model)

    return summary


def format_summary(
    summary: KneserNeyTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Interpolated Kneser-Ney trigram model artifact file",
        ),
        ("Continuation unigrams", f"{summary.continuation_unigram_count:,}"),
        ("Continuation bigram types", f"{summary.continuation_bigram_type_count:,}"),
        trigrams.discount_item(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train_kneser_ney_trigram_model,
    summary_items=format_summary,
    load_model=load_kneser_ney_trigram_model,
    evaluation_items=trigrams.discounted_evaluation_items,
    training_option_names=("discount",),
)


def collect_kneser_ney_continuation_counts(
    trigram_transitions: (
        defaultdict[trigrams.Context, Counter[int]]
        | dict[trigrams.Context, Counter[int]]
    ),
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
