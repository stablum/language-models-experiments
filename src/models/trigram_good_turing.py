"""Good-Turing-smoothed token-level autoregressive trigram model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import PrivateAttr
import sentencepiece as spm

from src.corpora import normalization
from src.models.core import formatting, ngram
from src.models.core import trigrams


_SCHEMA_TYPE = "good_turing_trigram"


@dataclass(frozen=True)
class GoodTuringDistribution:
    observed_probabilities: dict[int, float]
    unseen_mass: float
    backoff_mass: float
    unseen_count: int

    def probability(
        self,
        token_id: int,
        *,
        lower_probability: Callable[[int], float],
    ) -> float:
        if token_id in self.observed_probabilities:
            return self.observed_probabilities[token_id]
        if self.unseen_mass <= 0:
            return 0.0
        if self.backoff_mass > 0:
            return self.unseen_mass * lower_probability(token_id) / self.backoff_mass
        if self.unseen_count > 0:
            return self.unseen_mass / self.unseen_count
        return 0.0


class GoodTuringTrigramModel(trigrams.BaseTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        ngram.NgramEvaluationSummary
    )
    unigram_counts: dict[int, int]
    unigram_total: int
    _candidate_ids: tuple[int, ...] = PrivateAttr(default=())
    _candidate_id_set: frozenset[int] = PrivateAttr(default_factory=frozenset)
    _unigram_distribution: GoodTuringDistribution | None = PrivateAttr(default=None)
    _bigram_distributions: dict[int, GoodTuringDistribution] = PrivateAttr(
        default_factory=dict
    )
    _trigram_distributions: dict[trigrams.Context, GoodTuringDistribution] = PrivateAttr(
        default_factory=dict
    )

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        if not self._candidate_ids:
            self._candidate_ids = ngram.candidate_token_ids(self.vocab_size, self.bos_id)
        return self._candidate_ids

    @property
    def candidate_id_set(self) -> frozenset[int]:
        if not self._candidate_id_set:
            self._candidate_id_set = frozenset(self.candidate_ids)
        return self._candidate_id_set

    def next_token_predictions(
        self,
        context: trigrams.Context,
        *,
        top_k: int,
    ) -> list[ngram.NgramPrediction]:
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        distribution = self.trigram_distribution(context)
        previous_id = context[1]
        predictions = [
            ngram.NgramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=trigram_counts.get(token_id, 0),
                probability=distribution.probability(
                    token_id,
                    lower_probability=lambda lower_id: self.bigram_probability(
                        lower_id,
                        previous_id=previous_id,
                    ),
                ),
            )
            for token_id in self.candidate_ids
        ]
        predictions.sort(key=lambda prediction: (-prediction.probability, prediction.token_id))
        return predictions[:top_k] if top_k > 0 else predictions

    def transition_probability(
        self,
        next_id: int,
        context: trigrams.Context,
        *,
        row: trigrams.TrigramEvaluationRow | None = None,
        bigram_counts: dict[int, int] | None = None,
        trigram_counts: dict[int, int] | None = None,
        bigram_total: int | None = None,
        trigram_total: int | None = None,
    ) -> float:
        if next_id == self.bos_id or next_id not in self.candidate_id_set:
            return 0.0

        if row is not None or (
            bigram_counts is None
            and trigram_counts is None
            and bigram_total is None
            and trigram_total is None
        ):
            distribution = self.trigram_distribution(context)
            previous_id = context[1]
            return distribution.probability(
                next_id,
                lower_probability=lambda token_id: self.bigram_probability(
                    token_id,
                    previous_id=previous_id,
                ),
            )

        counts = self.resolved_context_counts(
            context,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )
        distribution = self.good_turing_distribution(
            counts.trigram_counts,
            total=counts.trigram_total,
            lower_probability=lambda token_id: self.bigram_probability(
                token_id,
                previous_id=counts.previous_id,
                counts=counts.bigram_counts,
                total=counts.bigram_total,
            ),
        )
        return distribution.probability(
            next_id,
            lower_probability=lambda token_id: self.bigram_probability(
                token_id,
                previous_id=counts.previous_id,
                counts=counts.bigram_counts,
                total=counts.bigram_total,
            ),
        )

    def context_probability(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        distribution = self.good_turing_distribution(
            counts.trigram_counts,
            total=counts.trigram_total,
            lower_probability=lambda token_id: self.bigram_probability(
                token_id,
                previous_id=counts.previous_id,
                counts=counts.bigram_counts,
                total=counts.bigram_total,
            ),
        )
        return distribution.probability(
            next_id,
            lower_probability=lambda token_id: self.bigram_probability(
                token_id,
                previous_id=counts.previous_id,
                counts=counts.bigram_counts,
                total=counts.bigram_total,
            ),
        )

    def ranked_token_ids(
        self,
        context: trigrams.Context,
        *,
        bigram_counts: dict[int, int],
        trigram_counts: dict[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> list[int]:
        distribution = self.trigram_distribution(context)
        previous_id = context[1]
        return sorted(
            self.candidate_ids,
            key=lambda token_id: (
                -distribution.probability(
                    token_id,
                    lower_probability=lambda lower_id: self.bigram_probability(
                        lower_id,
                        previous_id=previous_id,
                    ),
                ),
                token_id,
            ),
        )

    def trigram_distribution(
        self,
        context: trigrams.Context,
    ) -> GoodTuringDistribution:
        distribution = self._trigram_distributions.get(context)
        if distribution is None:
            previous_id = context[1]
            distribution = self.good_turing_distribution(
                dict(self.trigram_transitions.get(context, ())),
                lower_probability=lambda token_id: self.bigram_probability(
                    token_id,
                    previous_id=previous_id,
                ),
            )
            self._trigram_distributions[context] = distribution
        return distribution

    def bigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        counts: Mapping[int, int] | None = None,
        total: int | None = None,
    ) -> float:
        if token_id == self.bos_id or token_id not in self.candidate_id_set:
            return 0.0
        if counts is not None:
            distribution = self.good_turing_distribution(
                counts,
                total=total,
                lower_probability=self.unigram_probability,
            )
        else:
            distribution = self._bigram_distributions.get(previous_id)
            if distribution is None:
                distribution = self.good_turing_distribution(
                    dict(self.bigram_transitions.get(previous_id, ())),
                    lower_probability=self.unigram_probability,
                )
                self._bigram_distributions[previous_id] = distribution

        return distribution.probability(
            token_id,
            lower_probability=self.unigram_probability,
        )

    def unigram_probability(self, token_id: int) -> float:
        if token_id == self.bos_id or token_id not in self.candidate_id_set:
            return 0.0
        distribution = self._unigram_distribution
        if distribution is None:
            distribution = self.good_turing_distribution(
                self.unigram_counts,
                total=self.unigram_total,
                lower_probability=self.uniform_probability,
            )
            self._unigram_distribution = distribution
        return distribution.probability(
            token_id,
            lower_probability=self.uniform_probability,
        )

    def uniform_probability(self, token_id: int) -> float:
        if token_id == self.bos_id or token_id not in self.candidate_id_set:
            return 0.0
        candidate_count = len(self.candidate_ids)
        if candidate_count <= 0:
            return 0.0
        return 1 / candidate_count

    def good_turing_distribution(
        self,
        counts: Mapping[int, int],
        *,
        lower_probability: Callable[[int], float],
        total: int | None = None,
    ) -> GoodTuringDistribution:
        candidate_ids = self.candidate_ids
        if not candidate_ids:
            return GoodTuringDistribution(
                observed_probabilities={},
                unseen_mass=0.0,
                backoff_mass=0.0,
                unseen_count=0,
            )

        clean_counts = {
            token_id: int(count)
            for token_id, count in counts.items()
            if token_id in self.candidate_id_set and count > 0
        }
        observed_count = len(clean_counts)
        unseen_count = max(len(candidate_ids) - observed_count, 0)
        row_total = total if total is not None else sum(clean_counts.values())
        if row_total <= 0:
            return GoodTuringDistribution(
                observed_probabilities={},
                unseen_mass=1.0,
                backoff_mass=sum(lower_probability(token_id) for token_id in candidate_ids),
                unseen_count=len(candidate_ids),
            )

        frequency_counts = Counter(clean_counts.values())
        adjusted_counts = {
            token_id: good_turing_count(count, frequency_counts)
            for token_id, count in clean_counts.items()
        }
        adjusted_total = sum(adjusted_counts.values())
        raw_unseen_mass = (
            frequency_counts.get(1, 0) / row_total
            if unseen_count > 0
            else 0.0
        )
        raw_observed_mass = adjusted_total / row_total
        normalizer = raw_observed_mass + raw_unseen_mass
        if normalizer <= 0:
            adjusted_counts = {token_id: float(count) for token_id, count in clean_counts.items()}
            adjusted_total = sum(adjusted_counts.values())
            raw_observed_mass = adjusted_total / row_total
            normalizer = raw_observed_mass + raw_unseen_mass

        observed_probabilities = (
            {
                token_id: (adjusted_count / row_total) / normalizer
                for token_id, adjusted_count in adjusted_counts.items()
            }
            if normalizer > 0
            else {}
        )
        unseen_mass = raw_unseen_mass / normalizer if normalizer > 0 else 0.0
        observed_ids = frozenset(clean_counts)
        backoff_mass = sum(
            lower_probability(token_id)
            for token_id in candidate_ids
            if token_id not in observed_ids
        )
        return GoodTuringDistribution(
            observed_probabilities=observed_probabilities,
            unseen_mass=unseen_mass,
            backoff_mass=backoff_mass,
            unseen_count=unseen_count,
        )


def good_turing_count(count: int, frequency_counts: Mapping[int, int]) -> float:
    current_frequency = frequency_counts[count]
    next_frequency = frequency_counts.get(count + 1, 0)
    if next_frequency <= 0:
        return float(count)
    return (count + 1) * next_frequency / current_frequency


def load_good_turing_trigram_model(model_path: Path) -> GoodTuringTrigramModel:
    data, tokenizer_model, processor, vocab_size = trigrams.load_standard_trigram_payload(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return GoodTuringTrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        **ngram.sentencepiece_model_fields(data, processor, vocab_size),
        unigram_counts=trigrams.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def train_good_turing_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> trigrams.TrigramTrainingSummary:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    summary = trigrams.TrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        text_normalization=text_normalization,
    )
    counts = trigrams.collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )
    trigrams.apply_trigram_counts_to_summary(summary, counts)

    model = trigrams.standard_trigram_model_payload(
        processor,
        model_type=_SCHEMA_TYPE,
        tokenizer_model=tokenizer_model,
        stored_tokenizer_model=stored_tokenizer_model,
        vocab_size=summary.vocab_size,
        text_normalization=text_normalization,
        counts=counts,
    )
    ngram.write_json_model_payload(output_path, model)

    return summary


def format_summary(summary: trigrams.TrigramTrainingSummary) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Good-Turing trigram model artifact file",
        ),
        ("Smoothing", "Good-Turing"),
    ]


def format_evaluation(summary: ngram.NgramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        ("Smoothing", "Good-Turing"),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train_good_turing_trigram_model,
    summary_items=format_summary,
    load_model=load_good_turing_trigram_model,
    evaluation_items=format_evaluation,
)
