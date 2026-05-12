"""Good-Turing-smoothed token-level autoregressive trigram model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pydantic

from src.corpora import normalization
from src.models.core import formatting, ngram, trigrams
from src.tokenizers import core as tok_core


_SCHEMA_TYPE = "good_turing_trigram"
ProbabilityFn = Callable[[int], float]


@dataclass(frozen=True)
class GoodTuringDistribution:
    observed_probabilities: dict[int, float]
    reserved_mass: float
    lower_mass: float
    fallback_count: int

    def probability(
        self,
        token_id: int,
        *,
        lower_probability: ProbabilityFn,
    ) -> float:
        if token_id in self.observed_probabilities:
            return self.observed_probabilities[token_id]
        if self.reserved_mass <= 0:
            return 0.0
        if self.lower_mass > 0:
            return self.reserved_mass * lower_probability(token_id) / self.lower_mass
        if self.fallback_count > 0:
            return self.reserved_mass / self.fallback_count
        return 0.0


class GoodTuringTrigramModel(trigrams.BaseTrigramModel):
    unigram_counts: dict[int, int]
    unigram_total: int
    _unigram_distribution: GoodTuringDistribution | None = pydantic.PrivateAttr(
        default=None
    )
    _bigram_distributions: dict[int, GoodTuringDistribution] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _trigram_distributions: dict[trigrams.Context, GoodTuringDistribution] = (
        pydantic.PrivateAttr(default_factory=dict)
    )
    _unigram_ranked_token_ids: tuple[int, ...] = pydantic.PrivateAttr(default=())
    _bigram_ranked_token_ids: dict[int, tuple[int, ...]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def next_token_predictions(
        self,
        context: trigrams.Context,
        *,
        top_k: int,
    ) -> list[ngram.NgramPrediction]:
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        predictions = [
            ngram.NgramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=trigram_counts.get(token_id, 0),
                probability=self.trigram_probability(token_id, context),
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
        if next_id not in self.candidate_id_set:
            return 0.0
        if row is not None or all(
            value is None
            for value in (bigram_counts, trigram_counts, bigram_total, trigram_total)
        ):
            return self.trigram_probability(next_id, context)
        return super().transition_probability(
            next_id,
            context,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )

    def context_probability(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        p_lower = lambda token_id: self.bigram_probability(
            token_id,
            previous_id=counts.previous_id,
            counts=counts.bigram_counts,
            total=counts.bigram_total,
        )
        distribution = self.good_turing_distribution(
            counts.trigram_counts,
            total=counts.trigram_total,
            lower_probability=p_lower,
        )
        return distribution.probability(next_id, lower_probability=p_lower)

    def ranked_token_ids(
        self,
        context: trigrams.Context,
        *,
        bigram_counts: dict[int, int],
        trigram_counts: dict[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> list[int]:
        return self.top_token_ids(context, top_k=0)

    def evaluation_row(
        self,
        context: trigrams.Context,
        *,
        top_k: int,
    ) -> trigrams.TrigramEvaluationRow:
        previous_id = context[1]
        bigram_counts = dict(self.bigram_transitions.get(previous_id, ()))
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        ranked_token_ids = self.top_token_ids(context, top_k=top_k)
        return trigrams.TrigramEvaluationRow(
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=sum(bigram_counts.values()),
            trigram_total=sum(trigram_counts.values()),
            greedy_token_id=(
                ranked_token_ids[0]
                if ranked_token_ids
                else ngram.fallback_token_id(self.eos_id)
            ),
            top_k_token_ids=frozenset(ranked_token_ids[:top_k]) if top_k > 0 else frozenset(),
        )

    def top_token_ids(self, context: trigrams.Context, *, top_k: int) -> list[int]:
        distribution = self.trigram_distribution(context)
        previous_id = context[1]
        p_lower = lambda token_id: self.bigram_probability(
            token_id,
            previous_id=previous_id,
        )
        return ranked_distribution_token_ids(
            distribution,
            lower_ranked_token_ids=self.bigram_ranked_token_ids(previous_id),
            lower_probability=p_lower,
            top_k=top_k,
        )

    def trigram_probability(self, token_id: int, context: trigrams.Context) -> float:
        if token_id not in self.candidate_id_set:
            return 0.0
        previous_id = context[1]
        p_lower = lambda lower_id: self.bigram_probability(
            lower_id,
            previous_id=previous_id,
        )
        return self.trigram_distribution(context).probability(
            token_id,
            lower_probability=p_lower,
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
        if token_id not in self.candidate_id_set:
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

    def bigram_distribution(self, previous_id: int) -> GoodTuringDistribution:
        distribution = self._bigram_distributions.get(previous_id)
        if distribution is None:
            distribution = self.good_turing_distribution(
                dict(self.bigram_transitions.get(previous_id, ())),
                lower_probability=self.unigram_probability,
            )
            self._bigram_distributions[previous_id] = distribution
        return distribution

    def bigram_ranked_token_ids(self, previous_id: int) -> tuple[int, ...]:
        ranked = self._bigram_ranked_token_ids.get(previous_id)
        if ranked is None:
            distribution = self.bigram_distribution(previous_id)
            ranked = tuple(
                ranked_distribution_token_ids(
                    distribution,
                    lower_ranked_token_ids=self.unigram_ranked_token_ids(),
                    lower_probability=self.unigram_probability,
                    top_k=0,
                )
            )
            self._bigram_ranked_token_ids[previous_id] = ranked
        return ranked

    def unigram_probability(self, token_id: int) -> float:
        if token_id not in self.candidate_id_set:
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

    def unigram_ranked_token_ids(self) -> tuple[int, ...]:
        if not self._unigram_ranked_token_ids:
            distribution = self._unigram_distribution
            if distribution is None:
                distribution = self.good_turing_distribution(
                    self.unigram_counts,
                    total=self.unigram_total,
                    lower_probability=self.uniform_probability,
                )
                self._unigram_distribution = distribution
            self._unigram_ranked_token_ids = tuple(
                ranked_distribution_token_ids(
                    distribution,
                    lower_ranked_token_ids=self.candidate_ids,
                    lower_probability=self.uniform_probability,
                    top_k=0,
                )
            )
        return self._unigram_ranked_token_ids

    def uniform_probability(self, token_id: int) -> float:
        if token_id not in self.candidate_id_set or not self.candidate_ids:
            return 0.0
        return 1 / len(self.candidate_ids)

    def good_turing_distribution(
        self,
        counts: Mapping[int, int],
        *,
        lower_probability: ProbabilityFn,
        total: int | None = None,
    ) -> GoodTuringDistribution:
        return good_turing_distribution(
            counts,
            candidate_ids=self.candidate_ids,
            lower_probability=lower_probability,
            total=total,
        )


def good_turing_distribution(
    counts: Mapping[int, int],
    *,
    candidate_ids: tuple[int, ...],
    lower_probability: ProbabilityFn,
    total: int | None = None,
) -> GoodTuringDistribution:
    candidate_id_set = frozenset(candidate_ids)
    if not candidate_ids:
        return GoodTuringDistribution(
            observed_probabilities={},
            reserved_mass=0.0,
            lower_mass=0.0,
            fallback_count=0,
        )

    clean_counts = {
        token_id: int(count)
        for token_id, count in counts.items()
        if token_id in candidate_id_set and count > 0
    }
    unseen_count = max(len(candidate_ids) - len(clean_counts), 0)
    row_total = total if total is not None else sum(clean_counts.values())
    if row_total <= 0:
        return GoodTuringDistribution(
            observed_probabilities={},
            reserved_mass=1.0,
            lower_mass=1.0,
            fallback_count=len(candidate_ids),
        )

    Nr = Counter(clean_counts.values())
    adjusted_counts = {
        token_id: good_turing_count(count, Nr)
        for token_id, count in clean_counts.items()
    }
    adjusted_total = sum(adjusted_counts.values())
    raw_reserved_mass = Nr.get(1, 0) / row_total if unseen_count > 0 else 0.0
    raw_observed_mass = adjusted_total / row_total
    z = raw_observed_mass + raw_reserved_mass
    if z <= 0:
        adjusted_counts = {token_id: float(count) for token_id, count in clean_counts.items()}
        adjusted_total = sum(adjusted_counts.values())
        raw_observed_mass = adjusted_total / row_total
        z = raw_observed_mass + raw_reserved_mass

    observed_probabilities = (
        {
            token_id: (adjusted_count / row_total) / z
            for token_id, adjusted_count in adjusted_counts.items()
        }
        if z > 0
        else {}
    )
    observed_ids = frozenset(clean_counts)
    lower_mass = 1 - sum(lower_probability(token_id) for token_id in observed_ids)
    return GoodTuringDistribution(
        observed_probabilities=observed_probabilities,
        reserved_mass=raw_reserved_mass / z if z > 0 else 0.0,
        lower_mass=max(lower_mass, 0.0),
        fallback_count=unseen_count,
    )


def ranked_distribution_token_ids(
    distribution: GoodTuringDistribution,
    *,
    lower_ranked_token_ids: tuple[int, ...],
    lower_probability: ProbabilityFn,
    top_k: int,
) -> list[int]:
    observed_ids = frozenset(distribution.observed_probabilities)
    candidates = [
        (token_id, probability)
        for token_id, probability in distribution.observed_probabilities.items()
    ]

    unseen_limit = top_k if top_k > 0 else len(lower_ranked_token_ids)
    unseen_source = (
        lower_ranked_token_ids
        if distribution.reserved_mass > 0 and distribution.lower_mass > 0
        else tuple(sorted(lower_ranked_token_ids))
    )
    unseen_count = 0
    for token_id in unseen_source:
        if token_id in observed_ids:
            continue
        candidates.append(
            (
                token_id,
                distribution.probability(
                    token_id,
                    lower_probability=lower_probability,
                ),
            )
        )
        unseen_count += 1
        if unseen_count >= unseen_limit:
            break

    candidates.sort(key=lambda item: (-item[1], item[0]))
    token_ids = [token_id for token_id, _ in candidates]
    return token_ids[:top_k] if top_k > 0 else token_ids


def good_turing_count(count: int, Nr: Mapping[int, int]) -> float:
    next_frequency = Nr.get(count + 1, 0)
    if next_frequency <= 0:
        return float(count)
    return (count + 1) * next_frequency / Nr[count]


def load_good_turing_trigram_model(model_path: Path) -> GoodTuringTrigramModel:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return GoodTuringTrigramModel(
        **model_fields,
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
    tokenizer = tok_core.load_tokenizer(tokenizer_model)
    summary = trigrams.TrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=tokenizer.vocab_size,
        text_normalization=text_normalization,
    )
    counts = trigrams.collect_trigram_counts(
        texts,
        tokenizer,
        text_normalization=text_normalization,
    )
    trigrams.apply_trigram_counts_to_summary(summary, counts)

    model = trigrams.standard_trigram_model_payload(
        tokenizer,
        model_type=_SCHEMA_TYPE,
        tokenizer_model=tokenizer_model,
        stored_tokenizer_model=stored_tokenizer_model,
        text_normalization=text_normalization,
        counts=counts,
    )
    ngram.write_json_model_payload(output_path, model)

    return summary


def format_summary(summary: trigrams.TrigramTrainingSummary) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Good-Turing trigram model file",
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
