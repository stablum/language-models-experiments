"""Good-Turing-smoothed token-level autoregressive trigram model.

The textbook estimator replaces a raw count c with
``c_star = (c + 1) * N_{c+1} / N_c``, where N_r is the number of event types
seen exactly r times. In this implementation an "event type" is one candidate
next token inside a single conditional row, such as ``P(next | prev2, prev1)``.
The mass reserved for unseen continuations is then backed off to the next lower
order model.

Notation in comments uses ``h = (u, v)`` for trigram histories, ``w`` for the
candidate next token, and ``N_r`` for count-of-counts within one row.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pydantic

from src.corpora import normalization
from src.models.core import formatting
from src.models.core import good_turing
from src.models.core import ngram
from src.models.core import trigrams


_SCHEMA_TYPE = "good_turing_trigram"


class GoodTuringTrigramModel(trigrams.BaseTrigramModel):
    unigram_counts: dict[int, int]  # c(w), unigram counts.
    unigram_total: int  # N = sum_w c(w), the unigram normalizer.
    _unigram_distribution: good_turing.GoodTuringDistribution | None = (
        pydantic.PrivateAttr(default=None)
    )
    _bigram_distributions: dict[int, good_turing.GoodTuringDistribution] = (
        pydantic.PrivateAttr(default_factory=dict)
    )
    _trigram_distributions: dict[trigrams.Context, good_turing.GoodTuringDistribution] = (
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
        # context is h = (u, v). trigram_counts[token_id] is c(h, w).
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        return [
            ngram.NgramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=trigram_counts.get(token_id, 0),
                probability=self.trigram_probability(token_id, context),
            )
            for token_id in self.top_token_ids(context, top_k=top_k)
        ]

    def transition_probability(
        self,
        next_id: int,
        context: trigrams.Context,
        *,
        counts: trigrams.ResolvedTrigramContextCounts | None = None,
    ) -> float:
        if next_id not in self.candidate_id_set:
            return 0.0
        return self.trigram_probability(next_id, context)

    def context_probability(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        # next_id is w. counts.trigram_counts is the empirical row c(h, w).
        # Trigram Good-Turing only discounts the current history row. Any mass
        # reserved for unseen trigram continuations backs off to the bigram row.
        def lower_probability(token_id: int) -> float:
            # P_lower(w) is the Good-Turing-smoothed bigram row P(w | v).
            return self.bigram_probability(
                token_id,
                previous_id=counts.previous_id,
                counts=counts.bigram_counts,
                total=counts.bigram_total,
            )

        distribution = self.good_turing_distribution(
            counts.trigram_counts,
            total=counts.trigram_total,
            lower_probability=lower_probability,
        )
        return distribution.probability(next_id, lower_probability=lower_probability)

    def ranked_token_ids(
        self,
        context: trigrams.Context,
        *,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> list[int]:
        return self.top_token_ids(context, top_k=0)

    def evaluation_row(
        self,
        context: trigrams.Context,
        *,
        top_k: int,
    ) -> trigrams.TrigramEvaluationRow:
        counts = self.context_counts(context)
        ranked_token_ids = self.top_token_ids(context, top_k=top_k)
        return trigrams.TrigramEvaluationRow(
            counts=counts,
            greedy_token_id=ngram.greedy_token_id(ranked_token_ids, eos_id=self.eos_id),
            top_k_token_ids=ngram.top_k_token_id_set(ranked_token_ids, top_k=top_k),
        )

    def top_token_ids(self, context: trigrams.Context, *, top_k: int) -> list[int]:
        distribution = self.trigram_distribution(context)
        previous_id = context[1]  # v, the lower-order bigram history.
        # Ranking unseen trigram continuations follows the same lower-order
        # backoff distribution used when assigning their reserved probability.
        def lower_probability(token_id: int) -> float:
            return self.bigram_probability(
                token_id,
                previous_id=previous_id,
            )

        return good_turing.ranked_token_ids(
            distribution,
            lower_ranked_token_ids=self.bigram_ranked_token_ids(previous_id),
            lower_probability=lower_probability,
            top_k=top_k,
        )

    def trigram_probability(self, token_id: int, context: trigrams.Context) -> float:
        if token_id not in self.candidate_id_set:
            return 0.0
        previous_id = context[1]  # v in h = (u, v); token_id is w.
        # Textbook backoff chain for unseen events:
        # trigram row -> bigram row for the same previous token.
        def lower_probability(token_id: int) -> float:
            return self.bigram_probability(
                token_id,
                previous_id=previous_id,
            )

        return self.trigram_distribution(context).probability(
            token_id,
            lower_probability=lower_probability,
        )

    def trigram_distribution(
        self,
        context: trigrams.Context,
    ) -> good_turing.GoodTuringDistribution:
        distribution = self._trigram_distributions.get(context)
        if distribution is None:
            previous_id = context[1]  # v, the lower-order history in P(w | v).

            def lower_probability(token_id: int) -> float:
                return self.bigram_probability(
                    token_id,
                    previous_id=previous_id,
                )

            distribution = self.good_turing_distribution(
                dict(self.trigram_transitions.get(context, ())),
                lower_probability=lower_probability,
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
            # counts[token_id] is c(v, w), with total c(v).
            distribution = self.good_turing_distribution(
                counts,
                total=total,
                lower_probability=self.unigram_probability,
            )
        else:
            distribution = self.bigram_distribution(previous_id)

        return distribution.probability(
            token_id,
            lower_probability=self.unigram_probability,
        )

    def bigram_distribution(self, previous_id: int) -> good_turing.GoodTuringDistribution:
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
                good_turing.ranked_token_ids(
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
        # token_id is w. Good-Turing smooths the unigram row c(w).
        return self.unigram_distribution().probability(
            token_id,
            lower_probability=self.uniform_probability,
        )

    def unigram_ranked_token_ids(self) -> tuple[int, ...]:
        if not self._unigram_ranked_token_ids:
            self._unigram_ranked_token_ids = tuple(
                good_turing.ranked_token_ids(
                    self.unigram_distribution(),
                    lower_ranked_token_ids=self.candidate_ids,
                    lower_probability=self.uniform_probability,
                    top_k=0,
                )
            )
        return self._unigram_ranked_token_ids

    def unigram_distribution(self) -> good_turing.GoodTuringDistribution:
        distribution = self._unigram_distribution
        if distribution is None:
            distribution = self.good_turing_distribution(
                self.unigram_counts,
                total=self.unigram_total,
                lower_probability=self.uniform_probability,
            )
            self._unigram_distribution = distribution
        return distribution

    def uniform_probability(self, token_id: int) -> float:
        if token_id not in self.candidate_id_set or self.candidate_count <= 0:
            return 0.0
        # P_0(w) = 1 / |V| is the final backoff distribution.
        return 1 / self.candidate_count

    def good_turing_distribution(
        self,
        counts: Mapping[int, int],
        *,
        lower_probability: good_turing.ProbabilityFn,
        total: int | None = None,
    ) -> good_turing.GoodTuringDistribution:
        # Build a Good-Turing row from c(h, w) and its lower-order P_lower(w).
        return good_turing.distribution(
            counts,
            candidate_ids=self.candidate_ids,
            lower_probability=lower_probability,
            total=total,
        )


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
    artifacts = trigrams.collect_training_artifacts(
        texts,
        tokenizer_model=tokenizer_model,
        text_normalization=text_normalization,
    )
    summary = trigrams.TrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=artifacts.tokenizer.vocab_size,
        text_normalization=text_normalization,
    )
    trigrams.apply_trigram_counts_to_summary(summary, artifacts.counts)

    model = trigrams.standard_trigram_model_payload(
        artifacts.tokenizer,
        model_type=_SCHEMA_TYPE,
        tokenizer_model=tokenizer_model,
        stored_tokenizer_model=stored_tokenizer_model,
        text_normalization=text_normalization,
        counts=artifacts.counts,
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
