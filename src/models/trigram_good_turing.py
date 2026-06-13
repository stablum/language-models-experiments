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

from collections.abc import Mapping
from pathlib import Path

import pydantic

from src.models.core import formatting
from src.models.core import good_turing
from src.models.core import ngram
from src.models.core import token_sequences
from src.models.core import trigrams


CONTEXT_LENGTH = trigrams.CONTEXT_LENGTH  # len(h), inherited trigram history size.


class TrainingSummary(trigrams.TrigramTrainingSummary):
    """Store the smoothing family label with Good-Turing training counts."""

    smoothing_label: str = "Good-Turing"


class Model(trigrams.BaseTrigramModel):
    unigram_counts: dict[int, int]  # c(w), unigram counts.
    unigram_tot: int  # tot = N = sum_w c(w), the unigram normalizer.
    _unigram_distribution: good_turing.GoodTuringDistribution | None = (
        pydantic.PrivateAttr(default=None)
    )
    _bigram_distributions: dict[int, good_turing.GoodTuringDistribution] = (
        pydantic.PrivateAttr(default_factory=dict)
    )
    _trigram_distributions: dict[trigrams.Context, good_turing.GoodTuringDistribution] = (
        pydantic.PrivateAttr(default_factory=dict)
    )
    _unigram_ranked_ids: tuple[int, ...] = pydantic.PrivateAttr(default=())
    _bigram_ranked_ids: dict[int, tuple[int, ...]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def next_token_predictions(
        self,
        context: trigrams.Context,
        *,
        top_k: int,
    ) -> list[ngram.NgramPrediction]:
        # context is h = (u, v). trigram_counts[token_id] is c(h, w).
        counts = self.context_counts(context)
        return [
            ngram.NgramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=counts.trigram_counts.get(token_id, 0),
                prob=self._trigram_prob(token_id, context),
            )
            for token_id in self._top_ids(context, top_k=top_k)
        ]

    def transition_prob(
        self,
        next_id: int,
        context: trigrams.Context,
        *,
        counts: trigrams.ResolvedTrigramContextCounts | None = None,
    ) -> float:
        if next_id not in self.cand_id_set:
            return 0.0
        return self._trigram_prob(next_id, context)

    def context_prob(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        # next_id is w. counts.trigram_counts is the empirical row c(h, w).
        # Trigram Good-Turing only discounts the current history row. Any mass
        # reserved for unseen trigram continuations backs off to the bigram row.
        def lower_prob(token_id: int) -> float:
            # P_lower(w) is the Good-Turing-smoothed bigram row P(w | v).
            return self._bigram_prob(
                token_id,
                prev_id=counts.prev_id,
                counts=counts.bigram_counts,
                tot=counts.bigram_tot,
            )

        distribution = self._good_turing_dist(
            counts.trigram_counts,
            tot=counts.trigram_tot,
            lower_prob=lower_prob,
        )
        return distribution.prob(next_id, lower_prob=lower_prob)

    def ranked_ids(
        self,
        context: trigrams.Context,
        *,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> list[int]:
        return self._top_ids(context, top_k=0)

    def evaluation_row(
        self,
        context: trigrams.Context,
        *,
        top_k: int,
    ) -> trigrams.TrigramEvaluationRow:
        counts = self.context_counts(context)
        ranked_ids = self._top_ids(context, top_k=top_k)
        return trigrams.TrigramEvaluationRow(
            counts=counts,
            greedy_id=ngram.greedy_id(ranked_ids, eos_id=self.eos_id),
            top_k_ids=ngram.top_k_id_set(ranked_ids, top_k=top_k),
        )

    def _top_ids(self, context: trigrams.Context, *, top_k: int) -> list[int]:
        distribution = self._trigram_dist(context)
        prev_id = context[1]  # prev = v, the lower-order bigram history.
        # Ranking unseen trigram continuations follows the same lower-order
        # backoff distribution used when assigning their reserved probability.
        def lower_prob(token_id: int) -> float:
            return self._bigram_prob(
                token_id,
                prev_id=prev_id,
            )

        return good_turing.ranked_ids(
            distribution,
            lower_ranked_ids=self._ranked_bigram_ids(prev_id),
            lower_prob=lower_prob,
            top_k=top_k,
        )

    def _trigram_prob(self, token_id: int, context: trigrams.Context) -> float:
        if token_id not in self.cand_id_set:
            return 0.0
        prev_id = context[1]  # prev = v in h = (u, v); token_id is w.
        # Textbook backoff chain for unseen events:
        # trigram row -> bigram row for the same previous token.
        def lower_prob(token_id: int) -> float:
            return self._bigram_prob(
                token_id,
                prev_id=prev_id,
            )

        return self._trigram_dist(context).prob(
            token_id,
            lower_prob=lower_prob,
        )

    def _trigram_dist(
        self,
        context: trigrams.Context,
    ) -> good_turing.GoodTuringDistribution:
        distribution = self._trigram_distributions.get(context)
        if distribution is None:
            prev_id = context[1]  # prev = v, the lower-order history in P(w | v).

            def lower_prob(token_id: int) -> float:
                return self._bigram_prob(
                    token_id,
                    prev_id=prev_id,
                )

            trigram_row = self.trigram_transitions.get(context, ())
            trigram_counts = self.candidate_counts(trigram_row)
            distribution = self._good_turing_dist(trigram_counts, lower_prob=lower_prob)
            self._trigram_distributions[context] = distribution
        return distribution

    def _bigram_prob(
        self,
        token_id: int,
        *,
        prev_id: int,
        counts: Mapping[int, int] | None = None,
        tot: int | None = None,
    ) -> float:
        if token_id not in self.cand_id_set:
            return 0.0
        if counts is not None:
            # counts[token_id] is c(v, w), with tot = c(v).
            distribution = self._good_turing_dist(
                counts,
                tot=tot,
                lower_prob=self._unigram_prob,
            )
        else:
            distribution = self._bigram_dist(prev_id)

        return distribution.prob(
            token_id,
            lower_prob=self._unigram_prob,
        )

    def _bigram_dist(self, prev_id: int) -> good_turing.GoodTuringDistribution:
        distribution = self._bigram_distributions.get(prev_id)
        if distribution is None:
            bigram_row = self.bigram_transitions.get(prev_id, ())
            bigram_counts = self.candidate_counts(bigram_row)
            distribution = self._good_turing_dist(
                bigram_counts,
                lower_prob=self._unigram_prob,
            )
            self._bigram_distributions[prev_id] = distribution
        return distribution

    def _ranked_bigram_ids(self, prev_id: int) -> tuple[int, ...]:
        ranked = self._bigram_ranked_ids.get(prev_id)
        if ranked is None:
            distribution = self._bigram_dist(prev_id)
            ranked = tuple(
                good_turing.ranked_ids(
                    distribution,
                    lower_ranked_ids=self._ranked_unigram_ids(),
                    lower_prob=self._unigram_prob,
                    top_k=0,
                )
            )
            self._bigram_ranked_ids[prev_id] = ranked
        return ranked

    def _unigram_prob(self, token_id: int) -> float:
        if token_id not in self.cand_id_set:
            return 0.0
        # token_id is w. Good-Turing smooths the unigram row c(w).
        return self._unigram_dist().prob(
            token_id,
            lower_prob=self._uniform_prob,
        )

    def _ranked_unigram_ids(self) -> tuple[int, ...]:
        if not self._unigram_ranked_ids:
            self._unigram_ranked_ids = tuple(
                good_turing.ranked_ids(
                    self._unigram_dist(),
                    lower_ranked_ids=self.cand_ids,
                    lower_prob=self._uniform_prob,
                    top_k=0,
                )
            )
        return self._unigram_ranked_ids

    def _unigram_dist(self) -> good_turing.GoodTuringDistribution:
        distribution = self._unigram_distribution
        if distribution is None:
            distribution = self._good_turing_dist(
                self.unigram_counts,
                tot=self.unigram_tot,
                lower_prob=self._uniform_prob,
            )
            self._unigram_distribution = distribution
        return distribution

    def _uniform_prob(self, token_id: int) -> float:
        if token_id not in self.cand_id_set or self.cand_count <= 0:
            return 0.0
        # P_0(w) = 1 / |V| is the final backoff distribution.
        return 1 / self.cand_count

    def _good_turing_dist(
        self,
        counts: Mapping[int, int],
        *,
        lower_prob: good_turing.ProbFn,
        tot: int | None = None,
    ) -> good_turing.GoodTuringDistribution:
        # Build a Good-Turing row from c(h, w) and its lower-order P_lower(w).
        return good_turing.distribution(
            counts,
            cand_ids=self.cand_ids,
            lower_prob=lower_prob,
            tot=tot,
        )


def load(model_path: Path) -> Model:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        module_name=__name__,
    )

    return Model(
        **model_fields,
        unigram_counts=trigrams.parse_unigram_counts(data),
        unigram_tot=int(data["unigram_count"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def fit(
    corpus: token_sequences.TokenCorpus,
) -> ngram.TrainingResult[TrainingSummary]:
    """Fit trigram counts used by Good-Turing probability rows."""
    return trigrams.fit_counted_trigram_model(
        corpus,
        summary_type=TrainingSummary,
    )


def format_evaluation(summary: ngram.NgramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        ("Smoothing", "Good-Turing"),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]
