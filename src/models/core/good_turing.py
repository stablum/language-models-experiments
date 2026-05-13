"""Good-Turing row smoothing for fixed-vocabulary n-gram models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping

from src.models.core import ngram


ProbabilityFn = Callable[[int], float]


class GoodTuringDistribution(ngram.FrozenNgramModel):
    """One smoothed conditional row.

    ``observed_probabilities`` stores discounted probabilities for seen
    continuations. ``reserved_mass`` is the textbook N_1 / N mass for unseen
    continuations. ``lower_mass`` is the lower-order probability that remains
    after removing seen continuations, so backoff does not spend unseen mass on
    tokens already handled by Good-Turing counts.
    """

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
        # Good-Turing says how much total mass unseen events get, but not how
        # to split it; the lower-order model supplies that shape.
        if self.lower_mass > 0:
            return self.reserved_mass * lower_probability(token_id) / self.lower_mass
        # If the lower-order model is unusable for this unseen slice, fall back
        # to the plain "all unseen types are exchangeable" interpretation.
        if self.fallback_count > 0:
            return self.reserved_mass / self.fallback_count
        return 0.0


def distribution(
    counts: Mapping[int, int],
    *,
    candidate_ids: tuple[int, ...],
    lower_probability: ProbabilityFn,
    total: int | None = None,
) -> GoodTuringDistribution:
    """Build the Good-Turing row for one conditional history.

    ``counts`` is the empirical row c(h, w). The returned distribution contains
    explicit probabilities for observed continuations and enough bookkeeping to
    allocate the unseen-event mass through ``lower_probability``.
    """

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
        # With no evidence for this history, make the whole row a pure backoff
        # row. ``lower_mass`` stays at 1 because no observed events are removed.
        return GoodTuringDistribution(
            observed_probabilities={},
            reserved_mass=1.0,
            lower_mass=1.0,
            fallback_count=len(candidate_ids),
        )

    # N_r: count of token types observed exactly r times in this row.
    nr = Counter(clean_counts.values())
    adjusted_counts = {
        token_id: count_star(count, nr)
        for token_id, count in clean_counts.items()
    }
    adjusted_total = sum(adjusted_counts.values())
    # N_1 / N is the classic estimate for the total probability of unseen
    # types. If every candidate type has been observed, that mass has nowhere
    # valid to go inside this fixed vocabulary row.
    raw_reserved_mass = nr.get(1, 0) / row_total if unseen_count > 0 else 0.0
    raw_observed_mass = adjusted_total / row_total
    # Small rows often have gaps in N_r, so the adjusted counts plus unseen mass
    # do not necessarily sum to 1. z normalizes the row after discounting.
    z = raw_observed_mass + raw_reserved_mass
    if z <= 0:
        # Degenerate count-of-counts rows can discount everything away; keeping
        # raw counts is better than manufacturing an all-zero observed row.
        adjusted_counts = {
            token_id: float(count)
            for token_id, count in clean_counts.items()
        }
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
    # The lower-order model also gives probability to seen tokens. Remove that
    # part so reserved mass is renormalized only across unseen continuations.
    lower_mass = 1 - sum(lower_probability(token_id) for token_id in observed_ids)
    return GoodTuringDistribution(
        observed_probabilities=observed_probabilities,
        reserved_mass=raw_reserved_mass / z if z > 0 else 0.0,
        lower_mass=max(lower_mass, 0.0),
        fallback_count=unseen_count,
    )


def ranked_token_ids(
    distribution: GoodTuringDistribution,
    *,
    lower_ranked_token_ids: tuple[int, ...],
    lower_probability: ProbabilityFn,
    top_k: int,
) -> list[int]:
    """Rank observed and backoff-only continuations in one smoothed row."""

    observed_ids = frozenset(distribution.observed_probabilities)
    candidates = [
        (token_id, probability)
        for token_id, probability in distribution.observed_probabilities.items()
    ]

    unseen_limit = top_k if top_k > 0 else len(lower_ranked_token_ids)
    # When unseen probability is proportional to the lower-order row, the lower
    # ranking is already the right scan order; otherwise token id order gives a
    # stable fallback for the exchangeable-unseen case.
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


def count_star(count: int, nr: Mapping[int, int]) -> float:
    """Return c_star for one raw count using the count-of-counts table N_r."""

    next_frequency = nr.get(count + 1, 0)
    if next_frequency <= 0:
        # The unsmoothed estimator is undefined when N_{c+1} is missing. For
        # sparse language-model rows, preserving c avoids erasing rare maxima.
        return float(count)
    return (count + 1) * next_frequency / nr[count]
