"""Good-Turing row smoothing for fixed-vocabulary n-gram models.

For an event with raw count ``c``, Good-Turing uses
``c_star = (c + 1) N_{c+1} / N_c``. Within one history row ``h``, ``N_r`` is
the number of candidate next-token types ``w`` with ``c(h, w) = r``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping

from src.models.core import ngram


ProbFn = Callable[[int], float]


class GoodTuringDistribution(ngram.FrozenNgramSchema):
    """One smoothed conditional row.

    ``obs_probs`` stores discounted probabilities for seen
    continuations. ``reserved_mass`` is the textbook N_1 / N mass for unseen
    continuations. ``lower_mass`` is the lower-order probability that remains
    after removing seen continuations, so backoff does not spend unseen mass on
    tokens already handled by Good-Turing counts.
    """

    obs_probs: dict[int, float]  # obs = observed P_GT(w | h) for seen w.
    reserved_mass: float  # P_0 = N_1 / N, total mass for unseen w.
    lower_mass: float  # sum_{unseen w} P_lower(w).
    fallback_count: int  # number of unseen types when P_lower has no mass.

    def prob(
        self,
        token_id: int,
        *,
        lower_prob: ProbFn,
    ) -> float:
        if token_id in self.obs_probs:
            return self.obs_probs[token_id]
        if self.reserved_mass <= 0:
            return 0.0
        # Good-Turing says how much total mass unseen events get, but not how
        # to split it; the lower-order model supplies that shape.
        if self.lower_mass > 0:
            # P(w | h) = P_0 * P_lower(w) / sum_{unseen w'} P_lower(w').
            return self.reserved_mass * lower_prob(token_id) / self.lower_mass
        # If the lower-order model is unusable for this unseen slice, fall back
        # to the plain "all unseen types are exchangeable" interpretation.
        if self.fallback_count > 0:
            return self.reserved_mass / self.fallback_count
        return 0.0


def distribution(
    counts: Mapping[int, int],
    *,
    cand_ids: tuple[int, ...],
    lower_prob: ProbFn,
    tot: int | None = None,
) -> GoodTuringDistribution:
    """Build the Good-Turing row for one conditional history.

    ``counts`` is the empirical row c(h, w). The returned distribution contains
    explicit probabilities for observed continuations and enough bookkeeping to
    allocate the unseen-event mass through ``lower_prob``.
    """

    cand_id_set = frozenset(cand_ids)  # cand = V, the candidate vocabulary.
    if not cand_ids:
        return GoodTuringDistribution(
            obs_probs={},
            reserved_mass=0.0,
            lower_mass=0.0,
            fallback_count=0,
        )

    clean_counts = {
        token_id: int(count)
        for token_id, count in counts.items()
        if token_id in cand_id_set and count > 0
    }  # positive c(h, w) values inside V.
    unseen_count = max(len(cand_ids) - len(clean_counts), 0)  # N_0.
    row_tot = tot if tot is not None else sum(clean_counts.values())  # c(h).
    if row_tot <= 0:
        # With no evidence for this history, make the whole row a pure backoff
        # row. ``lower_mass`` stays at 1 because no observed events are removed.
        return GoodTuringDistribution(
            obs_probs={},
            reserved_mass=1.0,
            lower_mass=1.0,
            fallback_count=len(cand_ids),
        )

    # N_r: count of token types observed exactly r times in this row.
    nr = Counter(clean_counts.values())
    adj_counts = {
        token_id: count_star(count, nr)
        for token_id, count in clean_counts.items()
    }  # c_star(h, w) for seen events.
    adj_tot = sum(adj_counts.values())  # adj = sum_w c_star(h, w).
    # N_1 / N is the classic estimate for the total probability of unseen
    # types. If every candidate type has been observed, that mass has nowhere
    # valid to go inside this fixed vocabulary row.
    raw_reserved_mass = nr.get(1, 0) / row_tot if unseen_count > 0 else 0.0
    raw_obs_mass = adj_tot / row_tot  # obs = unnormalized seen mass.
    # Small rows often have gaps in N_r, so the adjusted counts plus unseen mass
    # do not necessarily sum to 1. z normalizes the row after discounting.
    z = raw_obs_mass + raw_reserved_mass  # row normalization constant Z.
    if z <= 0:
        # Degenerate count-of-counts rows can discount everything away; keeping
        # raw counts is better than manufacturing an all-zero observed row.
        adj_counts = {
            token_id: float(count)
            for token_id, count in clean_counts.items()
        }
        adj_tot = sum(adj_counts.values())
        raw_obs_mass = adj_tot / row_tot
        z = raw_obs_mass + raw_reserved_mass

    obs_probs = (
        {
            token_id: (adj_count / row_tot) / z
            for token_id, adj_count in adj_counts.items()
        }
        if z > 0
        else {}
    )
    obs_ids = frozenset(clean_counts)
    # The lower-order model also gives probability to seen tokens. Remove that
    # part so reserved mass is renormalized only across unseen continuations.
    lower_mass = 1 - sum(lower_prob(token_id) for token_id in obs_ids)
    return GoodTuringDistribution(
        obs_probs=obs_probs,
        reserved_mass=raw_reserved_mass / z if z > 0 else 0.0,
        lower_mass=max(lower_mass, 0.0),
        fallback_count=unseen_count,
    )


def ranked_ids(
    distribution: GoodTuringDistribution,
    *,
    lower_ranked_ids: tuple[int, ...],
    lower_prob: ProbFn,
    top_k: int,
) -> list[int]:
    """Rank observed and backoff-only continuations in one smoothed row."""

    obs_ids = frozenset(distribution.obs_probs)
    candidates = [
        (token_id, prob)
        for token_id, prob in distribution.obs_probs.items()
    ]

    unseen_limit = top_k if top_k > 0 else len(lower_ranked_ids)
    # When unseen probability is proportional to the lower-order row, the lower
    # ranking is already the right scan order; otherwise token id order gives a
    # stable fallback for the exchangeable-unseen case.
    unseen_source = (
        lower_ranked_ids
        if distribution.reserved_mass > 0 and distribution.lower_mass > 0
        else tuple(sorted(lower_ranked_ids))
    )
    unseen_count = 0
    for token_id in unseen_source:
        if token_id in obs_ids:
            continue
        candidates.append(
            (
                token_id,
                distribution.prob(
                    token_id,
                    lower_prob=lower_prob,
                ),
            )
        )
        unseen_count += 1
        if unseen_count >= unseen_limit:
            break

    candidates.sort(key=lambda item: (-item[1], item[0]))
    ranked_ids = [token_id for token_id, _ in candidates]
    return ranked_ids[:top_k] if top_k > 0 else ranked_ids


def count_star(count: int, nr: Mapping[int, int]) -> float:
    """Return c_star for one raw count using the count-of-counts table N_r."""

    next_frequency = nr.get(count + 1, 0)
    if next_frequency <= 0:
        # The unsmoothed estimator is undefined when N_{c+1} is missing. For
        # sparse language-model rows, preserving c avoids erasing rare maxima.
        return float(count)
    # c_star = (c + 1) N_{c+1} / N_c.
    return (count + 1) * next_frequency / nr[count]
