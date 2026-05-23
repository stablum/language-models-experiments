"""Shared prediction-event counting for token-level n-gram models."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence

from src.models.core import ngram


type Context = tuple[int, ...]  # h, the n-gram history before predicted token w.
type TransitionRows = dict[Context, Counter[int]]


class NgramOrderCounts(ngram.FrozenNgramModel):
    order: int  # n in c(h, w), with len(h) = n - 1.
    rows: TransitionRows  # h -> c(h, w).
    event_count: int  # sum_h c(h), the number of predicted tokens.

    @property
    def token_counts(self) -> Counter[int]:
        if self.order != 1:
            raise ValueError("token_counts is only defined for unigram counts")
        return Counter(self.rows.get((), Counter()))


class NgramCorpusCounts(ngram.FrozenNgramModel):
    sequence_count: int
    token_count: int
    orders: dict[int, NgramOrderCounts]

    def order_counts(self, order: int) -> NgramOrderCounts:
        try:
            return self.orders[order]
        except KeyError as error:
            raise KeyError(f"No counts collected for {order}-grams") from error

    def rows(self, order: int) -> TransitionRows:
        return self.order_counts(order).rows

    def event_count(self, order: int) -> int:
        return self.order_counts(order).event_count

    def token_counts(self, order: int = 1) -> Counter[int]:
        return self.order_counts(order).token_counts


def collect_ngram_counts(
    token_seqs: Iterable[Sequence[int]],
    *,
    orders: Iterable[int],
    prediction_order: int,
) -> NgramCorpusCounts:
    """Count requested lower-order rows on the same prediction frontier.

    ``prediction_order`` chooses which tokens are prediction events. For a
    trigram model this is 3, so unigram and bigram backing counts are aligned to
    tokens predicted with two-token histories instead of including earlier BOS
    warm-up events.
    """

    normalized_orders = normalize_orders(orders, prediction_order=prediction_order)
    rows_by_order: dict[int, defaultdict[Context, Counter[int]]] = {
        order: defaultdict(Counter)
        for order in normalized_orders
    }
    event_counts = {order: 0 for order in normalized_orders}
    sequence_count = 0
    token_count = 0

    for token_ids in token_seqs:
        sequence_count += 1
        token_count += len(token_ids)
        for next_idx in prediction_indices(token_ids, order=prediction_order):
            next_id = token_ids[next_idx]  # w, the predicted token.
            for order in normalized_orders:
                context = context_at(token_ids, next_idx, order=order)
                rows_by_order[order][context][next_id] += 1
                event_counts[order] += 1

    return NgramCorpusCounts(
        sequence_count=sequence_count,
        token_count=token_count,
        orders={
            order: NgramOrderCounts(
                order=order,
                rows=dict(rows_by_order[order]),
                event_count=event_counts[order],
            )
            for order in normalized_orders
        },
    )


def normalize_orders(orders: Iterable[int], *, prediction_order: int) -> tuple[int, ...]:
    if prediction_order < 1:
        raise ValueError("prediction_order must be positive")

    normalized_orders = tuple(sorted(set(orders)))
    if not normalized_orders:
        raise ValueError("At least one n-gram order must be requested")

    bad_orders = [
        order
        for order in normalized_orders
        if order < 1 or order > prediction_order
    ]
    if bad_orders:
        order_list = ", ".join(str(order) for order in bad_orders)
        raise ValueError(
            f"N-gram orders must be in [1, {prediction_order}]: {order_list}"
        )
    return normalized_orders


def iter_prediction_events(
    token_ids: Sequence[int],
    *,
    order: int,
) -> Iterator[tuple[Context, int]]:
    for next_idx in prediction_indices(token_ids, order=order):
        yield context_at(token_ids, next_idx, order=order), token_ids[next_idx]


def prediction_indices(token_ids: Sequence[int], *, order: int) -> range:
    if order < 1:
        raise ValueError("order must be positive")
    return range(order - 1, len(token_ids))


def context_at(token_ids: Sequence[int], next_idx: int, *, order: int) -> Context:
    if order < 1:
        raise ValueError("order must be positive")

    context_start = next_idx - order + 1
    if context_start < 0:
        raise ValueError("Not enough previous tokens for requested n-gram order")
    return tuple(token_ids[context_start:next_idx])


def observe_sequence(
    summary: ngram.NgramEvaluationSummary,
    token_ids: Sequence[int],
) -> None:
    summary.sequence_count += 1
    summary.token_count += len(token_ids)


def score_evaluation_event(
    summary: ngram.NgramEvaluationSummary,
    *,
    actual_token_id: int,
    greedy_token_id: int,
    top_k_token_ids: frozenset[int],
    probability: float,
) -> None:
    summary.transition_count += 1
    ngram.score_evaluation_transition(
        summary,
        actual_token_id=actual_token_id,
        greedy_token_id=greedy_token_id,
        top_k_token_ids=top_k_token_ids,
        probability=probability,
    )


def single_token_context_rows(
    rows: Mapping[Context, Counter[int]],
) -> dict[int, Counter[int]]:
    return {
        single_token_context_id(context): Counter(next_counts)
        for context, next_counts in rows.items()
    }


def single_token_context_id(context: Context) -> int:
    if len(context) != 1:
        raise ValueError(f"Expected a 1-token context, got {len(context)}")
    return context[0]
