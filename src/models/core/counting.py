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
        """Return unigram counts as c(w) for order-1 rows."""
        if self.order != 1:
            raise ValueError("token_counts is only defined for unigram counts")
        return Counter(self.rows.get((), Counter()))


class NgramCorpusCounts(ngram.FrozenNgramModel):
    sequence_count: int
    token_count: int
    orders: dict[int, NgramOrderCounts]

    def order_counts(self, order: int) -> NgramOrderCounts:
        """Return collected counts for a requested n-gram order."""
        try:
            return self.orders[order]
        except KeyError as error:
            raise KeyError(f"No counts collected for {order}-grams") from error

    def rows(self, order: int) -> TransitionRows:
        """Return transition rows h -> c(h, w) for one n-gram order."""
        return self.order_counts(order).rows

    def event_count(self, order: int) -> int:
        """Return the number of prediction events counted for one order."""
        return self.order_counts(order).event_count

    def token_counts(self, order: int = 1) -> Counter[int]:
        """Return token counts for an order, normally unigram c(w)."""
        return self.order_counts(order).token_counts


def collect_ngram_counts(
    tok_seqs: Iterable[Sequence[int]],
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

    norm_orders = normalize_orders(orders, prediction_order=prediction_order)
    rows_by_order: dict[int, defaultdict[Context, Counter[int]]] = {
        order: defaultdict(Counter)
        for order in norm_orders
    }
    event_counts = {order: 0 for order in norm_orders}
    seq_count = 0  # seq = input token sequence.
    tok_count = 0  # tok = token.

    for tok_ids in tok_seqs:
        seq_count += 1
        tok_count += len(tok_ids)
        for next_idx in prediction_indices(tok_ids, order=prediction_order):
            next_id = tok_ids[next_idx]  # w, the predicted token.
            for order in norm_orders:
                context = context_at(tok_ids, next_idx, order=order)
                rows_by_order[order][context][next_id] += 1
                event_counts[order] += 1

    return NgramCorpusCounts(
        sequence_count=seq_count,
        token_count=tok_count,
        orders={
            order: NgramOrderCounts(
                order=order,
                rows=dict(rows_by_order[order]),
                event_count=event_counts[order],
            )
            for order in norm_orders
        },
    )


def normalize_orders(orders: Iterable[int], *, prediction_order: int) -> tuple[int, ...]:
    """Validate requested count orders and return them sorted and unique."""
    if prediction_order < 1:
        raise ValueError("prediction_order must be positive")

    norm_orders = tuple(sorted(set(orders)))
    if not norm_orders:
        raise ValueError("At least one n-gram order must be requested")

    bad_orders = [
        order
        for order in norm_orders
        if order < 1 or order > prediction_order
    ]
    if bad_orders:
        order_list = ", ".join(str(order) for order in bad_orders)
        raise ValueError(
            f"N-gram orders must be in [1, {prediction_order}]: {order_list}"
        )
    return norm_orders


def iter_prediction_events(
    tok_ids: Sequence[int],
    *,
    order: int,
) -> Iterator[tuple[Context, int]]:
    """Yield each n-gram history h with its predicted next token w."""
    for next_idx in prediction_indices(tok_ids, order=order):
        yield context_at(tok_ids, next_idx, order=order), tok_ids[next_idx]


def prediction_indices(tok_ids: Sequence[int], *, order: int) -> range:
    """Return sequence positions that have enough history for prediction."""
    if order < 1:
        raise ValueError("order must be positive")
    return range(order - 1, len(tok_ids))


def context_at(tok_ids: Sequence[int], next_idx: int, *, order: int) -> Context:
    """Return the n-gram history h immediately before one token index."""
    if order < 1:
        raise ValueError("order must be positive")

    ctx_start = next_idx - order + 1  # ctx = n-gram history context.
    if ctx_start < 0:
        raise ValueError("Not enough previous tokens for requested n-gram order")
    return tuple(tok_ids[ctx_start:next_idx])


def single_token_context_rows(
    rows: Mapping[Context, Counter[int]],
) -> dict[int, Counter[int]]:
    """Flatten one-token context tuples into integer-keyed transition rows."""
    return {
        single_token_context_id(context): Counter(next_counts)
        for context, next_counts in rows.items()
    }


def single_token_context_id(context: Context) -> int:
    """Extract the token ID from a one-token history context."""
    if len(context) != 1:
        raise ValueError(f"Expected a 1-token context, got {len(context)}")
    return context[0]
