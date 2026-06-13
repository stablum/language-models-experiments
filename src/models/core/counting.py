"""Shared context-target counting for token-level n-gram models."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from src.models.core import ngram
from src.models.core import token_sequences


type TransitionRows = dict[token_sequences.Context, Counter[int]]
type MutableTransitionRows = defaultdict[token_sequences.Context, Counter[int]]


class NgramOrderCounts(ngram.FrozenNgramPydanticBase):
    """Store transition rows and event totals for one n-gram order."""

    order: int  # n in c(h, w), with len(h) = n - 1.
    rows: TransitionRows  # h -> c(h, w).
    event_count: int  # sum_h c(h), the number of target tokens.

    @property
    def token_counts(self) -> Counter[int]:
        """Return unigram counts as c(w) for order-1 rows."""
        if self.order != 1:
            raise ValueError("token_counts is only defined for unigram counts")
        return Counter(self.rows.get((), Counter()))


class NgramCorpusCounts(ngram.FrozenNgramPydanticBase):
    """Bundle aligned count tables across n-gram orders for one corpus."""

    vocab_size: int
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
        """Return the number of context-target pairs counted for one order."""
        return self.order_counts(order).event_count

    def token_counts(self, order: int = 1) -> Counter[int]:
        """Return token counts for an order, normally unigram c(w)."""
        return self.order_counts(order).token_counts


def collect_ngram_counts(
    corpus: token_sequences.TokenCorpus,
    *,
    orders: Iterable[int],
    target_order: int,
) -> NgramCorpusCounts:
    """Count requested lower-order rows on the same target frontier.

    ``target_order`` chooses which tokens become targets. For a
    trigram model this is 3, so unigram and bigram backing counts are aligned to
    tokens with two-token contexts instead of including earlier BOS
    warm-up events.
    """

    norm_orders = normalize_orders(orders, target_order=target_order)
    rows_by_order: dict[int, MutableTransitionRows] = {
        order: defaultdict(Counter)
        for order in norm_orders
    }
    event_counts = {order: 0 for order in norm_orders}
    stats = token_sequences.TokenCorpusStats()

    for order, context, target_id in corpus.iter_aligned_context_targets(
        orders=norm_orders,
        target_order=target_order,
        seq_observer=stats,
    ):
        rows_by_order[order][context][target_id] += 1
        event_counts[order] += 1

    return NgramCorpusCounts(
        vocab_size=corpus.vocab_size,
        sequence_count=stats.sequence_count,
        token_count=stats.token_count,
        orders={
            order: NgramOrderCounts(
                order=order,
                rows=dict(rows_by_order[order]),
                event_count=event_counts[order],
            )
            for order in norm_orders
        },
    )


def normalize_orders(orders: Iterable[int], *, target_order: int) -> tuple[int, ...]:
    """Validate requested count orders and return them sorted and unique."""
    if target_order < 1:
        raise ValueError("target_order must be positive")

    norm_orders = tuple(sorted(set(orders)))
    if not norm_orders:
        raise ValueError("At least one n-gram order must be requested")

    bad_orders = [
        order
        for order in norm_orders
        if order < 1 or order > target_order
    ]
    if bad_orders:
        order_list = ", ".join(str(order) for order in bad_orders)
        raise ValueError(
            f"N-gram orders must be in [1, {target_order}]: {order_list}"
        )
    return norm_orders


def single_token_context_rows(
    rows: Mapping[token_sequences.Context, Counter[int]],
) -> dict[int, Counter[int]]:
    """Flatten one-token context tuples into integer-keyed transition rows."""
    return {
        token_sequences.single_token_context_id(context): Counter(next_counts)
        for context, next_counts in rows.items()
    }
