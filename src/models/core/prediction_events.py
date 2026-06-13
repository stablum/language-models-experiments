"""Build autoregressive next-token prediction events from token sequences."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


type Context = tuple[int, ...]  # h, the recent history before predicted token w.


def iter_prediction_events(
    tok_ids: Sequence[int],
    *,
    order: int,
) -> Iterator[tuple[Context, int]]:
    """Yield each language-model event as history h and next token w."""
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
