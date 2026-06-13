"""Build autoregressive context-target pairs from token sequences."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


type Context = tuple[int, ...]  # h, the recent history before target token w.


def iter_context_targets(
    tok_ids: Sequence[int],
    *,
    order: int,
) -> Iterator[tuple[Context, int]]:
    """Yield each supervised language-model example as context h and target w."""
    for target_idx in target_indices(tok_ids, order=order):
        yield context_at(tok_ids, target_idx, order=order), tok_ids[target_idx]


def target_indices(tok_ids: Sequence[int], *, order: int) -> range:
    """Return target positions that have enough preceding context."""
    if order < 1:
        raise ValueError("order must be positive")
    return range(order - 1, len(tok_ids))


def context_at(tok_ids: Sequence[int], target_idx: int, *, order: int) -> Context:
    """Return the n-gram context h immediately before one target index."""
    if order < 1:
        raise ValueError("order must be positive")

    ctx_start = target_idx - order + 1  # ctx = n-gram history context.
    if ctx_start < 0:
        raise ValueError("Not enough previous tokens for requested n-gram order")
    return tuple(tok_ids[ctx_start:target_idx])
