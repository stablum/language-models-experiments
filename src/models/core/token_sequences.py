"""Token-space corpus containers for fitting and scoring language models."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, Self

import pydantic


type Context = tuple[int, ...]  # h, the recent token history before target w.
type OrderedContextTarget = tuple[int, Context, int]  # n, h, w.


class TokenSeqObserver(Protocol):
    """Accept sequence-level events during one-pass corpus traversal."""

    def observe_sequence(self, tok_seq: Sequence[int]) -> None:
        """Record that one token sequence has entered a traversal."""
        ...


class TokenCorpusStats(pydantic.BaseModel):
    """Accumulate sequence and token totals while streaming a token corpus.

    Example: count collection observes each sequence while yielding n-gram rows.
    """

    model_config = pydantic.ConfigDict(validate_assignment=True)

    sequence_count: int = 0
    token_count: int = 0

    def observe_sequence(self, tok_seq: Sequence[int]) -> None:
        """Add one sequence and its length to corpus-level totals."""
        self.sequence_count += 1
        self.token_count += len(tok_seq)


class TokenSeq(tuple[int, ...]):
    """Represent one token-ID sequence with local n-gram event helpers.

    Example: a bigram fit reads ``TokenSeq([0, 12, 1])`` as BOS, token, EOS.
    """

    def __new__(cls, ids: Iterable[int] = ()) -> Self:
        """Store token IDs immutably so repeated local indexing is stable."""
        return tuple.__new__(cls, ids)

    def iter_context_targets(
        self,
        *,
        order: int,
    ) -> Iterator[tuple[Context, int]]:
        """Yield each supervised LM example as context h and target w."""
        for target_idx in self.target_indices(order=order):
            yield self.context_at(target_idx, order=order), self[target_idx]

    def target_indices(self, *, order: int) -> range:
        """Return target positions that have enough preceding context."""
        if order < 1:
            raise ValueError("order must be positive")
        return range(order - 1, len(self))

    def context_at(self, target_idx: int, *, order: int) -> Context:
        """Return the n-gram context h immediately before one target index."""
        if order < 1:
            raise ValueError("order must be positive")

        ctx_start = target_idx - order + 1  # ctx = n-gram history context.
        if ctx_start < 0:
            raise ValueError("Not enough previous tokens for requested n-gram order")
        return tuple(self[ctx_start:target_idx])


class TokenCorpus:
    """Stream token-ID sequences together with the vocabulary size |V|.

    Example: runtime passes this object to module-level ``fit(...)``.
    """

    def __init__(
        self,
        seqs: Iterable[Sequence[int] | TokenSeq],
        *,
        vocab_size: int,
    ) -> None:
        """Bind a token stream to its finite token-space dimension."""
        if vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")
        self._seqs = seqs
        self.vocab_size = vocab_size

    def __iter__(self) -> Iterator[TokenSeq]:
        """Yield tuple-backed token sequences from the underlying stream."""
        for seq in self._seqs:
            yield seq if isinstance(seq, TokenSeq) else TokenSeq(seq)

    def iter_context_targets(
        self,
        *,
        order: int,
        seq_observer: TokenSeqObserver | None = None,
    ) -> Iterator[tuple[Context, int]]:
        """Yield all context-target pairs, observing each sequence if requested."""
        for seq in self:
            if seq_observer is not None:
                seq_observer.observe_sequence(seq)
            yield from seq.iter_context_targets(order=order)

    def iter_aligned_context_targets(
        self,
        *,
        orders: Iterable[int],
        target_order: int,
        seq_observer: TokenSeqObserver | None = None,
    ) -> Iterator[OrderedContextTarget]:
        """Yield (n, h, w) events for several orders on one target frontier."""
        norm_orders = tuple(orders)
        for seq in self:
            if seq_observer is not None:
                seq_observer.observe_sequence(seq)

            for target_idx in seq.target_indices(order=target_order):
                target_id = seq[target_idx]  # w, the observed next token.
                for order in norm_orders:
                    yield order, seq.context_at(target_idx, order=order), target_id


def single_token_context_id(context: Context) -> int:
    """Extract the token ID from a one-token history context."""
    if len(context) != 1:
        raise ValueError(f"Expected a 1-token context, got {len(context)}")
    return context[0]
