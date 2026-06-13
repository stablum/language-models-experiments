"""Very small token-level autoregressive bigram model training and querying.

For history ``h`` and next token ``w``, this model uses the add-k estimator
``P(w | h) = (c(h, w) + k) / (c(h) + k |V|)``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.models.core import context_targets, counting, ngram


CONTEXT_LENGTH = 1  # len(h), the previous-token history size.


class TrainingSummary(ngram.NgramTrainingSummary):
    """Store bigram training counts and add-k hyperparams for reports."""

    transition_count: int = 0  # sum_h c(h), the number of bigram events.
    smoothing: float = 0.0  # k, the additive smoothing pseudo-count.


class BigramCounts(counting.NgramCorpusCounts):
    @property
    def transitions(self) -> dict[int, Counter[int]]:
        """Return bigram transition rows keyed by the previous token h."""
        return counting.single_token_context_rows(self.rows(2))

    @property
    def transition_count(self) -> int:
        """Return the number of observed bigram context-target pairs."""
        return self.event_count(2)


class EvaluationRow(ngram.FrozenNgramPydanticBase):
    """Cache scoring data for repeated evaluation of one bigram history."""

    counts: dict[int, int]  # c(h, w), counts for one previous-token history h.
    denom: float  # denom = c(h) + k |V|, the add-k normalizer.
    greedy_id: int
    top_k_ids: frozenset[int]


class Model(ngram.BaseNgramModel):
    smoothing: float  # k, the additive smoothing pseudo-count.
    transitions: dict[int, tuple[tuple[int, int], ...]]  # h -> c(h, w).

    def context_for_tokens(self, token_ids: list[int]) -> int:
        """Choose the latest token, or BOS, as the bigram history h."""
        return token_ids[-1] if token_ids else self.bos_id

    def advance_context(self, context: int, next_id: int) -> int:
        """Advance the bigram history h to the generated next token w."""
        return next_id

    def next_token_predictions(
        self,
        prev_id: int,
        *,
        top_k: int,
    ) -> list[ngram.NgramPrediction]:
        """Return top next-token prediction records for one bigram history h."""
        counts = self._transition_counts(prev_id)
        tot = sum(counts.values())  # c(h), candidate-filtered row total.
        ranked_ids = self._ranked_ids(counts=counts)
        if top_k > 0:
            ranked_ids = ranked_ids[:top_k]

        cand_count = self.cand_count  # |V|, excluding BOS.
        return [
            ngram.NgramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=counts.get(token_id, 0),
                prob=ngram.add_k_prob(
                    token_id,
                    counts=counts,
                    tot=tot,
                    smoothing=self.smoothing,
                    cand_count=cand_count,
                ),
            )
            for token_id in ranked_ids
        ]

    def _transition_counts(self, prev_id: int) -> dict[int, int]:
        """Return candidate counts c(h, w) for one previous-token history h."""
        row = self.transitions.get(prev_id, ())
        return self.candidate_counts(row)

    def evaluate_token_ids(
        self,
        tok_seqs: Iterable[Sequence[int]],
        *,
        top_k: int = 5,
    ) -> ngram.NgramEvaluationSummary:
        """Score token ID sequences by cached bigram rows."""
        row_cache: dict[int, EvaluationRow] = {}

        summary = ngram.NgramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            top_k=top_k,
            text_normalization=self.text_normalization,
        )
        for tok_ids in tok_seqs:
            summary.observe_sequence(tok_ids)

            for context, next_id in context_targets.iter_context_targets(
                tok_ids,
                order=2,
            ):
                prev_id = counting.single_token_context_id(context)
                row = row_cache.get(prev_id)
                if row is None:
                    row = self._evaluation_row(
                        prev_id,
                        top_k=top_k,
                    )
                    row_cache[prev_id] = row

                summary.score_next_token(
                    actual_id=next_id,
                    greedy_id=row.greedy_id,
                    top_k_ids=row.top_k_ids,
                    prob=self._transition_prob(
                        next_id,
                        row=row,
                    ),
                )

        return summary

    def _evaluation_row(
        self,
        prev_id: int,
        *,
        top_k: int,
    ) -> EvaluationRow:
        """Precompute one history row for repeated evaluation events."""
        counts = self._transition_counts(prev_id)
        denom = sum(counts.values()) + self.smoothing * self.cand_count
        ranked_ids = self._ranked_ids(
            counts=counts,
        )
        return EvaluationRow(
            counts=counts,
            denom=denom,
            greedy_id=ngram.greedy_id(ranked_ids, eos_id=self.eos_id),
            top_k_ids=ngram.top_k_id_set(ranked_ids, top_k=top_k),
        )

    def _ranked_ids(
        self,
        *,
        counts: dict[int, int],
    ) -> list[int]:
        """Rank candidate next-token IDs by their bigram probability."""
        if self.smoothing > 0:
            # Ranking by c(h, w) + k is equivalent to ranking by P(w | h).
            return sorted(
                self.cand_ids,
                key=lambda token_id: (
                    -(counts.get(token_id, 0) + self.smoothing),
                    token_id,
                ),
            )
        return sorted(counts, key=lambda token_id: (-counts[token_id], token_id))

    def _transition_prob(
        self,
        next_id: int,
        *,
        row: EvaluationRow,
    ) -> float:
        """Compute P(w | h) for one next token from a cached row."""
        if row.denom <= 0 or next_id not in self.cand_id_set:
            return 0.0
        # next_id is w. Return (c(h, w) + k) / denom.
        return (row.counts.get(next_id, 0) + self.smoothing) / row.denom


def load(model_path: Path) -> Model:
    """Load a serialized bigram JSON artifact into a queryable model."""
    data = ngram.load_json_model_payload(
        model_path,
        module_name=__name__,
    )

    return Model(
        **ngram.load_token_space_model_fields(data, model_path),
        smoothing=float(data["smoothing"]),
        transitions=ngram.parse_token_transitions(data, "transitions"),
    )


def fit(
    tok_seqs: Iterable[Sequence[int]],
    *,
    token_space: ngram.TokenSpace,
    smoothing: float = 0.1,
) -> ngram.TrainingResult[TrainingSummary]:
    """Fit bigram counts from token ID sequences."""
    counts = collect_bigram_counts(tok_seqs)
    summary = TrainingSummary(
        vocab_size=token_space.vocab_size,
        sequence_count=counts.sequence_count,
        token_count=counts.token_count,
        transition_count=counts.transition_count,
        smoothing=smoothing,
    )

    return ngram.TrainingResult[TrainingSummary](
        summary=summary,
        payload={
            "smoothing": smoothing,
            "sequence_count": counts.sequence_count,
            "token_count": counts.token_count,
            "transition_count": counts.transition_count,
            "transitions": ngram.token_transition_payload(counts.transitions),
        },
    )


def collect_bigram_counts(
    tok_seqs: Iterable[Sequence[int]],
) -> BigramCounts:
    """Count c(h, w) rows for all bigram context-target pairs in the corpus."""
    counts = counting.collect_ngram_counts(
        tok_seqs,
        orders=(2,),
        target_order=2,
    )
    return BigramCounts(
        sequence_count=counts.sequence_count,
        token_count=counts.token_count,
        orders=counts.orders,
    )
