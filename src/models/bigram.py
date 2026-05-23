"""Very small token-level autoregressive bigram model training and querying.

For history ``h`` and next token ``w``, this model uses the add-k estimator
``P(w | h) = (c(h, w) + k) / (c(h) + k |V|)``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from src.corpora import normalization
from src.models.core import counting, ngram
from src.tokenizers import core as tok_core


class TrainingSummary(ngram.NgramTrainingSummary):
    transition_count: int = 0  # sum_h c(h), the number of bigram events.


class BigramCounts(counting.NgramCorpusCounts):
    @property
    def transitions(self) -> dict[int, Counter[int]]:
        return counting.single_token_context_rows(self.rows(2))

    @property
    def transition_count(self) -> int:
        return self.event_count(2)


class EvaluationRow(ngram.FrozenNgramModel):
    counts: dict[int, int]  # c(h, w), counts for one previous-token history h.
    denom: float  # denom = c(h) + k |V|, the add-k normalizer.
    greedy_id: int
    top_k_ids: frozenset[int]


class Model(ngram.BaseNgramModel):
    smoothing: float  # k, the additive smoothing pseudo-count.
    transitions: dict[int, tuple[tuple[int, int], ...]]  # h -> c(h, w).

    def context_for_tokens(self, token_ids: list[int]) -> int:
        return token_ids[-1] if token_ids else self.bos_id

    def advance_context(self, context: int, next_id: int) -> int:
        return next_id

    def next_token_predictions(
        self,
        prev_id: int,
        *,
        top_k: int,
    ) -> list[ngram.NgramPrediction]:
        # prev_id is h. obs[token_id] is c(h, w) for w = token_id.
        obs = dict(self.transitions.get(prev_id, ()))
        obs_tot = sum(
            obs.get(token_id, 0)
            for token_id in self.cand_ids
        )
        cand_count = self.cand_count  # cand = |V|, excluding BOS.
        denom = obs_tot + self.smoothing * cand_count  # c(h) + k |V|.

        if denom <= 0:
            return []

        return ngram.sorted_predictions(
            (
                ngram.NgramPrediction(
                    token_id=token_id,
                    piece=self.pieces[token_id],
                    count=obs.get(token_id, 0),
                    prob=ngram.add_k_prob(
                        token_id,
                        counts=obs,
                        tot=obs_tot,
                        smoothing=self.smoothing,
                        cand_count=cand_count,
                    ),
                )
                for token_id in self.cand_ids
                if obs.get(token_id, 0) > 0 or self.smoothing > 0
            ),
            top_k=top_k,
        )

    def evaluate(
        self,
        texts: Iterable[str],
        *,
        top_k: int = 5,
        text_normalization: normalization.TextNormalization | None = None,
    ) -> ngram.NgramEvaluationSummary:
        row_cache: dict[int, EvaluationRow] = {}

        text_norm = text_normalization or self.text_normalization
        summary = ngram.NgramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            top_k=top_k,
            text_normalization=text_norm,
        )
        for tok_ids in iter_token_sequences(
            texts,
            self.tokenizer,
            text_normalization=text_norm,
        ):
            counting.observe_sequence(summary, tok_ids)

            for context, next_id in counting.iter_prediction_events(tok_ids, order=2):
                prev_id = counting.single_token_context_id(context)
                row = row_cache.get(prev_id)
                if row is None:
                    row = self.evaluation_row(
                        prev_id,
                        top_k=top_k,
                    )
                    row_cache[prev_id] = row

                counting.score_evaluation_event(
                    summary,
                    actual_id=next_id,
                    greedy_id=row.greedy_id,
                    top_k_ids=row.top_k_ids,
                    prob=self.transition_prob(
                        next_id,
                        row=row,
                    ),
                )

        return summary

    def evaluation_row(
        self,
        prev_id: int,
        *,
        top_k: int,
    ) -> EvaluationRow:
        # Keep only candidate next-token types w in this history row h.
        counts = {
            token_id: count
            for token_id, count in self.transitions.get(prev_id, ())
            if token_id in self.cand_id_set
        }
        denom = sum(counts.values()) + self.smoothing * len(self.cand_ids)
        ranked_ids = self.ranked_ids(
            counts=counts,
        )
        return EvaluationRow(
            counts=counts,
            denom=denom,
            greedy_id=ngram.greedy_id(ranked_ids, eos_id=self.eos_id),
            top_k_ids=ngram.top_k_id_set(ranked_ids, top_k=top_k),
        )

    def ranked_ids(
        self,
        *,
        counts: dict[int, int],
    ) -> list[int]:
        if self.smoothing > 0:
            # Ranking by c(h, w) + k is equivalent to ranking by P(w | h).
            return sorted(
                self.cand_ids,
                key=lambda token_id: (-(counts.get(token_id, 0) + self.smoothing), token_id),
            )
        return sorted(counts, key=lambda token_id: (-counts[token_id], token_id))

    def transition_prob(
        self,
        next_id: int,
        *,
        row: EvaluationRow,
    ) -> float:
        if row.denom <= 0 or next_id not in self.cand_id_set:
            return 0.0
        # next_id is w. Return (c(h, w) + k) / denom.
        return (row.counts.get(next_id, 0) + self.smoothing) / row.denom


def load(model_path: Path) -> Model:
    data = ngram.load_json_model_payload(
        model_path,
        module_name=__name__,
    )

    return Model(
        **ngram.load_tokenizer_model_fields(data, model_path),
        smoothing=float(data["smoothing"]),
        transitions=ngram.parse_token_transitions(data, "transitions"),
    )


def iter_token_sequences(
    texts: Iterable[str],
    tokenizer: tok_core.TokenizerCodec,
    *,
    text_normalization: normalization.TextNormalization = "none",
) -> Iterator[list[int]]:
    yield from ngram.iter_token_sequences(
        texts,
        tokenizer,
        bos_count=1,
        min_length=2,
        text_normalization=text_normalization,
    )


def train(
    texts: Iterable[str],
    *,
    tokenizer: tok_core.TokenizerCodec,
    smoothing: float = 0.1,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> ngram.TrainingResult[TrainingSummary]:
    counts = collect_bigram_counts(
        texts,
        tokenizer=tokenizer,
        text_normalization=text_normalization,
    )
    summary = TrainingSummary(
        vocab_size=tokenizer.vocab_size,
        sequence_count=counts.sequence_count,
        token_count=counts.token_count,
        transition_count=counts.transition_count,
        text_normalization=text_normalization,
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
    texts: Iterable[str],
    tokenizer: tok_core.TokenizerCodec,
    *,
    text_normalization: normalization.TextNormalization = (
        normalization.DEFAULT_TEXT_NORMALIZATION
    ),
) -> BigramCounts:
    counts = counting.collect_ngram_counts(
        iter_token_sequences(
            texts,
            tokenizer,
            text_normalization=text_normalization,
        ),
        orders=(2,),
        prediction_order=2,
    )
    return BigramCounts(
        sequence_count=counts.sequence_count,
        token_count=counts.token_count,
        orders=counts.orders,
    )


def format_summary(summary: TrainingSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_training_summary_items(
            summary=summary,
            artifact_label="Bigram model file",
        ),
        ("Transitions", f"{summary.transition_count:,}"),
    ]
