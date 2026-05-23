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


class TrainingArtifacts(ngram.FrozenNgramModel):
    tokenizer: tok_core.TokenizerCodec
    counts: BigramCounts


class EvaluationRow(ngram.FrozenNgramModel):
    counts: dict[int, int]  # c(h, w), counts for one previous-token history h.
    denominator: float  # c(h) + k |V|, the add-k normalizer.
    greedy_token_id: int
    top_k_token_ids: frozenset[int]


class Model(ngram.BaseNgramModel):
    smoothing: float  # k, the additive smoothing pseudo-count.
    transitions: dict[int, tuple[tuple[int, int], ...]]  # h -> c(h, w).

    def context_for_tokens(self, token_ids: list[int]) -> int:
        return token_ids[-1] if token_ids else self.bos_id

    def advance_context(self, context: int, next_id: int) -> int:
        return next_id

    def next_token_predictions(
        self,
        previous_id: int,
        *,
        top_k: int,
    ) -> list[ngram.NgramPrediction]:
        # previous_id is h. observed[token_id] is c(h, w) for w = token_id.
        observed = dict(self.transitions.get(previous_id, ()))
        observed_total = sum(
            observed.get(token_id, 0)
            for token_id in self.candidate_ids
        )
        candidate_count = self.candidate_count  # |V|, excluding BOS.
        denominator = observed_total + self.smoothing * candidate_count

        if denominator <= 0:
            return []

        return ngram.sorted_predictions(
            (
                ngram.NgramPrediction(
                    token_id=token_id,
                    piece=self.pieces[token_id],
                    count=observed.get(token_id, 0),
                    probability=ngram.additive_smoothed_probability(
                        token_id,
                        counts=observed,
                        total=observed_total,
                        smoothing=self.smoothing,
                        candidate_count=candidate_count,
                    ),
                )
                for token_id in self.candidate_ids
                if observed.get(token_id, 0) > 0 or self.smoothing > 0
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

        resolved_text_normalization = text_normalization or self.text_normalization
        summary = ngram.NgramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            top_k=top_k,
            text_normalization=resolved_text_normalization,
        )
        for token_ids in iter_token_sequences(
            texts,
            self.tokenizer,
            text_normalization=resolved_text_normalization,
        ):
            counting.observe_sequence(summary, token_ids)

            for context, next_id in counting.iter_prediction_events(token_ids, order=2):
                previous_id = counting.single_token_context_id(context)
                row = row_cache.get(previous_id)
                if row is None:
                    row = self.evaluation_row(
                        previous_id,
                        top_k=top_k,
                    )
                    row_cache[previous_id] = row

                counting.score_evaluation_event(
                    summary,
                    actual_token_id=next_id,
                    greedy_token_id=row.greedy_token_id,
                    top_k_token_ids=row.top_k_token_ids,
                    probability=self.transition_probability(
                        next_id,
                        row=row,
                    ),
                )

        return summary

    def evaluation_row(
        self,
        previous_id: int,
        *,
        top_k: int,
    ) -> EvaluationRow:
        # Keep only candidate next-token types w in this history row h.
        counts = {
            token_id: count
            for token_id, count in self.transitions.get(previous_id, ())
            if token_id in self.candidate_id_set
        }
        denominator = sum(counts.values()) + self.smoothing * len(self.candidate_ids)
        ranked_token_ids = self.ranked_token_ids(
            counts=counts,
        )
        return EvaluationRow(
            counts=counts,
            denominator=denominator,
            greedy_token_id=ngram.greedy_token_id(ranked_token_ids, eos_id=self.eos_id),
            top_k_token_ids=ngram.top_k_token_id_set(ranked_token_ids, top_k=top_k),
        )

    def ranked_token_ids(
        self,
        *,
        counts: dict[int, int],
    ) -> list[int]:
        if self.smoothing > 0:
            # Ranking by c(h, w) + k is equivalent to ranking by P(w | h).
            return sorted(
                self.candidate_ids,
                key=lambda token_id: (-(counts.get(token_id, 0) + self.smoothing), token_id),
            )
        return sorted(counts, key=lambda token_id: (-counts[token_id], token_id))

    def transition_probability(
        self,
        next_id: int,
        *,
        row: EvaluationRow,
    ) -> float:
        if row.denominator <= 0 or next_id not in self.candidate_id_set:
            return 0.0
        # next_id is w. Return (c(h, w) + k) / (c(h) + k |V|).
        return (row.counts.get(next_id, 0) + self.smoothing) / row.denominator


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
    artifacts = collect_training_artifacts(
        texts,
        tokenizer=tokenizer,
        text_normalization=text_normalization,
    )
    summary = TrainingSummary(
        vocab_size=artifacts.tokenizer.vocab_size,
        text_normalization=text_normalization,
    )
    apply_bigram_counts_to_summary(summary, artifacts.counts)

    return ngram.TrainingResult[TrainingSummary](
        summary=summary,
        payload={
            "smoothing": smoothing,
            **bigram_counts_payload(artifacts.counts),
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


def collect_training_artifacts(
    texts: Iterable[str],
    *,
    tokenizer: tok_core.TokenizerCodec,
    text_normalization: normalization.TextNormalization = (
        normalization.DEFAULT_TEXT_NORMALIZATION
    ),
) -> TrainingArtifacts:
    counts = collect_bigram_counts(
        texts,
        tokenizer,
        text_normalization=text_normalization,
    )
    return TrainingArtifacts(tokenizer=tokenizer, counts=counts)


def bigram_counts_payload(counts: BigramCounts) -> dict[str, object]:
    return {
        "sequence_count": counts.sequence_count,
        "token_count": counts.token_count,
        "transition_count": counts.transition_count,
        "transitions": ngram.token_transition_payload(counts.transitions),
    }


def apply_bigram_counts_to_summary(
    summary: TrainingSummary,
    counts: BigramCounts,
) -> None:
    counting.apply_sequence_counts(summary, counts)
    summary.transition_count = counts.transition_count


def format_summary(summary: TrainingSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_training_summary_items(
            summary=summary,
            artifact_label="Bigram model file",
        ),
        ("Transitions", f"{summary.transition_count:,}"),
    ]
