"""Very small token-level autoregressive bigram model training and querying."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from src.corpora import normalization
from src.models.core import ngram
from src.tokenizers import core as tok_core


_SCHEMA_TYPE = "autoregressive_bigram"


class BigramTrainingSummary(ngram.NgramTrainingSummary):
    transition_count: int = 0


class BigramEvaluationRow(ngram.FrozenNgramModel):
    counts: dict[int, int]
    denominator: float
    greedy_token_id: int
    top_k_token_ids: frozenset[int]


class BigramModel(ngram.BaseNgramModel):
    smoothing: float
    transitions: dict[int, tuple[tuple[int, int], ...]]

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
        observed = dict(self.transitions.get(previous_id, ()))
        observed_total = sum(
            observed.get(token_id, 0)
            for token_id in self.candidate_ids
        )
        candidate_count = self.candidate_count
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
        row_cache: dict[int, BigramEvaluationRow] = {}

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
            summary.sequence_count += 1
            summary.token_count += len(token_ids)

            for previous_id, next_id in zip(token_ids, token_ids[1:]):
                summary.transition_count += 1
                row = row_cache.get(previous_id)
                if row is None:
                    row = self.evaluation_row(
                        previous_id,
                        top_k=top_k,
                    )
                    row_cache[previous_id] = row

                ngram.score_evaluation_transition(
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
    ) -> BigramEvaluationRow:
        counts = {
            token_id: count
            for token_id, count in self.transitions.get(previous_id, ())
            if token_id in self.candidate_id_set
        }
        denominator = sum(counts.values()) + self.smoothing * len(self.candidate_ids)
        ranked_token_ids = self.ranked_token_ids(
            counts=counts,
        )
        return BigramEvaluationRow(
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
            return sorted(
                self.candidate_ids,
                key=lambda token_id: (-(counts.get(token_id, 0) + self.smoothing), token_id),
            )
        return sorted(counts, key=lambda token_id: (-counts[token_id], token_id))

    def transition_probability(
        self,
        next_id: int,
        *,
        row: BigramEvaluationRow,
    ) -> float:
        if row.denominator <= 0 or next_id not in self.candidate_id_set:
            return 0.0
        return (row.counts.get(next_id, 0) + self.smoothing) / row.denominator


def load_bigram_model(model_path: Path) -> BigramModel:
    data = ngram.load_json_model_payload(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return BigramModel(
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


def train_bigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    smoothing: float = 0.1,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> BigramTrainingSummary:
    tokenizer = tok_core.load_tokenizer(tokenizer_model)
    summary = BigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=tokenizer.vocab_size,
        text_normalization=text_normalization,
    )
    transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)

    for token_ids in iter_token_sequences(
        texts,
        tokenizer,
        text_normalization=text_normalization,
    ):
        summary.sequence_count += 1
        summary.token_count += len(token_ids)

        for previous_id, next_id in zip(token_ids, token_ids[1:]):
            transitions[previous_id][next_id] += 1
            summary.transition_count += 1

    model = {
        "schema_version": 1,
        "model_type": _SCHEMA_TYPE,
        **ngram.tokenizer_model_payload(
            tokenizer,
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            text_normalization=text_normalization,
        ),
        "smoothing": smoothing,
        "sequence_count": summary.sequence_count,
        "token_count": summary.token_count,
        "transition_count": summary.transition_count,
        "transitions": ngram.token_transition_payload(transitions),
    }
    ngram.write_json_model_payload(output_path, model)

    return summary


def format_summary(summary: BigramTrainingSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_training_summary_items(
            summary=summary,
            artifact_label="Bigram model file",
        ),
        ("Transitions", f"{summary.transition_count:,}"),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train_bigram_model,
    summary_items=format_summary,
    load_model=load_bigram_model,
    training_option_names=("smoothing",),
)
