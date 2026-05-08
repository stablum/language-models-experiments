"""Good-Turing-smoothed token-level autoregressive trigram model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import sentencepiece as spm

from src.corpora import normalization
from src.models.core import formatting, ngram, trigrams


_SCHEMA_TYPE = "good_turing_trigram"


class GoodTuringTrigramModel(trigrams.BackoffTrigramModel):
    def smooth_counts(
        self,
        counts: Mapping[int, int],
        *,
        lower_probability: ngram.ProbabilityFn,
        total: int | None = None,
    ) -> ngram.BackoffDistribution:
        return ngram.good_turing_smoothed_distribution(
            counts,
            candidate_ids=self.candidate_ids,
            lower_probability=lower_probability,
            total=total,
        )


def load_good_turing_trigram_model(model_path: Path) -> GoodTuringTrigramModel:
    data, tokenizer_model, processor, vocab_size = trigrams.load_standard_trigram_payload(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return GoodTuringTrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        **ngram.sentencepiece_model_fields(data, processor, vocab_size),
        unigram_counts=trigrams.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def train_good_turing_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> trigrams.TrigramTrainingSummary:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    summary = trigrams.TrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        text_normalization=text_normalization,
    )
    counts = trigrams.collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )
    trigrams.apply_trigram_counts_to_summary(summary, counts)

    model = trigrams.standard_trigram_model_payload(
        processor,
        model_type=_SCHEMA_TYPE,
        tokenizer_model=tokenizer_model,
        stored_tokenizer_model=stored_tokenizer_model,
        vocab_size=summary.vocab_size,
        text_normalization=text_normalization,
        counts=counts,
    )
    ngram.write_json_model_payload(output_path, model)

    return summary


def format_summary(summary: trigrams.TrigramTrainingSummary) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Good-Turing trigram model artifact file",
        ),
        ("Smoothing", "Good-Turing"),
    ]


def format_evaluation(summary: ngram.NgramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        ("Smoothing", "Good-Turing"),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train_good_turing_trigram_model,
    summary_items=format_summary,
    load_model=load_good_turing_trigram_model,
    evaluation_items=format_evaluation,
)
