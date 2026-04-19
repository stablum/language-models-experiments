"""Shared pieces for token-level trigram models."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TextNormalization
from src.models.ngram import (
    DecodingMode,
    NgramEvaluationSummary,
    NgramPrediction,
    NgramQueryResult,
    candidate_token_ids,
    decode_continuation,
    encode_prompt,
    iter_sentencepiece_token_sequences,
    seeded_rng,
    select_next_token,
)


Context = tuple[int, int]
TrigramPrediction = NgramPrediction
TrigramQueryResult = NgramQueryResult


@dataclass(frozen=True)
class TrigramCounts:
    sequence_count: int
    token_count: int
    unigram_counts: Counter[int]
    bigram_transitions: defaultdict[int, Counter[int]]
    trigram_transitions: defaultdict[Context, Counter[int]]
    bigram_transition_count: int
    trigram_transition_count: int

    @property
    def unigram_count(self) -> int:
        return sum(self.unigram_counts.values())


@dataclass(frozen=True)
class TrigramEvaluationRow:
    bigram_counts: dict[int, int]
    trigram_counts: dict[int, int]
    bigram_total: int
    trigram_total: int
    greedy_token_id: int
    top_k_token_ids: frozenset[int]


class BaseTrigramModel:
    model_path: Path
    tokenizer_model: Path
    processor: spm.SentencePieceProcessor
    vocab_size: int
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    bigram_transitions: dict[int, tuple[tuple[int, int], ...]]
    trigram_transitions: dict[Context, tuple[tuple[int, int], ...]]
    text_normalization: str

    def encode_prompt(self, prompt: str) -> list[int]:
        return encode_prompt(
            self.processor,
            prompt,
            text_normalization=self.text_normalization,
        )

    def next_token_predictions(
        self,
        context: Context,
        *,
        top_k: int,
    ) -> list[TrigramPrediction]:
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        predictions = [
            TrigramPrediction(
                token_id=token_id,
                piece=self.pieces[token_id],
                count=trigram_counts.get(token_id, 0),
                probability=self.transition_probability(token_id, context),
            )
            for token_id in candidate_token_ids(self.vocab_size, self.bos_id)
        ]
        predictions.sort(key=lambda prediction: (-prediction.probability, prediction.token_id))
        return predictions[:top_k] if top_k > 0 else predictions

    def query(
        self,
        *,
        prompt: str = "",
        max_tokens: int = 80,
        top_k: int = 10,
        decoding: DecodingMode = "sample",
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> TrigramQueryResult:
        prompt_token_ids = self.encode_prompt(prompt)
        context = self.context_for_tokens(prompt_token_ids)
        next_token_predictions = self.next_token_predictions(context, top_k=top_k)
        rng = seeded_rng(seed)
        token_ids = list(prompt_token_ids)
        generated_token_ids: list[int] = []

        for _ in range(max_tokens):
            next_id = select_next_token(
                self.next_token_predictions(context, top_k=0),
                eos_id=self.eos_id,
                decoding=decoding,
                rng=rng,
                temperature=temperature,
            )
            if next_id == self.eos_id:
                break

            generated_token_ids.append(next_id)
            token_ids.append(next_id)
            context = (context[1], next_id)

        prompt_text = self.processor.decode(prompt_token_ids)
        generated_text = self.processor.decode(token_ids)
        continuation_text = decode_continuation(
            self.processor,
            generated_text=generated_text,
            prompt_text=prompt_text,
            generated_token_ids=generated_token_ids,
        )

        return TrigramQueryResult(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            decoding=decoding,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            unk_id=self.unk_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            continuation_text=continuation_text,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            token_ids=token_ids,
            next_token_predictions=next_token_predictions,
            text_normalization=self.text_normalization,
        )

    def evaluate(
        self,
        texts: Iterable[str],
        *,
        top_k: int = 5,
        text_normalization: TextNormalization | None = None,
    ) -> NgramEvaluationSummary:
        row_cache: dict[Context, TrigramEvaluationRow] = {}
        sequence_count = 0
        token_count = 0
        transition_count = 0
        correct_next_token_count = 0
        top_k_correct_next_token_count = 0
        negative_log_likelihood = 0.0
        zero_probability_count = 0

        resolved_text_normalization = text_normalization or self.text_normalization
        for token_ids in iter_trigram_token_sequences(
            texts,
            self.processor,
            text_normalization=resolved_text_normalization,
        ):
            sequence_count += 1
            token_count += len(token_ids)

            for previous_previous_id, previous_id, next_id in zip(
                token_ids,
                token_ids[1:],
                token_ids[2:],
            ):
                transition_count += 1
                context = (previous_previous_id, previous_id)
                row = row_cache.get(context)
                if row is None:
                    row = self.evaluation_row(context, top_k=top_k)
                    row_cache[context] = row

                if next_id == row.greedy_token_id:
                    correct_next_token_count += 1
                if next_id in row.top_k_token_ids:
                    top_k_correct_next_token_count += 1

                probability = self.transition_probability(next_id, context, row=row)
                if probability <= 0:
                    zero_probability_count += 1
                else:
                    negative_log_likelihood -= math.log(probability)

        return self.evaluation_summary(
            top_k=top_k,
            sequence_count=sequence_count,
            token_count=token_count,
            transition_count=transition_count,
            correct_next_token_count=correct_next_token_count,
            top_k_correct_next_token_count=top_k_correct_next_token_count,
            negative_log_likelihood=negative_log_likelihood,
            zero_probability_count=zero_probability_count,
            text_normalization=resolved_text_normalization,
        )

    def evaluation_summary(self, **kwargs: Any) -> NgramEvaluationSummary:
        return NgramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            **kwargs,
        )

    def transition_probability(
        self,
        next_id: int,
        context: Context,
        *,
        row: TrigramEvaluationRow | None = None,
        bigram_counts: dict[int, int] | None = None,
        trigram_counts: dict[int, int] | None = None,
        bigram_total: int | None = None,
        trigram_total: int | None = None,
    ) -> float:
        raise NotImplementedError

    def evaluation_row(
        self,
        context: Context,
        *,
        top_k: int,
    ) -> TrigramEvaluationRow:
        previous_id = context[1]
        bigram_counts = dict(self.bigram_transitions.get(previous_id, ()))
        trigram_counts = dict(self.trigram_transitions.get(context, ()))
        bigram_total = sum(bigram_counts.values())
        trigram_total = sum(trigram_counts.values())
        ranked_token_ids = self.ranked_token_ids(
            context,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )
        fallback_token_id = self.eos_id if self.eos_id >= 0 else 0
        greedy_token_id = ranked_token_ids[0] if ranked_token_ids else fallback_token_id
        return TrigramEvaluationRow(
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
            greedy_token_id=greedy_token_id,
            top_k_token_ids=frozenset(ranked_token_ids[:top_k]) if top_k > 0 else frozenset(),
        )

    def ranked_token_ids(
        self,
        context: Context,
        *,
        bigram_counts: dict[int, int],
        trigram_counts: dict[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> list[int]:
        return sorted(
            candidate_token_ids(self.vocab_size, self.bos_id),
            key=lambda token_id: (
                -self.transition_probability(
                    token_id,
                    context,
                    bigram_counts=bigram_counts,
                    trigram_counts=trigram_counts,
                    bigram_total=bigram_total,
                    trigram_total=trigram_total,
                ),
                token_id,
            ),
        )

    def context_for_tokens(self, token_ids: list[int]) -> Context:
        bos_id = self.bos_id if self.bos_id >= 0 else 0
        if len(token_ids) >= 2:
            return token_ids[-2], token_ids[-1]
        if len(token_ids) == 1:
            return bos_id, token_ids[-1]
        return bos_id, bos_id


def collect_trigram_counts(
    texts: Iterable[str],
    processor: spm.SentencePieceProcessor,
    *,
    text_normalization: TextNormalization = DEFAULT_TEXT_NORMALIZATION,
) -> TrigramCounts:
    unigram_counts: Counter[int] = Counter()
    bigram_transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    trigram_transitions: defaultdict[Context, Counter[int]] = defaultdict(Counter)
    sequence_count = 0
    token_count = 0
    bigram_transition_count = 0
    trigram_transition_count = 0

    for token_ids in iter_trigram_token_sequences(
        texts,
        processor,
        text_normalization=text_normalization,
    ):
        sequence_count += 1
        token_count += len(token_ids)
        unigram_counts.update(token_ids[2:])

        for previous_id, next_id in zip(token_ids[1:], token_ids[2:]):
            bigram_transitions[previous_id][next_id] += 1
            bigram_transition_count += 1

        for previous_previous_id, previous_id, next_id in zip(
            token_ids,
            token_ids[1:],
            token_ids[2:],
        ):
            trigram_transitions[(previous_previous_id, previous_id)][next_id] += 1
            trigram_transition_count += 1

    return TrigramCounts(
        sequence_count=sequence_count,
        token_count=token_count,
        unigram_counts=unigram_counts,
        bigram_transitions=bigram_transitions,
        trigram_transitions=trigram_transitions,
        bigram_transition_count=bigram_transition_count,
        trigram_transition_count=trigram_transition_count,
    )


def trigram_counts_payload(counts: TrigramCounts) -> dict[str, object]:
    return {
        "sequence_count": counts.sequence_count,
        "token_count": counts.token_count,
        "unigram_count": counts.unigram_count,
        "bigram_transition_count": counts.bigram_transition_count,
        "trigram_transition_count": counts.trigram_transition_count,
        "unigrams": token_counts_payload(counts.unigram_counts),
        "bigram_transitions": token_transition_payload(counts.bigram_transitions),
        "trigram_transitions": context_transition_payload(counts.trigram_transitions),
    }


def parse_unigram_counts(data: dict[str, object]) -> dict[int, int]:
    return parse_token_counts(data, "unigrams")


def parse_bigram_transitions(data: dict[str, object]) -> dict[int, tuple[tuple[int, int], ...]]:
    return parse_token_transitions(data, "bigram_transitions")


def parse_trigram_transitions(
    data: dict[str, object],
) -> dict[Context, tuple[tuple[int, int], ...]]:
    return parse_context_transitions(data, "trigram_transitions")


def token_counts_payload(counts: Counter[int] | dict[int, int]) -> list[tuple[int, int]]:
    return sorted(counts.items())


def token_transition_payload(
    transitions: defaultdict[int, Counter[int]] | dict[int, Counter[int]],
) -> dict[str, list[tuple[int, int]]]:
    return {
        str(previous_id): sorted(next_counts.items())
        for previous_id, next_counts in sorted(transitions.items())
    }


def context_transition_payload(
    transitions: defaultdict[Context, Counter[int]] | dict[Context, Counter[int]],
) -> dict[str, list[tuple[int, int]]]:
    return {
        context_key(previous_previous_id, previous_id): sorted(next_counts.items())
        for (
            previous_previous_id,
            previous_id,
        ), next_counts in sorted(transitions.items())
    }


def parse_token_counts(data: dict[str, object], key: str) -> dict[int, int]:
    return {
        int(token_id): int(count)
        for token_id, count in data[key]
    }


def parse_token_transitions(
    data: dict[str, object],
    key: str,
) -> dict[int, tuple[tuple[int, int], ...]]:
    return {
        int(previous_id): tuple(
            (int(next_id), int(count))
            for next_id, count in next_counts
        )
        for previous_id, next_counts in data[key].items()
    }


def parse_context_transitions(
    data: dict[str, object],
    key: str,
) -> dict[Context, tuple[tuple[int, int], ...]]:
    return {
        parse_context_key(context_key): tuple(
            (int(next_id), int(count))
            for next_id, count in next_counts
        )
        for context_key, next_counts in data[key].items()
    }


def iter_trigram_token_sequences(
    texts: Iterable[str],
    processor: spm.SentencePieceProcessor,
    *,
    text_normalization: TextNormalization = "none",
) -> Iterator[list[int]]:
    yield from iter_sentencepiece_token_sequences(
        texts,
        processor,
        bos_count=2,
        min_length=3,
        text_normalization=text_normalization,
    )


def context_key(previous_previous_id: int, previous_id: int) -> str:
    return f"{previous_previous_id},{previous_id}"


def parse_context_key(key: str) -> Context:
    previous_previous_id, previous_id = key.split(",", maxsplit=1)
    return int(previous_previous_id), int(previous_id)
