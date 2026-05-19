"""Interpolated Kneser-Ney token-level autoregressive trigram model.

The trigram row uses the usual absolute-discount shape:
``max(c(h, w) - D, 0) / c(h) + lambda(h) * P_lower(w)``. The Kneser-Ney part is
the choice of ``P_lower``: lower-order rows are trained from continuation
counts, so a token is valuable when it appears after many distinct histories,
not merely when it is frequent overall.

Notation in comments follows the usual language-model literature: ``h`` is the
history row, ``w`` is the candidate next token, ``D`` is the discount,
``T(h)`` is the number of observed continuation types, and ``N_{1+}`` is a
distinct-context count.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import ClassVar

from src.corpora import normalization
from src.models.core import ngram
from src.models.core import trigrams


_SCHEMA_TYPE = "interpolated_kneser_ney_trigram"


class KneserNeyTrigramTrainingSummary(trigrams.TrigramTrainingSummary):
    continuation_unigram_count: int = 0  # sum_w c_KN(w), unigram continuation mass.
    continuation_bigram_type_count: int = 0  # |{(v, w): N_{1+}(*, v, w) > 0}|.
    discount: float = 0.0  # D, the absolute discount.


class KneserNeyContinuationCounts(ngram.FrozenNgramModel):
    """Continuation-count tables used as Kneser-Ney lower-order evidence.

    In notation, ``bigram_transitions[v][w]`` stores
    ``c_KN(v, w) = N_{1+}(*, v, w)``, the number of distinct left contexts
    ``u`` with ``c(u, v, w) > 0``. ``unigram_counts[w]`` stores ``c_KN(w)``,
    the number of distinct previous-token types ``v`` that can precede ``w``.
    """

    unigram_counts: Counter[int]
    bigram_transitions: dict[int, Counter[int]]

    @property
    def unigram_count(self) -> int:
        return sum(self.unigram_counts.values())

    @property
    def bigram_type_count(self) -> int:
        return sum(len(next_counts) for next_counts in self.bigram_transitions.values())


class KneserNeyTrigramModel(trigrams.DiscountedTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        trigrams.DiscountedTrigramEvaluationSummary
    )
    unigram_counts: dict[int, int]  # c_KN(w), continuation unigram counts.
    unigram_total: int  # sum_w c_KN(w), the unigram row total.

    def context_probability(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        return self.trigram_probability(next_id, counts)

    def trigram_probability(
        self,
        token_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        # w is token_id. h = (u, v) is the trigram history, with
        # v = counts.previous_id. counts.trigram_counts[w] is c(h, w), and
        # counts.trigram_total is c(h).
        # For trigrams, the observed row is the ordinary c(prev2, prev1, next)
        # row. The backed-off probability is the Kneser-Ney bigram row for the
        # same prev1 token, built from continuation counts at training time.
        lower_order_probability = self.bigram_probability(
            token_id,
            previous_id=counts.previous_id,
            counts=counts.bigram_counts,
            total=counts.bigram_total,
        )
        return self._discounted_interpolation_probability(
            token_id,
            counts=counts.trigram_counts,
            total=counts.trigram_total,
            lower_order_probability=lower_order_probability,
        )

    def bigram_probability(
        self,
        token_id: int,
        *,
        previous_id: int,
        counts: Mapping[int, int] | None = None,
        total: int | None = None,
    ) -> float:
        if counts is None:
            counts = dict(self.bigram_transitions.get(previous_id, ()))
        if total is None:
            total = sum(counts.values())

        # Here the history is h = v = previous_id. counts[w] is
        # c_KN(v, w) = N_{1+}(*, v, w), and total is c_KN(v).
        # This is still discounted interpolation, but the counts are
        # continuation counts: "how many distinct left contexts support this
        # bigram type?" rather than raw bigram frequency.
        return self._discounted_interpolation_probability(
            token_id,
            counts=counts,
            total=total,
            lower_order_probability=self.unigram_probability(token_id),
        )

    def unigram_probability(self, token_id: int) -> float:
        candidate_count = self.candidate_count
        if candidate_count <= 0:
            return 0.0

        # The unigram KN distribution is also a continuation distribution:
        # tokens that appear after many different predecessors get more mass.
        # Uniform probability is only the final floor for unseen continuations.
        uniform_probability = 1 / candidate_count  # P_0(w) = 1 / |V|.
        return self._discounted_interpolation_probability(
            token_id,
            counts=self.unigram_counts,
            total=self.unigram_total,
            lower_order_probability=uniform_probability,
        )

    def _discounted_interpolation_probability(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        total: int,
        lower_order_probability: float,
    ) -> float:
        # Symbol map: w = token_id, c(h, w) = counts[w], c(h) = total,
        # D = self.discount, T(h) = len(counts), and
        # P_lower(w) = lower_order_probability.
        # Implements max(c - D, 0) / total + (D * T / total) * P_lower, where
        # T is len(counts), the number of observed continuation types in a row.
        return ngram.discounted_interpolation_probability(
            token_id,
            counts=counts,
            total=total,
            discount=self.discount,
            lower_order_probability=lower_order_probability,
        )


def load(model_path: Path) -> KneserNeyTrigramModel:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return KneserNeyTrigramModel(
        **model_fields,
        discount=float(data["discount"]),
        unigram_counts=ngram.parse_token_counts(data, "kneser_ney_unigrams"),
        unigram_total=int(data["kneser_ney_unigram_count"]),
        bigram_transitions=ngram.parse_token_transitions(
            data,
            "kneser_ney_bigram_transitions",
        ),
        trigram_transitions=trigrams.parse_context_transitions(
            data,
            "trigram_transitions",
        ),
    )


def train(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    discount: float = 0.75,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> KneserNeyTrigramTrainingSummary:
    artifacts = trigrams.collect_training_artifacts(
        texts,
        tokenizer_model=tokenizer_model,
        text_normalization=text_normalization,
    )
    summary = KneserNeyTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=artifacts.tokenizer.vocab_size,
        discount=discount,
        text_normalization=text_normalization,
    )
    # KN stores ordinary trigram rows for the highest-order evidence, but the
    # lower-order rows are continuation tables derived from trigram types.
    continuation_counts = collect_kneser_ney_continuation_counts(
        artifacts.counts.trigram_transitions,
    )
    trigrams.apply_trigram_counts_to_summary(summary, artifacts.counts)
    summary.continuation_unigram_count = continuation_counts.unigram_count
    summary.continuation_bigram_type_count = continuation_counts.bigram_type_count

    model = {
        **trigrams.standard_trigram_model_payload(
            artifacts.tokenizer,
            model_type=_SCHEMA_TYPE,
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            text_normalization=text_normalization,
            counts=artifacts.counts,
        ),
        "discount": summary.discount,
        "kneser_ney_unigram_count": summary.continuation_unigram_count,
        "kneser_ney_unigrams": ngram.token_counts_payload(
            continuation_counts.unigram_counts
        ),
        "kneser_ney_bigram_transitions": ngram.token_transition_payload(
            continuation_counts.bigram_transitions
        ),
    }
    ngram.write_json_model_payload(output_path, model)

    return summary


def format_summary(
    summary: KneserNeyTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Interpolated Kneser-Ney trigram model file",
        ),
        ("Continuation unigrams", f"{summary.continuation_unigram_count:,}"),
        ("Continuation bigram types", f"{summary.continuation_bigram_type_count:,}"),
        trigrams.discount_item(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train,
    summary_items=format_summary,
    load_model=load,
    evaluation_items=trigrams.discounted_evaluation_items,
    training_option_names=("discount",),
)


def collect_kneser_ney_continuation_counts(
    trigram_transitions: (
        defaultdict[trigrams.Context, Counter[int]]
        | dict[trigrams.Context, Counter[int]]
    ),
) -> KneserNeyContinuationCounts:
    """Collapse raw trigram types into continuation-count lower-order rows."""

    bigram_transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    unigram_predecessors: defaultdict[int, set[int]] = defaultdict(set)

    for (_left_id, previous_id), next_counts in trigram_transitions.items():
        # _left_id is u and previous_id is v in the trigram type c(u, v, w).
        for next_id, count in next_counts.items():
            if count <= 0:
                continue
            # next_id is w. This increments c_KN(v, w) = N_{1+}(*, v, w)
            # once for each distinct positive left context u.
            # Each positive trigram type contributes one continuation for the
            # lower-order bigram row, regardless of its raw token frequency.
            bigram_transitions[previous_id][next_id] += 1
            # Track the support set for c_KN(w) = N_{1+}(*, w).
            unigram_predecessors[next_id].add(previous_id)

    return KneserNeyContinuationCounts(
        unigram_counts=Counter(
            {
                token_id: len(predecessors)
                for token_id, predecessors in unigram_predecessors.items()
            }
        ),
        bigram_transitions=bigram_transitions,
    )
