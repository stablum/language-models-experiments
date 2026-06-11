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
from src.tokenizers import core as tok_core


class TrainingSummary(trigrams.TrigramTrainingSummary):
    """Store Kneser-Ney continuation counts and discount metadata."""

    continuation_unigram_count: int = 0  # sum_w c_KN(w), unigram continuation mass.
    continuation_bigram_type_count: int = 0  # |{(v, w): N_{1+}(*, v, w) > 0}|.
    discount: float = 0.0  # D, the absolute discount.


class ContinuationCounts(ngram.FrozenNgramPydanticBase):
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


class Model(trigrams.DiscountedTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        trigrams.DiscountedTrigramEvaluationSummary
    )
    unigram_counts: dict[int, int]  # c_KN(w), continuation unigram counts.
    unigram_tot: int  # tot = sum_w c_KN(w), the unigram row total.

    def context_prob(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        return self._trigram_prob(next_id, counts)

    def _trigram_prob(
        self,
        token_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        # w is token_id. h = (u, v) is the trigram history, with
        # v = counts.prev_id. counts.trigram_counts[w] is c(h, w), and
        # counts.trigram_tot is c(h).
        # For trigrams, the observed row is the ordinary c(prev2, prev1, next)
        # row. The backed-off probability is the Kneser-Ney bigram row for the
        # same prev1 token, built from continuation counts at training time.
        lower_prob = self._bigram_prob(
            token_id,
            prev_id=counts.prev_id,
            counts=counts.bigram_counts,
            tot=counts.bigram_tot,
        )
        return self._discounted_interp_prob(
            token_id,
            counts=counts.trigram_counts,
            tot=counts.trigram_tot,
            lower_prob=lower_prob,
        )

    def _bigram_prob(
        self,
        token_id: int,
        *,
        prev_id: int,
        counts: Mapping[int, int] | None = None,
        tot: int | None = None,
    ) -> float:
        if counts is None:
            row = self.bigram_transitions.get(prev_id, ())
            counts = self.candidate_counts(row)
        if tot is None:
            tot = sum(counts.values())

        # Here the history is h = v = prev_id. counts[w] is
        # c_KN(v, w) = N_{1+}(*, v, w), and tot is c_KN(v).
        # This is still discounted interpolation, but the counts are
        # continuation counts: "how many distinct left contexts support this
        # bigram type?" rather than raw bigram frequency.
        return self._discounted_interp_prob(
            token_id,
            counts=counts,
            tot=tot,
            lower_prob=self._unigram_prob(token_id),
        )

    def _unigram_prob(self, token_id: int) -> float:
        cand_count = self.cand_count
        if cand_count <= 0:
            return 0.0

        # The unigram KN distribution is also a continuation distribution:
        # tokens that appear after many different predecessors get more mass.
        # Uniform probability is only the final floor for unseen continuations.
        uniform_prob = 1 / cand_count  # P_0(w) = 1 / |V|.
        return self._discounted_interp_prob(
            token_id,
            counts=self.unigram_counts,
            tot=self.unigram_tot,
            lower_prob=uniform_prob,
        )

    def _discounted_interp_prob(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        tot: int,
        lower_prob: float,
    ) -> float:
        # Symbol map: w = token_id, c(h, w) = counts[w], c(h) = tot,
        # D = self.discount, T(h) = len(counts), and
        # P_lower(w) = lower_prob.
        # Implements max(c - D, 0) / tot + (D * T / tot) * P_lower, where
        # T is len(counts), the number of observed continuation types in a row.
        return ngram.discounted_interp_prob(
            token_id,
            counts=counts,
            tot=tot,
            discount=self.discount,
            lower_prob=lower_prob,
        )


def load(model_path: Path) -> Model:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        module_name=__name__,
    )

    return Model(
        **model_fields,
        discount=float(data["discount"]),
        unigram_counts=ngram.parse_token_counts(data, "kneser_ney_unigrams"),
        unigram_tot=int(data["kneser_ney_unigram_count"]),
        bigram_transitions=ngram.parse_token_transitions(
            data,
            "kneser_ney_bigram_transitions",
        ),
        trigram_transitions=trigrams.parse_context_transitions(
            data,
            "trigram_transitions",
        ),
    )


def fit(
    texts: Iterable[str],
    *,
    tokenizer: tok_core.TokenizerCodec,
    discount: float = 0.75,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> ngram.TrainingResult[TrainingSummary]:
    """Fit raw trigram counts plus Kneser-Ney continuation tables."""
    def payload(
        counts: trigrams.TrigramCounts,
        summary: TrainingSummary,
    ) -> dict[str, object]:
        # KN stores raw trigram rows plus continuation lower-order tables.
        # cont = continuation lower-order counts.
        cont_counts = collect_kneser_ney_continuation_counts(
            counts.trigram_transitions,
        )
        summary.continuation_unigram_count = cont_counts.unigram_count
        summary.continuation_bigram_type_count = cont_counts.bigram_type_count
        return {
            "discount": summary.discount,
            "kneser_ney_unigram_count": summary.continuation_unigram_count,
            "kneser_ney_unigrams": ngram.token_counts_payload(
                cont_counts.unigram_counts
            ),
            "kneser_ney_bigram_transitions": ngram.token_transition_payload(
                cont_counts.bigram_transitions
            ),
        }

    return trigrams.fit_counted_trigram_model(
        texts,
        tokenizer,
        text_normalization=text_normalization,
        summary_type=TrainingSummary,
        summary_fields={"discount": discount},
        extra_payload=payload,
    )

def collect_kneser_ney_continuation_counts(
    trigram_transitions: (
        defaultdict[trigrams.Context, Counter[int]]
        | dict[trigrams.Context, Counter[int]]
    ),
) -> ContinuationCounts:
    """Collapse raw trigram types into continuation-count lower-order rows."""

    bigram_transitions: defaultdict[int, Counter[int]] = defaultdict(Counter)
    # preds = predecessor token IDs supporting each continuation.
    unigram_preds: defaultdict[int, set[int]] = defaultdict(set)

    for (_left_id, prev_id), next_counts in trigram_transitions.items():
        # _left_id is u and prev_id is v in the trigram type c(u, v, w).
        for next_id, count in next_counts.items():
            if count <= 0:
                continue
            # next_id is w. This increments c_KN(v, w) = N_{1+}(*, v, w)
            # once for each distinct positive left context u.
            # Each positive trigram type contributes one continuation for the
            # lower-order bigram row, regardless of its raw token frequency.
            bigram_transitions[prev_id][next_id] += 1
            # Track the support set for c_KN(w) = N_{1+}(*, w).
            unigram_preds[next_id].add(prev_id)

    return ContinuationCounts(
        unigram_counts=Counter(
            {
                token_id: len(predecessors)
                for token_id, predecessors in unigram_preds.items()
            }
        ),
        bigram_transitions=bigram_transitions,
    )
