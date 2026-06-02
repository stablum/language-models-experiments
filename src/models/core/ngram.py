"""Shared helpers for small token-level n-gram models."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import pydantic

from src.corpora import normalization
from src.ml_core import json_io
from src.models.core import formatting
from src.models.core import naming
from src.tokenizers import core as tok_core


DecodingMode = Literal["sample", "most-probable"]  # Restrict configs to supported decoders.
MODEL_SCHEMA_VERSION = 1  # Pin artifact envelopes so incompatible JSON fails fast.


class NgramSchema(pydantic.BaseModel):
    """Centralize Pydantic validation settings for n-gram data shapes."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class NgramQueryCfg(NgramSchema):
    """Validate user-facing generation controls before querying a model."""

    prompt: str = ""
    max_tokens: int = pydantic.Field(default=80, ge=0)
    top_k: int = pydantic.Field(default=10, ge=1)
    decoding: DecodingMode = "sample"
    temperature: float = pydantic.Field(default=1.0, ge=0)
    seed: int | None = None


class FrozenNgramSchema(NgramSchema):
    """Freeze reusable count/probability data to protect cached rows."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        validate_assignment=True,
    )


class NgramPrediction(NgramSchema):
    """Describe one next-token candidate with its evidence and probability."""

    token_id: int
    piece: str
    count: int
    prob: float  # prob = P(w | h), the next-token probability.


class NgramQueryResult(NgramSchema):
    """Bundle generated text with prompt metadata and the initial prediction row."""

    model_path: Path
    tokenizer_model: Path
    decoding: DecodingMode
    bos_id: int
    eos_id: int
    unk_id: int
    prompt: str
    prompt_token_ids: list[int]
    continuation_text: str
    generated_text: str
    generated_token_ids: list[int]
    token_ids: list[int]
    next_token_predictions: list[NgramPrediction]
    text_normalization: str = "none"


class NgramTrainingSummary(NgramSchema):
    """Carry common training metadata shared by all n-gram model families."""

    output_path: Path | None = None
    tokenizer_model: Path | None = None
    vocab_size: int = 0
    sequence_count: int = 0
    token_count: int = 0
    text_normalization: str = "none"


TrainSummaryT = TypeVar(
    "TrainSummaryT",
    bound=NgramTrainingSummary,
)  # Preserve each model family's concrete summary type.


class TrainingResult(NgramSchema, Generic[TrainSummaryT]):
    """Pair a typed training summary with its JSON-ready model payload."""

    summary: TrainSummaryT
    payload: dict[str, object]


class EvaluationAccumulatorMixin:
    """Mutate an evaluation summary as token sequences and events are scored."""

    def observe_sequence(self, tok_ids: Sequence[int]) -> None:
        """Record one evaluated token sequence before scoring its events."""
        self.sequence_count += 1
        self.token_count += len(tok_ids)

    def score_next_token(
        self,
        *,
        actual_id: int,
        greedy_id: int,
        top_k_ids: frozenset[int],
        prob: float,
    ) -> None:
        """Accumulate accuracy and log-loss metrics for one prediction event."""
        # actual_id is the observed next token w; prob is P(w | h).
        self.transition_count += 1
        if actual_id == greedy_id:
            self.correct_next_token_count += 1
        if actual_id in top_k_ids:
            self.top_k_correct_next_token_count += 1

        if prob <= 0:
            self.zero_probability_count += 1
        else:
            self.negative_log_likelihood -= math.log(prob)


class EvaluationMetricsMixin:
    """Expose derived metrics computed from accumulated evaluation counts."""

    @property
    def next_token_accuracy(self) -> float | None:
        """Return the greedy next-token accuracy over evaluated events."""
        return self._event_rate(self.correct_next_token_count)

    @property
    def top_k_accuracy(self) -> float | None:
        """Return the top-k next-token accuracy over evaluated events."""
        return self._event_rate(self.top_k_correct_next_token_count)

    @property
    def average_negative_log_likelihood(self) -> float | None:
        """Return the mean NLL per prediction event, or infinity for zeros."""
        if self.transition_count == 0:
            return None
        if self.zero_probability_count:
            return math.inf
        return self.negative_log_likelihood / self.transition_count

    @property
    def cross_entropy_bits(self) -> float | None:
        """Return the average negative log-likelihood measured in bits."""
        average_nll = self.average_negative_log_likelihood
        if average_nll is None:
            return None
        return average_nll / math.log(2)

    @property
    def perplexity(self) -> float | None:
        """Return exp(mean NLL), the standard token-level perplexity."""
        average_nll = self.average_negative_log_likelihood
        if average_nll is None:
            return None
        if math.isinf(average_nll):
            return math.inf
        return math.exp(average_nll)

    def _event_rate(self, numerator: int) -> float | None:
        """Compute rates over scored events so metric denominators stay local."""
        if self.transition_count == 0:
            return None
        return numerator / self.transition_count


class NgramEvaluationSummary(
    EvaluationAccumulatorMixin,
    EvaluationMetricsMixin,
    NgramSchema,
):
    """Store raw and derived statistics for an n-gram evaluation run."""

    model_path: Path
    tokenizer_model: Path
    top_k: int
    sequence_count: int = 0
    token_count: int = 0
    transition_count: int = 0
    correct_next_token_count: int = 0
    top_k_correct_next_token_count: int = 0
    negative_log_likelihood: float = 0.0
    zero_probability_count: int = 0
    text_normalization: str = "none"


class BaseNgramModel(NgramSchema):
    """Provide shared query/generation behavior for concrete n-gram ML models."""

    model_path: Path
    tokenizer_model: Path
    tokenizer: tok_core.TokenizerCodec
    vocab_size: int
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    text_normalization: str = "none"
    _cand_ids: tuple[int, ...] = pydantic.PrivateAttr(default=())
    _cand_id_set: frozenset[int] = pydantic.PrivateAttr(default_factory=frozenset)

    @property
    def cand_ids(self) -> tuple[int, ...]:
        """Return candidate next-token IDs, excluding BOS as a target."""
        # cand = candidate next-token IDs, i.e. V without BOS.
        if not self._cand_ids:
            self._cand_ids = tuple(
                token_id
                for token_id in range(self.vocab_size)
                if token_id != self.bos_id
            )
        return self._cand_ids

    @property
    def cand_id_set(self) -> frozenset[int]:
        """Return the candidate next-token IDs as a cached membership set."""
        # cand = candidate next-token ID set for O(1) membership tests.
        if not self._cand_id_set:
            self._cand_id_set = frozenset(self.cand_ids)
        return self._cand_id_set

    @property
    def cand_count(self) -> int:
        """Return the number of candidate next-token IDs."""
        # cand = candidate vocabulary size |V|.
        return len(self.cand_ids)

    def candidate_counts(
        self,
        row: Mapping[int, int] | Iterable[tuple[int, int]],
    ) -> dict[int, int]:
        """Return positive c(h, w) counts restricted to candidate next tokens."""
        items = row.items() if isinstance(row, Mapping) else row
        cand_ids = self.cand_id_set
        return {
            token_id: count
            for token_id, count in items
            if token_id in cand_ids and count > 0
        }

    def encode_prompt(self, prompt: str) -> list[int]:
        """Normalize and tokenize a query prompt with the model tokenizer."""
        return tok_core.encode_prompt(
            self.tokenizer,
            prompt,
            text_normalization=self.text_normalization,
        )

    def context_for_tokens(self, token_ids: list[int]) -> Any:
        """Build a model-specific history context from prompt token IDs."""
        raise NotImplementedError

    def advance_context(self, context: Any, next_id: int) -> Any:
        """Update a model-specific history context after generating one token."""
        raise NotImplementedError

    def next_token_predictions(
        self,
        context: Any,
        *,
        top_k: int,
    ) -> list[NgramPrediction]:
        """Return ranked next-token predictions for a model-specific context."""
        raise NotImplementedError

    def query(self, cfg: NgramQueryCfg | None = None) -> NgramQueryResult:
        """Generate a continuation and expose the initial next-token row."""
        cfg = cfg or NgramQueryCfg()
        prompt_ids = self.encode_prompt(cfg.prompt)  # ids = token IDs for prompt.
        context = self.context_for_tokens(prompt_ids)
        next_preds = self.next_token_predictions(
            context,
            top_k=cfg.top_k,
        )
        gen_top_k = generation_prediction_top_k(
            decoding=cfg.decoding,
            temperature=cfg.temperature,
        )
        # Deterministic text sampling RNG; not used for secrets or security choices.
        rng = random.Random(cfg.seed)  # nosec B311
        all_ids = list(prompt_ids)  # ids = prompt plus generated token IDs.
        gen_ids: list[int] = []  # gen = generated continuation token IDs.

        for _ in range(cfg.max_tokens):
            next_id = select_next_token(
                self.next_token_predictions(context, top_k=gen_top_k),
                eos_id=self.eos_id,
                decoding=cfg.decoding,
                rng=rng,
                temperature=cfg.temperature,
            )
            if next_id == self.eos_id:
                break

            gen_ids.append(next_id)
            all_ids.append(next_id)
            context = self.advance_context(context, next_id)

        prompt_text = self.tokenizer.decode(prompt_ids)
        generated_text = self.tokenizer.decode(all_ids)
        continuation_text = tok_core.decode_continuation(
            self.tokenizer,
            generated_text=generated_text,
            prompt_text=prompt_text,
            generated_token_ids=gen_ids,
        )

        return NgramQueryResult(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            decoding=cfg.decoding,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            unk_id=self.unk_id,
            prompt=cfg.prompt,
            prompt_token_ids=prompt_ids,
            continuation_text=continuation_text,
            generated_text=generated_text,
            generated_token_ids=gen_ids,
            token_ids=all_ids,
            next_token_predictions=next_preds,
            text_normalization=self.text_normalization,
        )


def select_next_token(
    preds: Sequence[NgramPrediction],
    *,
    eos_id: int,
    decoding: DecodingMode,
    rng: random.Random,
    temperature: float,
) -> int:
    """Choose the next token according to the requested decoding policy."""
    if decoding == "most-probable":
        return most_probable_token(preds, eos_id=eos_id)
    if decoding == "sample":
        return sample_token(preds, eos_id=eos_id, rng=rng, temperature=temperature)
    raise ValueError(f"Unsupported decoding mode: {decoding}")


def most_probable_token(
    preds: Sequence[NgramPrediction],
    *,
    eos_id: int,
) -> int:
    """Return the highest-probability token, falling back to EOS if empty."""
    if not preds:
        return fallback_token_id(eos_id)
    return preds[0].token_id


def sample_token(
    preds: Sequence[NgramPrediction],
    *,
    eos_id: int,
    rng: random.Random,
    temperature: float,
) -> int:
    """Sample one token from temperature-scaled next-token probabilities."""
    if not preds:
        return fallback_token_id(eos_id)
    if temperature == 0:
        return preds[0].token_id
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    ws = [pred.prob ** (1 / temperature) for pred in preds]  # w = sample weights.
    if not any(ws):
        return preds[0].token_id

    return rng.choices(
        [pred.token_id for pred in preds],
        weights=ws,
        k=1,
    )[0]


def fallback_token_id(eos_id: int) -> int:
    """Return a safe terminal token when a prediction row is unavailable."""
    return eos_id if eos_id >= 0 else 0


def generation_prediction_top_k(*, decoding: DecodingMode, temperature: float) -> int:
    """Choose how many candidates generation must request from the model."""
    # top_k=0 means "all candidates"; greedy paths only need the first row.
    if decoding == "most-probable" or temperature == 0:
        return 1
    return 0


def sorted_predictions(
    preds: Iterable[NgramPrediction],
    *,
    top_k: int,
) -> list[NgramPrediction]:
    """Sort predictions by probability and optionally truncate to top-k."""
    ranked_preds = sorted(
        preds,
        key=lambda pred: (-pred.prob, pred.token_id),
    )
    return ranked_preds[:top_k] if top_k > 0 else ranked_preds


def greedy_id(ranked_ids: Sequence[int], *, eos_id: int) -> int:
    """Return the first ranked token ID, or a fallback terminal token."""
    if ranked_ids:
        return ranked_ids[0]
    return fallback_token_id(eos_id)


def top_k_id_set(ranked_ids: Sequence[int], *, top_k: int) -> frozenset[int]:
    """Return the top-k ranked token IDs as a membership set."""
    return frozenset(ranked_ids[:top_k]) if top_k > 0 else frozenset()


def resolve_stored_path(stored_path: Path, model_path: Path) -> Path:
    """Resolve artifact-relative paths beside their serialized model file."""
    if stored_path.is_absolute() or stored_path.exists():
        return stored_path

    model_relative_path = model_path.parent / stored_path
    if model_relative_path.exists():
        return model_relative_path

    return stored_path


def model_schema_payload(module_name: str) -> dict[str, object]:
    """Build the common schema envelope for a model artifact."""
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": naming.model_type_from_module(module_name),
    }


def load_json_model_payload(
    model_path: Path,
    *,
    module_name: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Read and validate a JSON model artifact for one model module."""
    model_type = naming.model_type_from_module(module_name)
    data = json_io.read_mapping(model_path)
    if data.get("model_type") != model_type:
        raise ValueError(f"Not {label or naming.schema_label(model_type)}: {model_path}")
    return data


def write_json_model_payload(output_path: Path, model: dict[str, object]) -> None:
    """Write a JSON model artifact to its chosen output path."""
    json_io.write_json(output_path, model)


def tokenizer_model_payload(
    tokenizer: tok_core.TokenizerCodec,
    *,
    tokenizer_model: Path,
    stored_tokenizer_model: Path | None,
    text_normalization: normalization.TextNormalization,
) -> dict[str, object]:
    """Build tokenizer metadata fields stored inside model artifacts."""
    return tok_core.tokenizer_payload(
        tokenizer,
        tokenizer_model=tokenizer_model,
        stored_tokenizer_model=stored_tokenizer_model,
        text_normalization=text_normalization,
    )


def load_tokenizer_model_fields(
    data: dict[str, object],
    model_path: Path,
) -> dict[str, object]:
    """Load tokenizer-dependent constructor fields from model JSON data."""
    tokenizer_model = resolve_stored_path(Path(data["tokenizer_model"]), model_path)
    tokenizer = tok_core.load_tokenizer(
        tokenizer_model,
        tokenizer_algo=str(data["tokenizer_algo"]) if data.get("tokenizer_algo") else None,
    )
    vocab_size = int(data.get("vocab_size", tokenizer.vocab_size))
    stored_pieces = data.get("pieces")
    pieces = (
        tuple(str(piece) for piece in stored_pieces)
        if stored_pieces
        else tuple(tokenizer.id_to_piece(idx) for idx in range(vocab_size))
    )
    return {
        "model_path": model_path,
        "tokenizer_model": tokenizer_model,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "bos_id": int(data["bos_id"]),
        "eos_id": int(data["eos_id"]),
        "unk_id": int(data["unk_id"]),
        "pieces": pieces,
        "text_normalization": str(data.get("text_normalization", "none")),
    }


def add_k_prob(
    token_id: int,
    *,
    counts: Mapping[int, int],
    tot: int,
    smoothing: float,
    cand_count: int,
) -> float:
    """Compute the Lidstone/add-k probability for one candidate token."""
    denom = tot + smoothing * cand_count  # denom = c(h) + k |V|.
    if denom <= 0:
        return 0.0
    return (counts.get(token_id, 0) + smoothing) / denom


def ml_prob(
    token_id: int,
    *,
    counts: Mapping[int, int],
    tot: int,
) -> float:
    """Compute the maximum-likelihood probability for one candidate token."""
    if tot <= 0:
        return 0.0
    return counts.get(token_id, 0) / tot


def discounted_interp_prob(
    token_id: int,
    *,
    counts: Mapping[int, int],
    tot: int,
    discount: float,
    lower_prob: float,
) -> float:
    """Compute one discounted row probability interpolated with lower order."""
    if tot <= 0:
        return lower_prob

    obs_count = counts.get(token_id, 0)  # obs = observed count c(h, w).
    disc_prob = max(obs_count - discount, 0.0) / tot
    interp_w = discount * len(counts) / tot  # w = interpolation weight lambda(h).
    return disc_prob + interp_w * lower_prob


def base_training_summary_items(
    *,
    summary: NgramTrainingSummary,
    artifact_label: str,
) -> list[tuple[str, str]]:
    """Format common training-summary fields for model display."""
    return [
        ("Tokenizer model file", formatting.artifact_filename(summary.tokenizer_model)),
        (artifact_label, formatting.artifact_filename(summary.output_path)),
        ("Text normalization", summary.text_normalization),
        ("Vocabulary size", f"{summary.vocab_size:,}"),
        ("Sequences", f"{summary.sequence_count:,}"),
        ("Tokens", f"{summary.token_count:,}"),
    ]


def base_evaluation_items(summary: NgramEvaluationSummary) -> list[tuple[str, str]]:
    """Format common evaluation artifact fields for model display."""
    return [
        ("Model file", formatting.artifact_filename(summary.model_path)),
        ("Tokenizer model file", formatting.artifact_filename(summary.tokenizer_model)),
        ("Text normalization", summary.text_normalization),
    ]


def parse_token_transitions(
    data: dict[str, object],
    key: str,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Parse sparse token transition rows from serialized JSON data."""
    return {
        int(prev_id): tuple(
            (int(next_id), int(count))
            for next_id, count in next_counts
        )
        for prev_id, next_counts in data[key].items()
    }


def parse_token_counts(data: dict[str, object], key: str) -> dict[int, int]:
    """Parse serialized token-count pairs into an integer-keyed mapping."""
    return {
        int(token_id): int(count)
        for token_id, count in data[key]
    }


def token_transition_payload(
    transitions: defaultdict[int, Counter[int]] | dict[int, Counter[int]],
) -> dict[str, list[tuple[int, int]]]:
    """Serialize sparse transition counters into deterministic JSON rows."""
    return {
        str(prev_id): sorted(next_counts.items())
        for prev_id, next_counts in sorted(transitions.items())
    }


def token_counts_payload(counts: Counter[int] | Mapping[int, int]) -> list[tuple[int, int]]:
    """Serialize token counts as sorted token-count pairs."""
    return sorted(counts.items())


def iter_token_sequences(
    texts: Iterable[str],
    tokenizer: tok_core.TokenizerCodec,
    *,
    bos_count: int,
    min_length: int,
    text_normalization: normalization.TextNormalization = "none",
) -> Iterator[list[int]]:
    """Yield normalized token sequences with configured BOS/EOS handling."""
    yield from tok_core.iter_token_sequences(
        texts,
        tokenizer,
        bos_count=bos_count,
        min_length=min_length,
        text_normalization=text_normalization,
    )
