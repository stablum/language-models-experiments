"""Shared training-summary formatting for registered language models."""

from __future__ import annotations

from collections.abc import Mapping

from src.models.core import formatting
from src.models.core import ngram


_BASE_FIELDS = frozenset(
    (
        "output_path",
        "tokenizer_model",
        "vocab_size",
        "sequence_count",
        "token_count",
        "text_normalization",
    )
)
_ORDERED_MODEL_FIELDS = (
    "transition_count",
    "unigram_count",
    "bigram_transition_count",
    "trigram_transition_count",
    "continuation_unigram_count",
    "continuation_bigram_type_count",
    "smoothing_label",
    "smoothing",
    "discount",
)
_INTERP_FIELDS = frozenset(
    (
        "unigram_weight",
        "bigram_weight",
        "trigram_weight",
        "beta_2",
        "beta_3",
    )
)
_FIELD_LABELS = {
    "transition_count": "Transitions",
    "unigram_count": "Unigrams",
    "bigram_transition_count": "Bigram transitions",
    "trigram_transition_count": "Trigram transitions",
    "continuation_unigram_count": "Continuation unigrams",
    "continuation_bigram_type_count": "Continuation bigram types",
    "smoothing_label": "Smoothing",
    "smoothing": "Smoothing",
    "discount": "Discount",
}


def format_items(
    summary: ngram.NgramTrainingSummary,
    *,
    model_label: str,
) -> list[tuple[str, str]]:
    """Format common artifact rows plus model-owned summary fields."""
    return [
        ("Model", model_label),
        *ngram.base_training_summary_items(
            summary=summary,
            artifact_label="Model file",
        ),
        *model_items(summary),
    ]


def model_items(summary: object) -> list[tuple[str, str]]:
    """Format every known model-owned field present on a training summary."""
    fields = summary_fields(summary)
    field_set = frozenset(fields)
    handled = set(_BASE_FIELDS)
    items = [
        item
        for name in _ORDERED_MODEL_FIELDS
        if name in field_set and (item := field_item(summary, name)) is not None
    ]
    handled.update(_ORDERED_MODEL_FIELDS)

    if _INTERP_FIELDS <= field_set:
        items.extend(interpolation_items(summary))
        handled.update(_INTERP_FIELDS)

    items.extend(
        item
        for name in fields
        if name not in handled and (item := field_item(summary, name)) is not None
    )
    return items


def summary_fields(summary: object) -> tuple[str, ...]:
    """Return declared summary data fields, preferring Pydantic model fields."""
    model_fields = getattr(type(summary), "model_fields", None)
    if isinstance(model_fields, Mapping):
        return tuple(str(name) for name in model_fields)
    return tuple(
        name
        for name in vars(summary)
        if not name.startswith("_")
    )


def field_item(summary: object, name: str) -> tuple[str, str] | None:
    """Format one summary field row, skipping absent or empty values."""
    value = getattr(summary, name, None)
    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    return field_label(name), format_value(value)


def interpolation_items(summary: object) -> list[tuple[str, str]]:
    """Format lambda and beta interpolation params as grouped summary rows."""
    return [
        (
            "Interpolation weights",
            formatting.format_interpolation_weights(
                unigram_weight=float(getattr(summary, "unigram_weight")),
                bigram_weight=float(getattr(summary, "bigram_weight")),
                trigram_weight=float(getattr(summary, "trigram_weight")),
            ),
        ),
        (
            "Interpolation betas",
            (
                f"beta_2={float(getattr(summary, 'beta_2')):.3f}, "
                f"beta_3={float(getattr(summary, 'beta_3')):.3f}"
            ),
        ),
    ]


def field_label(name: str) -> str:
    """Return the display label for one summary field name."""
    return _FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


def format_value(value: object) -> str:
    """Format scalar summary values for compact CLI/report display."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
