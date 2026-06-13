# Adding A New Model

Add one concrete language-model module under `src/models`:

```text
src/models/my_model.py
```

The registered CLI name is derived from the module filename by replacing
underscores with hyphens. For example, `src/models/trigram_add_k.py` is
registered as `trigram-add-k`.

Shared helpers that are useful to more than one model belong under
`src/models/core`. Keep concrete model modules in `src/models`; keep reusable
math, serialization, formatting, or count-collection code in `src/models/core`.

## Discovery Contract

The registry imports every non-package module in `src/models` whose filename
does not start with `_`. A module is registered when it exposes:

- `fit(...)`
- `load(model_path)`
- `format_summary(summary)`

To keep a work-in-progress module in `src/models` without registering or
importing it, add this top-level source flag:

```python
REGISTER_MODEL = False
```

Leave the flag absent or set it to `True` when the model should be registered.
Because model modules are imported during CLI startup, keep module top-level
code side-effect free. Do not train, read large artifacts, or call external
services at import time.

## Usual Module Structure

A typical n-gram model module should expose these pieces, in this order:

```python
"""Short explanation of the model and its probability estimate."""

from __future__ import annotations

from pathlib import Path

from src.models.core import ngram
from src.models.core import token_sequences


CONTEXT_LENGTH = 1  # len(h), the token history size required by this model.


class TrainingSummary(ngram.NgramTrainingSummary):
    """Store model-specific training metrics."""


class Model(ngram.BaseNgramModel):
    """Store learned token-space state and expose scoring behavior."""


def load(model_path: Path) -> Model:
    """Load a persisted model artifact into a queryable model."""
    ...


def fit(
    corpus: token_sequences.TokenCorpus,
    *,
    # model hyperparameters go here
) -> ngram.TrainingResult[TrainingSummary]:
    """Fit learned state from token IDs and return a JSON-ready payload."""
    ...


def format_summary(summary: TrainingSummary) -> list[tuple[str, str]]:
    """Format training metrics for CLI and tracker display."""
    ...
```

Omit the module-local `TrainingSummary` when a shared summary type already has
all the fields the module needs. In that case, annotate `fit(...)` and
`format_summary(...)` with the shared type directly.

## Required Pieces

`TrainingSummary` or shared summary type

Use the nearest shared pydantic summary type, normally
`ngram.NgramTrainingSummary`, `trigrams.TrigramTrainingSummary`, or
`trigrams.InterpolatedTrigramTrainingSummary`. Define a module-local
`TrainingSummary` only when the module needs additional fields beyond that
base, such as a new training count, model-family diagnostic, or resolved
hyperparameter.

When a module-local summary class is needed, use a module-local name such as
`TrainingSummary`; the module namespace already carries the model identity.

`Model`

For simple n-gram models, inherit from `ngram.BaseNgramModel`. It stores
token-space metadata and generic candidate helpers. You must provide:

- `context_for_tokens(token_ids)`
- `advance_context(context, next_id)`
- `next_token_predictions(context, *, top_k)`
- `evaluate_token_corpus(corpus, *, top_k)`

The runtime owns text prompt encoding, generated-token decoding, and corpus
tokenization. Trigram models should usually inherit from
`trigrams.BaseTrigramModel`, `trigrams.InterpolatedTrigramModel`, or
`trigrams.DiscountedTrigramModel`, which provide most of the token scoring
machinery.

For non-n-gram models, keep the same module-level `fit`, `load`, and
`format_summary` contract. The loaded `Model` object should still expose
token-space context, next-token, and evaluation methods when the full pipeline
should support query/evaluation stages.

`load(model_path)`

Read the persisted artifact, validate the current schema version plus
`model_type`, load any token-space/model fields, and return the model object.
Reuse helpers such as:

- `ngram.load_json_model_payload(...)`
- `ngram.load_token_space_model_fields(...)`
- `trigrams.load_standard_trigram_model_fields(...)`
- `ngram.parse_token_counts(...)`
- `ngram.parse_token_transitions(...)`
- `trigrams.parse_trigram_transitions(...)`

Pass `module_name=__name__` to the standard load helpers. They derive the
artifact `model_type` from the module leaf, so `src.models.my_model` expects
`model_type: "my_model"`. Do not add a second hand-written schema name unless
the model really needs a separate artifact family.

`fit(...)`

Expose `fit(...)`, not `train(...)`. In this project, `fit` means "estimate
learned state from data and return the artifact payload." For n-grams that
learned state is usually sufficient statistics such as `c(h,w)` count rows.
For gradient-based models, `fit(...)` may instantiate a live `Model`, pass it
to a trainer, update its weights batch by batch, run epoch validation, and then
return a checkpoint/artifact summary.

Required shape:

- first parameter: `corpus: token_sequences.TokenCorpus`

`corpus` streams `token_sequences.TokenSeq` rows that are already normalized,
tokenized, and padded with the model's BOS context tokens by the runtime
adapter. It also carries `corpus.vocab_size`, the token-space dimension |V|.
Fit functions should not accept raw text, tokenizer objects,
text-normalization settings, artifact paths, or tokenizer display metadata.

`fit(...)` should return `ngram.TrainingResult[SummaryType]`, which contains:

- `summary`: the training summary object
- `payload`: only the model-owned JSON payload fields

The pipeline/runtime owns artifact paths, tokenizer metadata, text
normalization, schema envelope fields, and portable tokenizer references. Do
not write the final JSON model artifact inside the model module's `fit(...)`.

If a model needs validation data during fitting, add this keyword-only
parameter:

```python
validation_corpus: token_sequences.TokenCorpus | None = None
```

Use validation data for epoch metrics, early stopping, or checkpoint selection.
Keep final benchmark evaluation in the evaluation stage unless the model
genuinely needs validation feedback while fitting.

Model hyperparameters should be keyword-only parameters on `fit(...)`. The
pipeline passes matching CLI/pipeline options through to the model module.

`CONTEXT_LENGTH`

Expose a non-negative module-level `CONTEXT_LENGTH`. This is `len(h)`, the
number of previous token IDs the model needs before a target token. The runtime
uses it to build token sequences from text:

- bigram: `CONTEXT_LENGTH = 1`
- trigram: `CONTEXT_LENGTH = 2`

`format_summary(summary)`

Return a `list[tuple[str, str]]` for CLI/ClearML display. Reuse
`ngram.base_training_summary_items(...)` or
`trigrams.base_training_summary_items(...)` when possible.

## Optional Hooks

Add these functions only when the model needs them:

- `format_query(result)` if the standard n-gram query display is not enough
- `format_evaluation(summary)` if the standard evaluation display is not enough
- `validate_fit_options(options)` for coupled or non-trivial option checks

If you add `validate_fit_options`, import
`from src.ml_core.models import definition as model_def` and raise
`model_def.ModelOptionError` when user-supplied options are invalid.

## Hyperparameters

If a new model uses existing hyperparameters such as `smoothing`, `discount`,
`unigram_weight`, `bigram_weight`, `trigram_weight`, `beta_2`, or `beta_3`,
add them as keyword-only parameters on `fit(...)`.

If the model needs a brand-new hyperparameter, add it consistently in:

- `src/pipelines/language_model/model_options.py`
- `src/cli/options.py`
- `src/cli/model_training_flow.py`
- `src/cli/train.py`
- `src/cli/model_training_defaults.py`
- `src/pipelines/language_model/artifacts.py` when it should be logged
- `src/pipelines/language_model/optuna.py` when it should be searchable

Then add the new keyword-only parameter to the model module's `fit(...)`
signature.

## N-Gram Starting Points

Use the existing modules as templates:

- `src/models/bigram.py` for a compact first-order model.
- `src/models/trigram_add_k.py` for an interpolated trigram using shared
  interpolation helpers.
- `src/models/trigram_absolute_discount.py` for a discounted trigram.
- `src/models/trigram_good_turing.py` for a model that customizes ranking and
  probability caching.
- `src/models/trigram_kneser_ney.py` for a model with extra derived count
  tables.

Prefer pulling shared count collection, probability formulas, payload parsing,
or formatting into `src/models/core` once a second model needs the same logic.

## Documentation Checklist

When a real model is added, also update `MODELS.md` with the model's registered
name, probability definition, main assumptions, hyperparameters, and evaluation
interpretation notes.
