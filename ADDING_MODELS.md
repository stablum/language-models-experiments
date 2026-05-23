# Adding A New Model

This project discovers concrete language-model implementations from
`src/models`. A new model normally starts as one new module:

```text
src/models/my_model.py
```

The module adapter derives the registered model name from the module name by
replacing underscores with hyphens. For example, `src/models/trigram_add_k.py`
is registered as `trigram-add-k`.

Shared helpers that are useful to more than one model belong under
`src/models/core`. Keep concrete model modules in `src/models`; keep reusable
math, serialization, formatting, or count-collection code in `src/models/core`.

## Discovery Contract

`src.models.core.registry` imports every non-package module in `src/models`
whose filename does not start with `_`. A module is registered when it exposes
the conventional model functions:

- `train(...)`
- `load(model_path)`
- `format_summary(summary)`

`src.models.core.model_modules` adapts those functions into the shared
`src.ml_core.models.definition.ModelDefinition` contract used by the CLIs and
pipelines. The registered model name is derived from the module name by
replacing underscores with hyphens.

Because the registry imports model modules during CLI startup, keep module
top-level code side-effect free. Do not train, read large artifacts, or call
external services at import time.

## Usual Module Structure

A typical n-gram model module should expose these pieces, in this order:

```python
"""Short explanation of the model and its probability estimate."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.corpora import normalization
from src.models.core import ngram
from src.tokenizers import core as tok_core


class TrainingSummary(ngram.NgramTrainingSummary):
    ...


class Model(ngram.BaseNgramModel):
    ...


def load(model_path: Path) -> Model:
    ...


def train(
    texts: Iterable[str],
    *,
    tokenizer: tok_core.TokenizerCodec,
    text_normalization: normalization.TextNormalization = (
        normalization.DEFAULT_TEXT_NORMALIZATION
    ),
    # model hyperparameters go here
) -> ngram.TrainingResult[TrainingSummary]:
    ...


def format_summary(summary: TrainingSummary) -> list[tuple[str, str]]:
    ...
```

Omit the module-local `TrainingSummary` when a shared summary type already has
all the fields the module needs. In that case, annotate `train(...)` and
`format_summary(...)` with the shared type directly.

## Required Pieces

`TrainingSummary` or shared summary type

Use the nearest shared pydantic summary type, normally
`ngram.NgramTrainingSummary`, `trigrams.TrigramTrainingSummary`, or
`trigrams.InterpolatedTrigramTrainingSummary`. Define a module-local
`TrainingSummary` only when the module needs additional fields beyond that
base. Add only those additional summary values, such as a new training count,
model-family diagnostic, or resolved hyperparameter.

When a module-local summary class is needed, use a module-local name such as
`TrainingSummary`; the module namespace already carries the model identity.

`Model`

For simple n-gram models, inherit from `ngram.BaseNgramModel`. It already
implements prompt encoding and generic autoregressive query generation. You
must provide:

- `context_for_tokens(token_ids)`
- `advance_context(context, next_id)`
- `next_token_predictions(context, *, top_k)`

If the model should support evaluation, also implement `evaluate(...)`, or
inherit from a helper that already does. Trigram models should usually inherit
from `trigrams.BaseTrigramModel`, `trigrams.InterpolatedTrigramModel`, or
`trigrams.DiscountedTrigramModel`, which provide most of the query/evaluation
machinery.

`load(model_path)`

Read the JSON artifact, validate `model_type`, load the tokenizer fields, and
return the model object. Reuse helpers such as:

- `ngram.load_json_model_payload(...)`
- `ngram.load_tokenizer_model_fields(...)`
- `trigrams.load_standard_trigram_model_fields(...)`
- `ngram.parse_token_counts(...)`
- `ngram.parse_token_transitions(...)`
- `trigrams.parse_trigram_transitions(...)`

Pass `module_name=__name__` to the standard load helpers. They derive the
artifact `model_type` from the module leaf, so `src.models.my_model` expects
`model_type: "my_model"`. Do not add a second hand-written schema name unless
you first introduce a new adapter convention that truly needs one.

`train(...)`

This is the function called by the model-module adapter. Its required keyword
parameters are:

- `tokenizer: tok_core.TokenizerCodec`
- `text_normalization: normalization.TextNormalization`

It should train from `texts` and return `ngram.TrainingResult[SummaryType]`,
which contains the training summary and the module-owned JSON payload. Simple
n-gram payloads normally include:

- count tables or learned weights needed by the loader
- model hyperparameters needed at query/evaluation time

The adapter loads the tokenizer, adds schema and tokenizer fields, writes the
JSON artifact to its chosen `output_path`, and records the final artifact paths
on the returned summary. Model modules should not know about staging paths,
portable tokenizer references, or ClearML upload details.

The standard trigram training helpers build the common trigram count payload
for you. Add only the extra payload fields your model needs.

Model hyperparameters should be keyword-only parameters on `train(...)`. The
adapter infers their option names by excluding the infrastructure parameters
listed above.

`format_summary(summary)`

Return a `list[tuple[str, str]]` for CLI/ClearML display. Reuse
`ngram.base_training_summary_items(...)` or
`trigrams.base_training_summary_items(...)` when possible.

Optional module functions:

- `format_query(result)` if the standard n-gram query display is not enough
- `format_evaluation(summary)` if the standard evaluation display is not enough
- `validate_training_options(options)` for coupled or non-trivial option checks

The full model-training pipeline requires both query and evaluation support.
The adapter supplies query support through the loaded model's `query`
method and evaluation support through the loaded model's `evaluate` method.

## Hyperparameters

If a new model uses only existing hyperparameters such as `smoothing`,
`discount`, `unigram_weight`, `bigram_weight`, `trigram_weight`, `beta_2`, or
`beta_3`, add them as keyword-only parameters on `train(...)`. The adapter will
pass through any matching CLI/pipeline option.

If the model needs a brand-new hyperparameter, add it consistently in:

- `src/pipelines/language_model/model_options.py`
- `src/cli/model_training.py`
- `src/cli/train.py`
- `src/cli/model_training_flow.py`
- `src/cli/model_training_defaults.py`
- `src/pipelines/language_model/artifacts.py` when it should be logged
- `src/pipelines/language_model/optuna.py` when it should be searchable

Then add the new keyword-only parameter to the model module's `train(...)`
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

## Custom Non-N-Gram Models

For a model that does not fit the current adapter, extend
`src.models.core.model_modules` with a new convention or strategy. Keep the
concrete model module as the source of truth, and adapt it into
`src.ml_core.models.definition.ModelDefinition` in the shared registry layer
rather than adding one-off registration objects to concrete modules.

If you add a new registry convention, adapt it into `ModelDefinition`
callables with these signatures. These are adapter-level callables, not extra
functions required on ordinary concrete model modules:

- `train(texts, options) -> summary`
- `validate_options(options) -> None`
- `summary_items(summary) -> list[tuple[str, str]]`
- `query(options) -> result`
- `query_lines(result) -> list[str]`
- `evaluate(texts, options) -> summary`
- `evaluation_items(summary) -> list[tuple[str, str]]`

Raise `model_def.ModelOptionError` from validators when user-supplied options
are invalid.

## Documentation Checklist

When a real model is added, also update `MODELS.md` with the model's registered
name, probability definition, main assumptions, hyperparameters, and evaluation
interpretation notes.
