# Adding A New Model

This project discovers concrete language-model implementations from
`src/models`. A new model normally starts as one new module:

```text
src/models/my_model.py
```

When using the standard n-gram helper, the registered model name is derived
from the module name by replacing underscores with hyphens. For example,
`src/models/trigram_add_k.py` is registered as `trigram-add-k`.

Shared helpers that are useful to more than one model belong under
`src/models/core`. Keep concrete model modules in `src/models`; keep reusable
math, serialization, formatting, or count-collection code in `src/models/core`.

## Discovery Contract

`src.models.core.registry` imports every non-package module in `src/models`
whose filename does not start with `_`. A module is registered only when it
defines this module variable:

```python
MODEL_DEFINITION = ...
```

`MODEL_DEFINITION` must be an instance of `src.ml_core.models.definition.ModelDefinition`.
For current token n-gram models, create it with
`src.models.core.ngram.model_definition(...)`; this wires the model into the
training, evaluation, and query stages with the local conventions.

Because the registry imports model modules during CLI startup, keep module
top-level code side-effect free. Do not train, read large artifacts, or call
external services at import time.

## Usual Module Structure

A typical n-gram model module should define these pieces, in this order:

```python
"""Short explanation of the model and its probability estimate."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.corpora import normalization
from src.models.core import ngram


_SCHEMA_TYPE = "my_model"  # Unique model_type stored in the JSON artifact.


class MyModelTrainingSummary(ngram.NgramTrainingSummary):
    ...


class MyModel(ngram.BaseNgramModel):
    ...


def load(model_path: Path) -> MyModel:
    ...


def train(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    text_normalization: normalization.TextNormalization = (
        normalization.DEFAULT_TEXT_NORMALIZATION
    ),
    # model hyperparameters go here
) -> MyModelTrainingSummary:
    ...


def format_summary(summary: MyModelTrainingSummary) -> list[tuple[str, str]]:
    ...


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train,
    summary_items=format_summary,
    load_model=load,
    training_option_names=("smoothing",),
)
```

## Required Pieces

`_SCHEMA_TYPE`

Use a short, unique string for the JSON `model_type`. This is not required by
the registry, but it is the current single-source-of-truth convention for
checking that a loaded artifact belongs to the expected model class.

`TrainingSummary`

Use a pydantic model, normally by inheriting from `ngram.NgramTrainingSummary`
or `trigrams.TrigramTrainingSummary`. Add fields for model-specific metrics or
hyperparameters that should be printed, logged, or uploaded.

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

`train(...)`

This is the function called by `ngram.model_definition(...)`. Its required
keyword parameters are:

- `tokenizer_model: Path`
- `output_path: Path`
- `stored_tokenizer_model: Path | None = None`
- `text_normalization: normalization.TextNormalization`

It should train from `texts`, write one JSON model artifact to `output_path`,
and return the training summary. Include these common artifact fields:

- `schema_version`
- `model_type`
- tokenizer payload from `ngram.tokenizer_model_payload(...)` or
  `trigrams.standard_trigram_model_payload(...)`
- count tables or learned weights needed by the loader
- model hyperparameters needed at query/evaluation time

`format_summary(summary)`

Return a `list[tuple[str, str]]` for CLI/ClearML display. Reuse
`ngram.base_training_summary_items(...)` or
`trigrams.base_training_summary_items(...)` when possible.

`MODEL_DEFINITION`

This module variable is the actual registry hook. For n-gram models, pass:

- `module_name=__name__`
- `train_model=<training function>`
- `summary_items=<summary formatter>`
- `load_model=<loader>`
- `training_option_names=(...)` for hyperparameters consumed by the trainer
- `query_lines=...` only if the standard n-gram query display is not enough
- `evaluation_items=...` if the standard evaluation display is not enough
- `validate_training_options=...` for coupled or non-trivial option checks

The full model-training pipeline requires both query and evaluation support.
The n-gram helper supplies query support through the loaded model's `query`
method and evaluation support through the loaded model's `evaluate` method.

## Hyperparameters

If a new model uses only existing hyperparameters such as `smoothing`,
`discount`, `unigram_weight`, `bigram_weight`, `trigram_weight`, `beta_2`, or
`beta_3`, list the relevant names in `training_option_names`.

If the model needs a brand-new hyperparameter, add it consistently in:

- `src/pipelines/language_model/model_options.py`
- `src/cli/model_training.py`
- `src/cli/train.py`
- `src/cli/model_training_flow.py`
- `src/cli/model_training_defaults.py`
- `src/pipelines/language_model/artifacts.py` when it should be logged
- `src/pipelines/language_model/optuna.py` when it should be searchable

Then include the new name in the model module's `training_option_names`.

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

For a model that does not fit `ngram.model_definition(...)`, create
`MODEL_DEFINITION` directly:

```python
from src.ml_core.models import definition as model_def


MODEL_DEFINITION = model_def.ModelDefinition(
    name="my-model",
    train=train,
    validate_options=validate_options,
    summary_items=format_summary,
    query=query,
    validate_query_options=validate_query_options,
    query_lines=format_query,
    evaluate=evaluate,
    validate_evaluation_options=validate_evaluation_options,
    evaluation_items=format_evaluation,
)
```

The callable signatures are:

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
