# Adding A New Corpus

This project keeps corpus integrations small. A corpus module knows how to load
one upstream dataset and expose the text column defaults. The shared CLIs and
pipelines handle reusable train/validation partitioning, tokenizer training,
model training, evaluation, and query.

Existing corpus modules live under `src/corpora`:

```text
src/corpora/babylm.py
src/corpora/europarl.py
src/corpora/tinystories.py
```

## Discovery Contract

Corpora are not auto-discovered. Add a loader module under `src/corpora`, then
register it in `src/corpora/registry.py`.

The registered name is the value users pass with `--corpus`. Choose a stable
kebab-case name such as `openwebtext` or `my-corpus-en`.

The registry adapts a module into `CorpusDefinition` by reading these module
constants:

- `DATASET_ID`
- `DATASET_REVISION`
- `DEFAULT_SPLIT`
- `AVAILABLE_SPLITS`
- `SPLIT_NOTE`
- `TEXT_COLUMN`
- `load_dataset(...)`

Because the registry imports corpus modules during CLI startup, keep module
top-level code side-effect free. Do not download datasets or inspect remote
metadata at import time.

## Usual Module Structure

Most Hugging Face datasets can follow this shape:

```python
"""Dataset loading for the My Corpus corpus."""

from __future__ import annotations

from typing import Any

from src.corpora import loading


DATASET_ID = "org/my-corpus"
DATASET_REVISION = "pinned-commit-sha"
DEFAULT_SPLIT = None
AVAILABLE_SPLITS = ("train",)
SPLIT_NOTE = (
    "My Corpus exposes one source split, train. The project creates reusable "
    "train/validation partitions from that source split."
)
TEXT_COLUMN = "text"


def load_dataset(
    *,
    dataset_id: str = DATASET_ID,
    revision: str | None = DATASET_REVISION,
    split: str | None = DEFAULT_SPLIT,
    streaming: bool = False,
) -> Any:
    if dataset_id != DATASET_ID and revision == DATASET_REVISION:
        revision = None
    return loading.load_hf_dataset(
        dataset_id,
        revision=revision,
        split=split,
        streaming=streaming,
    )
```

Use `loading.load_hf_dataset(...)` for normal Hugging Face datasets. If the
dataset needs a configuration, pass `config=...` as `europarl.py` does.

## Registration

Import the module in `src/corpora/registry.py` and add it to `CORPORA`:

```python
from src.corpora import babylm, europarl, my_corpus, tinystories


CORPORA = {
    name: _definition(name, module)
    for name, module in (
        ("babylm-2026-strict-small", babylm),
        ("europarl", europarl),
        ("my-corpus", my_corpus),
        ("tinystories", tinystories),
    )
}
```

The first item in `CORPORA` is the default corpus used by the CLI. Add new
corpora deliberately so the default does not change by accident.

## Source Splits And Project Partitions

Source splits are upstream dataset shards such as `train`, `validation`, or
`test`. Project partitions are the reusable partitions this repo creates:

```text
train
validation
```

When `--source-split` is omitted, the CLIs load all registered
`AVAILABLE_SPLITS` and merge them into one logical row stream. The project then
uses a deterministic hash of `split_seed`, source split, and row index to assign
each row to a reusable project partition.

Set `DEFAULT_SPLIT` to:

- `None` when the normal behavior should merge all available source splits
- a split name such as `"train"` when only one source split should be used by
  default

Set `AVAILABLE_SPLITS` to the source split names in the preferred stable order.
This order becomes part of the split-plan identity, so keep it stable.

## Text Columns

`TEXT_COLUMN` can name either a top-level field or a dotted nested path.

```text
text
translation.en
```

The shared text reader converts missing values to empty strings and raises a
clear error when the configured path does not exist. Use `--text-column` for
one-off experiments on another field, and change `TEXT_COLUMN` only when the new
default should apply to everyone.

## Dataset Revisions

Pin `DATASET_REVISION` to a commit hash when possible. The split plan includes
the revision, so pinning keeps tokenizer/model/evaluation runs reproducible.

Keep the existing override pattern:

```python
if dataset_id != DATASET_ID and revision == DATASET_REVISION:
    revision = None
```

This lets `--dataset-id` point to another compatible dataset without carrying
the original dataset's pinned revision into the override.

Use `None` only when the upstream source cannot be pinned or reproducibility is
intentionally delegated to the dataset provider.

## Corpus-Specific Loading

If a corpus is not a simple Hugging Face dataset, its `load_dataset(...)` should
still return either:

- an iterable of row mappings
- a mapping from source split names to iterable row mappings

Rows must be mappings because the shared text-column reader works from mapping
keys and dotted nested paths.

Keep special loading code inside the corpus module or pull reusable pieces into
`src/corpora/loading.py` after a second corpus needs the same behavior.

## Smoke Test Checklist

Inspect a small slice first:

```powershell
uv run python -m src.cli.corpus_stats --corpus my-corpus --streaming --limit 100
```

Then train a tiny tokenizer:

```powershell
uv run python -m src.cli.tokenizer_training --corpus my-corpus --streaming --limit 50 --vocab-size 100 --artifact-name my-corpus-tokenizer-smoke --no-sentencepiece-hard-vocab-limit
```

Check that:

- `corpus_stats` prints non-empty rows
- the reported features include the expected text column
- `--source-split` works for each registered split
- `--text-column` can override the default when useful
- the tokenizer stage uploads a `data-split-plan-json` artifact
- the split plan records the expected `dataset_id`, revision, source splits,
  train ratio, and seed

## Documentation Checklist

When a real corpus is added, also update:

- `README.md` with the registered corpus name, upstream dataset ID, source
  splits, and text column notes
- `config.toml` only when the new corpus should become the default
- this file if the corpus introduces a new loading pattern
