# language-models-experiments

Small language-model experiments around BabyLM-style corpora.

The project is intentionally local and lightweight. It is not configured as an installable Python package; use `uv run python -m ...` from the repository root.

## Setup

Use the latest stable Python line supported by the project:

```powershell
uv sync
```

The project currently requires Python 3.14 or newer.

## Layout

```text
src/
  cli/          Command-line entry points
  corpora/      Dataset loading, registry, and corpus text helpers
  models/       Small language model training utilities
  tokenizers/   Tokenizer training utilities
artifacts/      Local generated outputs, ignored by git
```

## Corpus Stats

Print simple row, character, and whitespace-token statistics:

```powershell
uv run python -m src.cli.corpus_stats --streaming
```

For quick checks:

```powershell
uv run python -m src.cli.corpus_stats --streaming --limit 1000
```

## SentencePiece Tokenizer

Train a 1000-vocabulary SentencePiece tokenizer:

```powershell
uv run python -m src.cli.train_sentencepiece --streaming --vocab-size 1000 --max-sentence-length 8192
```

Default outputs:

```text
artifacts/tokenizers/babylm-2026-strict-small-sentencepiece-1000.model
artifacts/tokenizers/babylm-2026-strict-small-sentencepiece-1000.vocab
```

## Bigram Model

Train a very simple autoregressive token bigram model from the SentencePiece tokenizer:

```powershell
uv run python -m src.cli.train --model bigram --streaming
```

Default output:

```text
artifacts/models/babylm-2026-strict-small-sentencepiece-bigram.json
```

The model stores sparse transition counts for `P(next_token | previous_token)`, plus tokenizer metadata and an add-k smoothing value. It is meant as a simple baseline, not a serious neural language model.

## Corpora

The CLI is corpus-generic. BabyLM 2026 Strict-Small is currently registered as:

```text
babylm-2026-strict-small
```

To add another corpus, add a loader module under `src/corpora/` and register a new `CorpusDefinition` in `src/corpora/registry.py`.

## Models

The model-training CLI is model-generic. `bigram` is currently registered as the first trainable model.

To add another model, add its training code under `src/models/` and register a new `ModelDefinition` in `src/models/registry.py`.

## Generated Files

Generated models, tokenizer files, and other experiment outputs belong under `artifacts/`. That directory is ignored by git.
