# language-models-experiments

Small language-model experiments around BabyLM-style corpora.

The project is intentionally local and lightweight. It is not configured as an installable Python package; use `uv run python -m ...` from the repository root.

## Setup

Use the latest stable Python line supported by the project:

```powershell
uv sync
```

The project currently requires Python 3.14 or newer. A new machine needs Python 3.14, `uv`, Docker Desktop or another Docker Engine with Compose support, and enough disk space for Hugging Face datasets plus local experiment artifacts.

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

Stats use the deliberately lossy `lossy-ascii` text normalization by default. It lowercases text, strips accents, maps common Unicode punctuation to ASCII, drops characters that still are not ASCII, and collapses whitespace. Use `--text-normalization none` to inspect the raw corpus text instead.

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

Tokenizer training uses `--text-normalization lossy-ascii` by default. This keeps the learned vocabulary English-focused and ASCII-only apart from SentencePiece's internal word-boundary marker. Pass `--text-normalization none` when you intentionally want the tokenizer to learn from the original Unicode text.

## N-Gram Models

Train a very simple autoregressive token bigram model from the SentencePiece tokenizer:

```powershell
uv run python -m src.cli.train --model bigram --streaming
```

Default output:

```text
artifacts/models/babylm-2026-strict-small-sentencepiece-bigram.json
```

The model stores readable indented JSON with sparse transition counts for `P(next_token | previous_token)`, plus tokenizer metadata, text-normalization metadata, and an add-k smoothing value. It is meant as a simple baseline, not a serious neural language model.

Train an interpolated trigram model:

```powershell
uv run python -m src.cli.train --model trigram --streaming
```

Default output:

```text
artifacts/models/babylm-2026-strict-small-sentencepiece-trigram.json
```

The trigram model estimates `P(next_token | previous_previous_token, previous_token)` with linear interpolation over add-k smoothed unigram, bigram, and trigram probabilities. The default weights are `0.1 / 0.3 / 0.6`; adjust them with `--unigram-weight`, `--bigram-weight`, and `--trigram-weight`.

Train an absolute-discount trigram model:

```powershell
uv run python -m src.cli.train --model trigram-absolute-discount --streaming
```

Default output:

```text
artifacts/models/babylm-2026-strict-small-sentencepiece-trigram-absolute-discount.json
```

The absolute-discount trigram model subtracts a fixed discount from observed trigram counts, then backs off to an ordinary add-k smoothed bigram distribution with the reserved probability mass. The default discount is `0.75`; adjust it with `--discount`.

Train an interpolated Kneser-Ney trigram model:

```powershell
uv run python -m src.cli.train --model trigram-kneser-ney --streaming
```

Default output:

```text
artifacts/models/babylm-2026-strict-small-sentencepiece-trigram-kneser-ney.json
```

This is the recursive discounted/interpolated model usually called interpolated Kneser-Ney smoothing. It discounts the trigram distribution, interpolates with a lower-order Kneser-Ney bigram distribution built from continuation counts, then recursively discounts and interpolates that lower-order distribution down to a uniform base. The default discount is `0.75`; adjust it with `--discount`.

Query a trained model and generate a short sample:

```powershell
uv run python -m src.cli.query --model bigram --max-tokens 80 --seed 1
```

Condition the sample on a prompt:

```powershell
uv run python -m src.cli.query --model bigram --prompt "Once upon" --max-tokens 80 --seed 1
```

Ask for the most probable continuation after a prompt:

```powershell
uv run python -m src.cli.query --model bigram --prompt "Once upon" --decoding most-probable --max-tokens 80
```

The same query and evaluation commands work with `--model trigram`, `--model trigram-absolute-discount`, or `--model trigram-kneser-ney` after training that model.

The query command normalizes prompts with the mode stored in the model file. It also prints the most likely next tokens for the prompt, with special tokens shown as labels such as `[EOS]`. The bigram model conditions on the last prompt token; the trigram models condition on the last two prompt tokens. `--decoding most-probable` chooses the highest-probability next token at each step.

Evaluate a trained model:

```powershell
uv run python -m src.cli.evaluate --model bigram --streaming --limit 1000
```

The evaluation command reports next-token accuracy, top-k accuracy, average negative log-likelihood, cross-entropy, and perplexity. Use `--split` when a corpus has a held-out validation or test split; evaluating on the training split is mainly a sanity check.

## Corpora

The CLI is corpus-generic. BabyLM 2026 Strict-Small is currently registered as:

```text
babylm-2026-strict-small
```

To add another corpus, add a loader module under `src/corpora/` and register a new `CorpusDefinition` in `src/corpora/registry.py`.

## Models

The model training, query, and evaluation CLIs are model-generic. `bigram`, `trigram`, `trigram-absolute-discount`, and `trigram-kneser-ney` are currently registered.

See [MODELS.md](MODELS.md) for the probability formulas and brief model descriptions.

To add another model, add its code under `src/models/` and register a new `ModelDefinition` in `src/models/registry.py`.

## ClearML

Start a local ClearML server:

```powershell
docker compose -f docker-compose.clearml.yml up -d
```

The ClearML UI runs at:

```text
http://localhost:8080
```

The API and file server are exposed at:

```text
http://localhost:8008
http://localhost:8081
```

The Docker services store their state under `.clearml/` inside this repository. That directory is ignored by git.

Create a local SDK config file:

```powershell
New-Item -ItemType Directory -Force .clearml
Copy-Item clearml.local.conf.example .clearml/clearml.conf
$env:CLEARML_CONFIG_FILE = (Resolve-Path .clearml/clearml.conf).Path
```

`clearml.local.conf.example` uses ClearML Server's public default local-development credentials. They are fine for a private laptop smoke test, but replace them with credentials from the ClearML UI before exposing the server outside your machine or trusted network. You can also run `uv run clearml-init` and paste credentials generated from the UI instead of copying the example file.

Then pass `--clearml` to register a CLI run. For example:

```powershell
uv run python -m src.cli.train_sentencepiece --streaming --limit 1000 --clearml
uv run python -m src.cli.train --model bigram --streaming --limit 1000 --clearml
uv run python -m src.cli.evaluate --model bigram --streaming --limit 1000 --clearml
uv run python -m src.cli.query --model bigram --prompt "Once upon" --clearml
```

ClearML tracking is opt-in. When enabled, the CLIs create a run, connect CLI options as hyperparameters, report final metrics, upload useful artifacts, and register trained tokenizer/model files. Use `--clearml-project`, `--clearml-task-name`, `--clearml-output-uri`, and repeated `--clearml-tag` options to customize the task.

### ClearML Smoke Test

On a new machine, this is the shortest end-to-end ClearML check:

```powershell
uv sync
docker compose -f docker-compose.clearml.yml up -d
New-Item -ItemType Directory -Force .clearml
Copy-Item clearml.local.conf.example .clearml/clearml.conf
$env:CLEARML_CONFIG_FILE = (Resolve-Path .clearml/clearml.conf).Path

uv run python -m src.cli.train_sentencepiece --streaming --limit 50 --vocab-size 100 --output-prefix artifacts/tokenizers/clearml-smoke-sentencepiece-100 --no-hard-vocab-limit --clearml --clearml-task-name "clearml smoke sentencepiece" --clearml-tag smoke --clearml-output-uri http://localhost:8081
uv run python -m src.cli.train --model bigram --streaming --limit 5 --tokenizer-model artifacts/tokenizers/clearml-smoke-sentencepiece-100.model --output artifacts/models/clearml-smoke-bigram.json --clearml --clearml-task-name "clearml smoke bigram" --clearml-tag smoke --clearml-output-uri http://localhost:8081
uv run python -m src.cli.train --model trigram --streaming --limit 5 --tokenizer-model artifacts/tokenizers/clearml-smoke-sentencepiece-100.model --output artifacts/models/clearml-smoke-trigram.json --clearml --clearml-task-name "clearml smoke trigram" --clearml-tag smoke --clearml-output-uri http://localhost:8081
```

Expected result:

```text
ClearML task pages are printed in the terminal.
The ClearML UI shows completed tasks tagged smoke.
Each training task has CLI hyperparameters, final scalar metrics, an input tokenizer artifact, and a trained model JSON artifact.
The Models page contains registered model records for the tokenizer and n-gram model files.
The uploaded files are also visible under .clearml/fileserver/.
```

To stop the local ClearML server:

```powershell
docker compose -f docker-compose.clearml.yml down
```

## Generated Files

Generated models, tokenizer files, and other experiment outputs belong under `artifacts/`. That directory is ignored by git.
