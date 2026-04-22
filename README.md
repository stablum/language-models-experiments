# language-models-experiments

Small language-model experiments around BabyLM-style corpora.

The project is intentionally lightweight, but ClearML is the experiment system of record. Training, evaluation, query, and corpus-stat commands create ClearML tasks and store generated experiment artifacts there. Local storage is reserved for corpus caches and the repo-local ClearML/Docker service state.

It is not configured as an installable Python package; use `uv run python -m ...` from the repository root.

## Setup

Use the latest stable Python line supported by the project:

```powershell
uv sync
```

The project currently requires Python 3.14 or newer. A new machine needs Python 3.14, `uv`, Docker Desktop or another Docker Engine with Compose support, and enough disk space for Hugging Face datasets plus repo-local ClearML server storage.

Start the repo-local ClearML server before running experiment CLIs:

```powershell
docker compose -f docker-compose.clearml.yml up -d
New-Item -ItemType Directory -Force .clearml
Copy-Item clearml.local.conf.example .clearml/clearml.conf
$env:CLEARML_CONFIG_FILE = (Resolve-Path .clearml/clearml.conf).Path
$env:CLEARML_OUTPUT_URI = "http://localhost:8081"
```

## Layout

```text
src/
  cli/          Command-line entry points
  corpora/      Dataset loading, registry, and corpus text helpers
  models/       Small language model training utilities
  tokenizers/   Tokenizer training utilities
.clearml/       Local ClearML Server state, ignored by git
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

## End-to-End Pipeline

Run tokenizer training, language-model training, and evaluation as one ClearML task:

```powershell
uv run python -m src.cli.pipeline --model bigram --streaming
```

For a quick smoke test:

```powershell
uv run python -m src.cli.pipeline --model bigram --streaming --limit 50 --vocab-size 100 --no-hard-vocab-limit --clearml-tag smoke
```

The pipeline command prints one ClearML task ID and stores:

```text
ClearML artifact: sentencepiece-model
ClearML artifact: sentencepiece-vocabulary
ClearML artifact: input-tokenizer-model
ClearML artifact: trained-model-json
ClearML artifact: evaluation-summary
ClearML artifact: pipeline-summary
```

The printed pipeline task ID can also be used as a model task ID for later query or re-evaluation commands:

```powershell
uv run python -m src.cli.query --model bigram --model-task-id <PIPELINE_TASK_ID> --prompt "Once upon" --max-tokens 80 --seed 1
uv run python -m src.cli.evaluate --model bigram --model-task-id <PIPELINE_TASK_ID> --streaming --limit 1000
```

Use `--tokenizer-limit`, `--training-limit`, and `--evaluation-limit` when those stages should use different row counts. Use `--evaluation-split` when evaluating on a different split from training.

## SentencePiece Tokenizer

Train a 1000-vocabulary SentencePiece tokenizer:

```powershell
uv run python -m src.cli.train_sentencepiece --streaming --vocab-size 1000 --max-sentence-length 8192
```

The command stores generated tokenizer files in ClearML and prints the task ID. Downstream model training uses that task ID.

```text
ClearML artifact: sentencepiece-model
ClearML artifact: sentencepiece-vocabulary
```

Tokenizer training uses `--text-normalization lossy-ascii` by default. This keeps the learned vocabulary English-focused and ASCII-only apart from SentencePiece's internal word-boundary marker. Pass `--text-normalization none` when you intentionally want the tokenizer to learn from the original Unicode text.

## N-Gram Models

Train a very simple autoregressive token bigram model from the SentencePiece tokenizer:

```powershell
uv run python -m src.cli.train --model bigram --streaming --tokenizer-task-id <TOKENIZER_TASK_ID>
```

The command stores the trained language model and tokenizer input in ClearML and prints the model-training task ID.

```text
ClearML artifact: trained-model-json
ClearML artifact: input-tokenizer-model
```

The model stores readable indented JSON with sparse transition counts for `P(next_token | previous_token)`, plus tokenizer metadata, text-normalization metadata, and an add-k smoothing value. It is meant as a simple baseline, not a serious neural language model.

Train an interpolated trigram model:

```powershell
uv run python -m src.cli.train --model trigram --streaming --tokenizer-task-id <TOKENIZER_TASK_ID>
```

The trigram model estimates `P(next_token | previous_previous_token, previous_token)` with linear interpolation over add-k smoothed unigram, bigram, and trigram probabilities. The default weights are `0.1 / 0.3 / 0.6`; adjust them with `--unigram-weight`, `--bigram-weight`, and `--trigram-weight`.

Train an absolute-discount trigram model:

```powershell
uv run python -m src.cli.train --model trigram-absolute-discount --streaming --tokenizer-task-id <TOKENIZER_TASK_ID>
```

The absolute-discount trigram model subtracts a fixed discount from observed trigram counts, then backs off to an ordinary add-k smoothed bigram distribution with the reserved probability mass. The default discount is `0.75`; adjust it with `--discount`.

Train an interpolated Kneser-Ney trigram model:

```powershell
uv run python -m src.cli.train --model trigram-kneser-ney --streaming --tokenizer-task-id <TOKENIZER_TASK_ID>
```

This is the recursive discounted/interpolated model usually called interpolated Kneser-Ney smoothing. It discounts the trigram distribution, interpolates with a lower-order Kneser-Ney bigram distribution built from continuation counts, then recursively discounts and interpolates that lower-order distribution down to a uniform base. The default discount is `0.75`; adjust it with `--discount`.

Query a trained model and generate a short sample:

```powershell
uv run python -m src.cli.query --model bigram --model-task-id <MODEL_TRAIN_TASK_ID> --max-tokens 80 --seed 1
```

Condition the sample on a prompt:

```powershell
uv run python -m src.cli.query --model bigram --model-task-id <MODEL_TRAIN_TASK_ID> --prompt "Once upon" --max-tokens 80 --seed 1
```

Ask for the most probable continuation after a prompt:

```powershell
uv run python -m src.cli.query --model bigram --model-task-id <MODEL_TRAIN_TASK_ID> --prompt "Once upon" --decoding most-probable --max-tokens 80
```

The same query and evaluation commands work with `--model trigram`, `--model trigram-absolute-discount`, or `--model trigram-kneser-ney` after training that model.

The query command normalizes prompts with the mode stored in the model file. It also prints the most likely next tokens for the prompt, with special tokens shown as labels such as `[EOS]`. The bigram model conditions on the last prompt token; the trigram models condition on the last two prompt tokens. `--decoding most-probable` chooses the highest-probability next token at each step.

Evaluate a trained model:

```powershell
uv run python -m src.cli.evaluate --model bigram --model-task-id <MODEL_TRAIN_TASK_ID> --streaming --limit 1000
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

ClearML is required for experiment tracking and durable artifact storage. Start a local ClearML server:

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
$env:CLEARML_OUTPUT_URI = "http://localhost:8081"
```

`clearml.local.conf.example` uses ClearML Server's public default local-development credentials. They are fine for a private laptop smoke test, but replace them with credentials from the ClearML UI before exposing the server outside your machine or trusted network. You can also run `uv run clearml-init` and paste credentials generated from the UI instead of copying the example file.

For most experiments, use the end-to-end pipeline:

```powershell
uv run python -m src.cli.pipeline --model bigram --streaming
```

You can also run the stages manually. Every CLI run creates a ClearML task. Use task IDs printed by earlier commands to connect the artifact flow:

```powershell
uv run python -m src.cli.train_sentencepiece --streaming --limit 1000

$tokenizerTaskId = "<task ID printed by train_sentencepiece>"
uv run python -m src.cli.train --model bigram --streaming --limit 1000 --tokenizer-task-id $tokenizerTaskId

$modelTaskId = "<task ID printed by train>"
uv run python -m src.cli.evaluate --model bigram --streaming --limit 1000 --model-task-id $modelTaskId
uv run python -m src.cli.query --model bigram --prompt "Once upon" --model-task-id $modelTaskId
```

The CLIs connect options as hyperparameters, report final metrics, upload useful artifacts, and register trained tokenizer/model files. Use `--clearml-project`, `--clearml-task-name`, `--clearml-output-uri`, and repeated `--clearml-tag` options to customize the task.

### ClearML Smoke Test

On a new machine, this is the shortest end-to-end ClearML check:

```powershell
uv sync
docker compose -f docker-compose.clearml.yml up -d
New-Item -ItemType Directory -Force .clearml
Copy-Item clearml.local.conf.example .clearml/clearml.conf
$env:CLEARML_CONFIG_FILE = (Resolve-Path .clearml/clearml.conf).Path
$env:CLEARML_OUTPUT_URI = "http://localhost:8081"

uv run python -m src.cli.pipeline --model bigram --streaming --limit 50 --vocab-size 100 --no-hard-vocab-limit --clearml-task-name "clearml smoke pipeline" --clearml-tag smoke
```

Expected result:

```text
The ClearML task ID and task page are printed in the terminal.
The ClearML UI shows a completed task tagged smoke.
The task has CLI hyperparameters, tokenizer/model/evaluation scalar metrics, tokenizer artifacts, a trained model JSON artifact, and evaluation summary artifacts.
The Models page contains registered model records for the tokenizer and n-gram model files.
The uploaded files are also visible under .clearml/fileserver/.
```

To stop the local ClearML server:

```powershell
docker compose -f docker-compose.clearml.yml down
```

## Generated Files

Generated models, tokenizer files, evaluation summaries, query results, and other experiment outputs belong in ClearML. The CLIs may create temporary local staging files while they run, but those files are cleaned up after upload. Corpus caches and `.clearml/` Docker/ClearML server state remain local and are ignored by git.
