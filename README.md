# language-models-experiments

Small language-model experiments around BabyLM-style corpora.

The project is intentionally lightweight, but ClearML is the experiment system of record. The end-to-end pipeline runs as a ClearML PipelineController DAG, and the training, evaluation, and query entrypoints create or resume PipelineController runs rather than standalone experiment tasks. Local storage is reserved for corpus caches and the repo-local ClearML/Docker service state.

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
Copy-Item clearml.local.conf.example clearml.conf
```

## Layout

```text
src/
  cli/          Command-line entry points
  corpora/      Dataset loading, registry, and corpus text helpers
  models/       Small language model training utilities
  tokenizers/   Tokenizer training utilities
.clearml/       Local ClearML Server state, ignored by git
config.toml     Repo-local CLI defaults
```

## CLI Defaults

Most omitted CLI options are read from [config.toml](config.toml). The precedence is:

```text
command line option > environment variable > config.toml > built-in default
```

The checked-in config uses the repo-local ClearML server, points the ClearML SDK at `clearml.conf`, checks ClearML endpoint connectivity before SDK task initialization, uses download/cache dataset loading by default, and leaves row limits unset for full runs. Edit `config.toml` to change everyday defaults, or point one command at another file:

```powershell
$env:LME_CONFIG_FILE = "config.smoke.toml"
uv run python -m src.cli.pipeline
```

Use `model = "bigram"` in config sections; it maps to the CLI `--model` option. Keys may be written as `snake_case` or `kebab-case`.

Python CLI output lines are prepended with a local timestamp and per-line delta in `[YYYY-MM-DD HH:MM:SS] [+0.237s]` format. ClearML also captures Python stdout/stderr for each task. Long-running commands print numbered stage titles such as `Stage 3/5 - Model training:`. Stage titles are bold cyan, timestamps are gray, delta times are yellow, error lines are red, and warning lines are yellow. Set `NO_COLOR=1` or `LME_COLOR=never` to disable ANSI colors. Native library stdout/stderr writes bypass timestamping by default to avoid pipe deadlocks in C/C++ extensions such as SentencePiece; set `LME_CAPTURE_NATIVE_OUTPUT=1` only when you explicitly want the old fd-level capture behavior.

## Data Splits

Registered corpus source splits are treated as input shards, not evaluation partitions. When `--source-split` is omitted, the CLIs load all available source splits and merge them into one logical row stream. They then assign rows to reusable project partitions named `train` and `validation`.

The default split is `train_ratio = 0.8`, so roughly 80% of merged rows go to `train` and 20% go to `validation`. Change it with `--train-ratio` or `train_ratio` in `config.toml`. The assignment is deterministic and randomized with `--split-seed` / `split_seed`; it hashes the source split name, row index, and seed, so the same dataset, ratio, and seed always produce the same partitions without storing duplicated split datasets.

Every training, evaluation, and pipeline stage task logs a split ID and uploads a `data-split-plan-json` artifact. Downstream model training inherits the split plan from the tokenizer task when `--tokenizer-task-id` is used, and evaluation inherits the split plan embedded in the trained model JSON when `--model-task-id` is used. This keeps tokenizer training, model training, and validation evaluation on the same reusable partition definition.

Evaluation defaults to the `validation` partition. To intentionally inspect training-partition metrics, pass `--evaluation-partition train`.

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

Run tokenizer training, language-model training, evaluation, and a final query as a ClearML PipelineController DAG:

```powershell
uv run python -m src.cli.pipeline --model bigram --streaming
```

For a quick smoke test:

```powershell
uv run python -m src.cli.pipeline --model bigram --streaming --limit 50 --vocab-size 100 --no-hard-vocab-limit --clearml-tag smoke
```

With the checked-in defaults, this can also be shortened to:

```powershell
uv run python -m src.cli.pipeline
```

The pipeline command prints the controller task ID and creates stage tasks named:

```text
train_tokenizer
train_model
evaluate
query
```

The pipeline identity is the ClearML project plus `pipeline_name` plus `pipeline_version`. The default `pipeline_version` follows the project version in `pyproject.toml`. Keep `pipeline_name` stable when you want repeated runs of the same DAG definition instead of a separate pipeline identity.

By default, the controller and step tasks execute locally through ClearML PipelineController. To enqueue the controller and step tasks on ClearML agents, pass queues explicitly:

```powershell
uv run python -m src.cli.pipeline --pipeline-queued --controller-queue services --execution-queue default
```

The controller task monitors the main stage artifacts, and the stage tasks store the canonical artifacts:

```text
train_tokenizer artifact: sentencepiece-model
train_tokenizer artifact: sentencepiece-vocabulary
train_model artifact: input-tokenizer-model
train_model artifact: trained-model-json
train_model artifact: data-split-plan-json
evaluate artifact: evaluation-summary
query artifact: query-result
```

The `train_model` stage task ID is the canonical model task ID for later query or re-evaluation commands. The controller task also monitors the trained model artifacts after a completed run:

```powershell
uv run python -m src.cli.query --model bigram --model-task-id <TRAIN_MODEL_STAGE_TASK_ID> --prompt "Once upon" --max-tokens 80 --seed 1
uv run python -m src.cli.evaluate --model bigram --model-task-id <TRAIN_MODEL_STAGE_TASK_ID> --streaming --limit 1000
```

Use `--tokenizer-limit`, `--training-limit`, and `--evaluation-limit` when those stages should use different row counts. These limits are applied after partitioning, so they do not create overlapping train/validation slices. Use `--source-split` only when you want to restrict the source rows before partitioning; leave it unset to merge all source splits. Use `--evaluation-partition` to choose which reusable project partition is evaluated. Use `--query-prompt`, `--query-max-tokens`, `--query-decoding`, `--query-temperature`, and `--query-seed` to control the mandatory final query.

## SentencePiece Tokenizer

Train a 1000-vocabulary SentencePiece tokenizer:

```powershell
uv run python -m src.cli.train_sentencepiece --streaming --vocab-size 1000 --max-sentence-length 8192
```

The command stores generated tokenizer files in ClearML and prints the task ID. Downstream model training uses that task ID.

```text
ClearML artifact: sentencepiece-model
ClearML artifact: sentencepiece-vocabulary
ClearML artifact: data-split-plan-json
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
ClearML artifact: data-split-plan-json
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

The evaluation command reports next-token accuracy, top-k accuracy, average negative log-likelihood, cross-entropy, and perplexity on the `validation` partition by default. ClearML records the partition in the Data/Data split sections and prefixes evaluation metric series with the partition name, such as `validation/perplexity`. Pass `--evaluation-partition train` only when you intentionally want training-partition diagnostics.

## Corpora

The CLI is corpus-generic. These corpora are currently registered:

```text
babylm-2026-strict-small
tinystories
```

The registered BabyLM corpus uses the Hugging Face dataset `BabyLM-community/BabyLM-2026-Strict-Small`, whose only known source split is `train`. The project still creates reusable `train` and `validation` partitions from that source split.

The registered TinyStories corpus uses the Hugging Face dataset `roneneldan/TinyStories`,
whose source splits are `train` and `validation`. Because source splits are input
shards in this project, omitting `--source-split` merges both TinyStories source splits
before creating reusable project partitions. Pass `--source-split train` when you want
to train and evaluate only from the original TinyStories training split.

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
Copy-Item clearml.local.conf.example clearml.conf
```

`clearml.local.conf.example` uses ClearML Server's public default local-development credentials. They are fine for a private laptop smoke test, but replace them with credentials from the ClearML UI before exposing the server outside your machine or trusted network. You can also run `uv run clearml-init` and paste credentials generated from the UI instead of copying the example file.

The default `config.toml` sets `clearml_config_file = "clearml.conf"` and `clearml_output_uri = "http://localhost:8081"`, so the experiment CLIs can run without setting `CLEARML_CONFIG_FILE` or `CLEARML_OUTPUT_URI` in every PowerShell session. It also sets `clearml_connectivity_check = true`, which makes CLIs fail fast with a clear message when the configured ClearML server is down. Use `--clearml-config-file`, `CLEARML_CONFIG_FILE`, `--no-clearml-connectivity-check`, `CLEARML_CONNECTIVITY_CHECK`, or another `LME_CONFIG_FILE` when you want different SDK or connectivity-check behavior.

For most experiments, use the end-to-end pipeline:

```powershell
uv run python -m src.cli.pipeline --model bigram --streaming
```

You can also stop and resume the same ClearML PipelineController run at stage boundaries. Start the canonical DAG through the tokenizer stage:

```powershell
uv run python -m src.cli.train_sentencepiece --streaming --limit 1000
```

Then resume the newest eligible controller run for later stages:

```powershell
uv run python -m src.cli.train --model bigram --streaming --limit 1000
uv run python -m src.cli.evaluate --model bigram --streaming --limit 1000
uv run python -m src.cli.query --model bigram --prompt "Once upon"
```

Stage resume commands re-enqueue the controller task on the controller queue, so run a ClearML agent for queued continuation. You can disambiguate with `--pipeline-controller-id`, or use the full pipeline CLI directly:

```powershell
uv run python -m src.cli.pipeline --run-until-stage train_model --model bigram --streaming
uv run python -m src.cli.pipeline --pipeline-queued --no-wait --run-stage evaluate --model bigram --streaming
```

The pipeline and stage CLIs connect options as grouped ClearML hyperparameter sections, report final metrics, upload useful artifacts, and register trained tokenizer/model files. Stage identity comes from the controller task plus child task names and `parent` links, not custom stage tags. Evaluation metrics in ClearML are partition-prefixed, and split plans are uploaded as `data-split-plan-json`. Use `--clearml-project`, `--pipeline-name`, `--pipeline-version`, `--clearml-config-file`, `--clearml-output-uri`, and repeated `--clearml-tag` options to customize the pipeline run.

### ClearML Smoke Test

On a new machine, this is the shortest end-to-end ClearML check:

```powershell
uv sync
docker compose -f docker-compose.clearml.yml up -d
Copy-Item clearml.local.conf.example clearml.conf

uv run python -m src.cli.pipeline --model bigram --streaming --limit 50 --vocab-size 100 --no-hard-vocab-limit --clearml-tag smoke
```

Expected result:

```text
The ClearML pipeline controller task ID and task page are printed in the terminal.
The ClearML UI shows a completed pipeline controller tagged smoke.
The pipeline has stage tasks named train_tokenizer, train_model, evaluate, and query.
The stage tasks have grouped hyperparameter sections, tokenizer/model/evaluation/query scalar metrics, tokenizer artifacts, a data split plan artifact, a trained model JSON artifact, evaluation summary artifacts, and query result artifacts.
The Models page contains registered model records for the tokenizer and n-gram model files.
The uploaded files are also visible under .clearml/fileserver/.
```

To stop the local ClearML server:

```powershell
docker compose -f docker-compose.clearml.yml down
```

## Generated Files

Generated models, tokenizer files, evaluation summaries, query results, and other experiment outputs belong in ClearML. The CLIs may create temporary local staging files while they run, but those files are cleaned up after upload. Corpus caches and `.clearml/` Docker/ClearML server state remain local and are ignored by git.

## License

This project is licensed under the GNU General Public License v3.0. See `LICENSE`.
