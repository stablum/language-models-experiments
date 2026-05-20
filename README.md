# language-models-experiments

Small language-model experiments around BabyLM-style corpora.

The project is intentionally lightweight, but ClearML is the experiment system of record. Tokenizer training, model training, and repeatable query exploration run as ClearML PipelineController DAGs. Local storage is reserved for corpus caches and the repo-local ClearML/Docker service state.

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
  ml_core/      Reusable ML experiment infrastructure
  models/       Small language model training utilities
  pipelines/    Pipeline DAG, task, and stage definitions
  tokenizers/   Tokenizer training utilities
.clearml/       Local ClearML Server state, ignored by git
config.toml     Repo-local CLI defaults
```

## Reusable ML Core

Language-model-specific code lives in `src/corpora/`, `src/models/`,
`src/pipelines/`, and `src/tokenizers/`. `src/cli/` is reserved for Click
entrypoints. Shared experiment infrastructure lives in `src/ml_core/` so it can
be reused by non-language-model ML projects without bringing along tokenizer,
prompt, corpus, or next-token assumptions.

Reusable pieces include:

```text
src/ml_core/cli/        TOML-backed Click defaults, timestamped output, staging dirs
src/ml_core/data/       deterministic train/validation partition plans
src/ml_core/models/     generic model registry contracts
src/ml_core/pipeline.py ClearML PipelineController construction and control
src/ml_core/pipeline_tasks.py ClearML PipelineController task lookup and status helpers
src/ml_core/tracking.py ClearML run, metric, and connectivity helpers
src/ml_core/clearml_tasks.py ClearML task, artifact, and model helpers
```

Language-model pipeline code is organized here:

```text
src/pipelines/language_model/tokenizer_training.py  tokenizer-training DAG definition and task wiring
src/pipelines/language_model/model_training.py      model-training DAG definition and task wiring
src/pipelines/language_model/definition.py          shared stage names and controller lookup
src/pipelines/language_model/*_stage.py         importable tokenizer/model/evaluation/query stage functions
src/pipelines/language_model/stage_entries.py   ClearML function-step entry wrappers
src/pipelines/language_model/artifacts.py       model staging and metric payload helpers
src/pipelines/language_model/optuna.py          Optuna trial metric helpers
```

Shared infrastructure should be added under `src/ml_core/`. Keep
domain-specific adapters, artifact names, metrics, and CLI vocabulary in the
language-model layer.

## CLI Defaults

Most omitted CLI options are read from [config.toml](config.toml). The precedence is:

```text
command line option > environment variable > config.toml > built-in default
```

The checked-in config uses the repo-local ClearML server, points the ClearML SDK at `clearml.conf`, checks ClearML endpoint connectivity before SDK task initialization, uses download/cache dataset loading by default, and leaves row limits unset for full runs. Edit `config.toml` to change everyday defaults, or point one command at another file:

```powershell
$env:LME_CONFIG_FILE = "config.smoke.toml"
uv run python -m src.cli.model_training
```

Use `model = "bigram"` in `[train]` to choose the trained model type; evaluation, query, and model training inherit that model by default. Use `tokenizer_algo` in `[tokenizer-training]` to choose the tokenizer implementation, and `tokenizer_model_name` in `[train]` to choose which reusable tokenizer model record model training consumes. Keys may be written as `snake_case` or `kebab-case`; `model` maps to the CLI `--model` option. Keep `[model-training]`, `[tokenizer-training]`, and `[query-pipeline]` for orchestration settings such as queues, run numbering, and pipeline names.

Python CLI output lines are prepended with a local timestamp and per-line delta in `[YYYY-MM-DD HH:MM:SS] [+0.237s]` format. ClearML also captures Python stdout/stderr for each task. Long-running commands print numbered stage titles such as `Stage 1/3 - Model training:` and `Stage 1/1 - Tokenizer training:`. Stage titles are bold cyan, timestamps are gray, delta times are yellow, error lines are red, and warning lines are yellow. Set `NO_COLOR=1` or `LME_COLOR=never` to disable ANSI colors. Native library stdout/stderr writes bypass timestamping by default to avoid pipe deadlocks in C/C++ extensions such as SentencePiece; set `LME_CAPTURE_NATIVE_OUTPUT=1` only when you explicitly want the old fd-level capture behavior.

## Data Splits

Registered corpus source splits are treated as input shards, not evaluation partitions. When `--source-split` is omitted, the CLIs load all available source splits and merge them into one logical row stream. They then assign rows to reusable project partitions named `train` and `validation`.

The default split is `train_ratio = 0.8`, so roughly 80% of merged rows go to `train` and 20% go to `validation`. Change it with `--train-ratio` or `train_ratio` in `config.toml`. The assignment is deterministic and randomized with `--split-seed` / `split_seed`; each row gets an independent score by hashing the seed, source split name, and row index. Rows below the train-ratio threshold go to `train`, and the rest go to `validation`.

The core split decision is equivalent to:

```text
key = split_seed + "\0" + source_split + "\0" + row_index
score = int(blake2b(key, digest_size=8)) / 2**64

if score < train_ratio:
    partition = "train"
else:
    partition = "validation"
```

`split_seed` is the configured seed, `source_split` is the upstream corpus split name such as `train` or `validation`, and `row_index` is the row's position within that source split. The resulting `score` is a stable floating-point value in `[0, 1)`, so changing `train_ratio` moves the threshold while keeping each row's score fixed.

This avoids mutable random-number-generator state entirely. Extra calls to `random` elsewhere in the program cannot shift the split assignments, and streaming the corpus does not require a precomputed list of row IDs. The split artifact stores the recipe, not the corpus rows or membership lists, so reproducibility depends on the upstream dataset ID, source split names, row order, ratio, seed, and split algorithm staying stable.

Every tokenizer, training, evaluation, and pipeline stage task logs a split ID and uploads a `data-split-plan-json` artifact. Model training inherits the split plan from the resolved tokenizer task, and evaluation inherits the split plan embedded in the trained model JSON. This keeps tokenizer training, model training, and validation evaluation on the same reusable partition definition.

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

## Tokenizer And Model Training

Train or refresh the reusable tokenizer first. SentencePiece is the default tokenizer algorithm:

```powershell
uv run python -m src.cli.tokenizer_training --streaming
```

Then run language-model training, evaluation, and a final query. The model-training pipeline resolves the latest completed tokenizer-training run that matches the configured `corpus` and `tokenizer_model_name`, then downloads the tokenizer from that stage task's ClearML output model.

```powershell
uv run python -m src.cli.model_training --model bigram --streaming
```

For a quick smoke test:

```powershell
uv run python -m src.cli.tokenizer_training --streaming --limit 50 --vocab-size 100 --artifact-name tinystories-sentencepiece-smoke --no-sentencepiece-hard-vocab-limit --clearml-tag smoke
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-sentencepiece-smoke --streaming --limit 50 --clearml-tag smoke
```

With the checked-in defaults, this can also be shortened to:

```powershell
uv run python -m src.cli.model_training
```

The tokenizer-training pipeline creates one stage task named:

```text
train_tokenizer
```

The model-training pipeline prints the resolved tokenizer controller/stage task IDs, then creates stage tasks named:

```text
train_model
evaluate
query
```

Each pipeline identity is the ClearML project plus its `pipeline_name` plus `pipeline_version`. The default `pipeline_version` follows the project version in `pyproject.toml`. Keep `pipeline_name` stable when you want repeated runs of the same DAG definition instead of a separate pipeline identity.

The checked-in config uses `[tokenizer-training]`, `[model-training]`, and `[query-pipeline]`; their default ClearML DAG names are `tokenizer-training`, `model-training`, and `query`.

Pipeline stage parameters use the same config sections as their stage CLIs: shared data options such as `corpus` and split settings come from `[defaults]`; tokenizer options come from `[tokenizer-training]`; model-training options, the canonical `model`, and `tokenizer_model_name` come from `[train]`; evaluation options from `[evaluate]`; query options from `[query]`; and query orchestration options from `[query-pipeline]`. Pass a pipeline CLI option to override the config for one run, or set a key in an orchestration section only when you intentionally want a pipeline-specific override.

By default, the controller and step tasks execute locally through ClearML PipelineController. To enqueue the controller and step tasks on ClearML agents, pass queues explicitly:

```powershell
uv run python -m src.cli.model_training --pipeline-queued --controller-queue services --execution-queue default
```

The controller task monitors the main non-model artifacts, and the stage tasks store model files as ClearML Models:

```text
train_tokenizer model: tokenizer .model file
train_tokenizer artifact: tokenizer-vocabulary
train_tokenizer artifact: data-split-plan-json
train_model input model: tokenizer .model file
train_model output model: trained model JSON
train_model artifact: data-split-plan-json
evaluate artifact: evaluation-summary
query artifact: query-result
query debug sample: Query / result generated-text sample
```

The `train_tokenizer` stage task is resolved from tokenizer training by `corpus` plus `tokenizer_model_name`. Among matching tokenizer-training controller runs, the newest completed run with a completed `train_tokenizer` stage and a matching output model wins. The `train_model` stage task ID is the canonical model producer for evaluation and for repeatable query pipelines; downstream stages retrieve its ClearML output model and linked tokenizer input model.

Use `--training-limit` and `--evaluation-limit` when those stages should use different row counts. These limits are applied after partitioning, so they do not create overlapping train/validation slices. Use `--source-split` only when you want to restrict the source rows before partitioning; leave it unset to merge all source splits. Use `--evaluation-partition` to choose which reusable project partition is evaluated. Use `--query-prompt`, `--query-max-tokens`, `--query-decoding`, `--query-temperature`, and `--query-seed` to control the mandatory final query.

## Tokenizers

Train a 1000-vocabulary SentencePiece tokenizer:

```powershell
uv run python -m src.cli.tokenizer_training --streaming --vocab-size 1000 --max-sentence-length 8192
```

Train a Hugging Face WordLevel tokenizer with whitespace pre-tokenization:

```powershell
uv run python -m src.cli.tokenizer_training --streaming --tokenizer-algo wordlevel-whitespace --vocab-size 1000
```

The tokenizer-training pipeline stores generated tokenizer files in ClearML and prints both the controller task ID and the tokenizer stage task ID. Downstream model training resolves the tokenizer by `corpus` and `tokenizer_model_name`.

```text
ClearML model: tokenizer model
ClearML artifact: tokenizer-vocabulary
ClearML artifact: data-split-plan-json
```

Tokenizer training uses `--text-normalization lossy-ascii` by default. This keeps the learned vocabulary English-focused and ASCII-only apart from tokenizer-internal markers. Pass `--text-normalization none` when you intentionally want the tokenizer to learn from the original Unicode text.

## N-Gram Models

Train a very simple autoregressive token bigram model from the reusable tokenizer:

```powershell
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-sentencepiece-1000 --streaming
```

The command stores the trained language model as a ClearML output model, links the tokenizer as a ClearML input model, and prints the model-training task ID.

```text
ClearML output model: trained-model-json
ClearML input model: tokenizer model
ClearML artifact: data-split-plan-json
```

The model stores readable indented JSON with sparse transition counts for `P(next_token | previous_token)`, plus tokenizer metadata, text-normalization metadata, and an add-k smoothing value. It is meant as a simple baseline, not a serious neural language model.

Train an interpolated add-k trigram model:

```powershell
uv run python -m src.cli.model_training --model trigram-add-k --tokenizer-model-name tinystories-sentencepiece-1000 --streaming
```

The add-k trigram model estimates `P(next_token | previous_previous_token, previous_token)` with linear interpolation over add-k smoothed unigram, bigram, and trigram probabilities. The default weights are `0.1 / 0.3 / 0.6`; adjust them with `--unigram-weight`, `--bigram-weight`, and `--trigram-weight`, or set recursive interpolation values with `--beta-2` and `--beta-3`. The model artifact records both lambdas and betas.

Train a Jelinek-Mercer trigram model:

```powershell
uv run python -m src.cli.model_training --model trigram-jelinek-mercer --tokenizer-model-name tinystories-sentencepiece-1000 --streaming
```

The Jelinek-Mercer trigram model uses the same fixed interpolation weights over unigram, bigram, and trigram probabilities, but those component probabilities are unsmoothed maximum-likelihood estimates. It also accepts either flat lambda weights or recursive beta values.

Train an absolute-discount trigram model:

```powershell
uv run python -m src.cli.model_training --model trigram-absolute-discount --tokenizer-model-name tinystories-sentencepiece-1000 --streaming
```

The absolute-discount trigram model subtracts a fixed discount from observed trigram counts, then backs off to an ordinary add-k smoothed bigram distribution with the reserved probability mass. The default discount is `0.75`; adjust it with `--discount`.

Train an interpolated Kneser-Ney trigram model:

```powershell
uv run python -m src.cli.model_training --model trigram-kneser-ney --tokenizer-model-name tinystories-sentencepiece-1000 --streaming
```

This is the recursive discounted/interpolated model usually called interpolated Kneser-Ney smoothing. It discounts the trigram distribution, interpolates with a lower-order Kneser-Ney bigram distribution built from continuation counts, then recursively discounts and interpolates that lower-order distribution down to a uniform base. The default discount is `0.75`; adjust it with `--discount`.

Train a Good-Turing trigram model:

```powershell
uv run python -m src.cli.model_training --model trigram-good-turing --tokenizer-model-name tinystories-sentencepiece-1000 --streaming
```

The Good-Turing model adjusts each trigram row from its count-of-counts, reserves the estimated unseen mass for unseen continuations, and recursively backs off through Good-Turing bigram and unigram distributions. It has no smoothing hyperparameter.

The model-training pipeline runs one canonical query stage after model training. Configure that query with pipeline options:

```powershell
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-sentencepiece-1000 --query-max-tokens 80 --query-seed 1
```

Condition the sample on a prompt:

```powershell
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-sentencepiece-1000 --query-prompt "Once upon" --query-max-tokens 80 --query-seed 1
```

Ask for the most probable continuation after a prompt:

```powershell
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-sentencepiece-1000 --query-prompt "Once upon" --query-decoding most-probable --query-max-tokens 80
```

The same query and evaluation commands work with `--model trigram-add-k`, `--model trigram-jelinek-mercer`, `--model trigram-absolute-discount`, `--model trigram-good-turing`, or `--model trigram-kneser-ney` after training that model.

For repeated prompt and sampling exploration, use the query CLI. It resolves the newest completed `train_model` stage matching `model`, `corpus`, and `tokenizer_model_name`, then creates a fresh single-stage `query` pipeline for each query:

```powershell
uv run python -m src.cli.query --model bigram --prompt "Once upon" --temperature 0.7 --seed 2
uv run python -m src.cli.query --model bigram --prompt "Once upon" --temperature 1.2 --seed 3
```

Use `--model-training-controller-id` to query a specific model-training run, `--model-task-id` to query a specific completed `train_model` task, or `--model-path` for a local trained model JSON. The query command normalizes prompts with the mode stored in the model file. It also prints the most likely next tokens for the prompt, with special tokens shown as labels such as `[EOS]`. ClearML stores the structured query payload as the `query-result` artifact and the generated prompt/continuation text as a `Query` / `result` Debug Samples text file. The bigram model conditions on the last prompt token; the trigram models condition on the last two prompt tokens. `--decoding most-probable` chooses the highest-probability next token at each step.

The model-training pipeline also runs evaluation on the configured partition:

```powershell
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-sentencepiece-1000 --evaluation-limit 1000
```

The evaluation stage reports next-token accuracy, top-k accuracy, average negative log-likelihood, cross-entropy, and perplexity on the `validation` partition by default. ClearML records the partition in the Data/Data split sections and prefixes evaluation metric series with the partition name, such as `validation/perplexity`. Pass `--evaluation-partition train` only when you intentionally want training-partition diagnostics.

### Optuna Hyperparameter Optimization

`src.cli.model_training` can run an Optuna study where each trial launches the full model-training pipeline. A successful trial always completes `train_model`, `evaluate`, and `query`; Optuna then reads the selected metric from the evaluation summary artifact. The default objective is `perplexity` with `minimize`.

```powershell
uv run python -m src.cli.model_training --streaming --limit 1000 --optuna-trials 10 --optuna-search smoothing=float:1e-4:1.0:log --optuna-search discount=float:0.1:0.95
```

Search specs are repeatable and use one of these forms:

```text
name=float:low:high[:log]
name=int:low:high[:step][:log]
name=categorical:value1,value2
```

Supported names are `model`, `smoothing`, `discount`, `unigram_weight`, `bigram_weight`, `trigram_weight`, `beta_2`, `beta_3`, `top_k`, `query_max_tokens`, `query_top_k`, `query_decoding`, `query_temperature`, and `query_seed`. Use `--optuna-metric next_token_accuracy --optuna-direction maximize` to optimize an accuracy metric instead of perplexity. Persistent studies can be configured with `--optuna-study-name` and `--optuna-storage`, for example `sqlite:///artifacts/optuna.db`.

The same settings can live in `config.toml`, either in `[optuna]` or as command-specific keys in `[model-training]`:

```toml
[optuna]
optuna_trials = 10
optuna_metric = "perplexity"
optuna_direction = "minimize"
optuna_search = [
  "smoothing=float:1e-4:1.0:log",
  "discount=float:0.1:0.95",
]
```

## Corpora

The CLI is corpus-generic. These corpora are currently registered:

```text
babylm-2026-strict-small
europarl
tinystories
```

The registered BabyLM corpus uses the Hugging Face dataset `BabyLM-community/BabyLM-2026-Strict-Small`, whose only known source split is `train`. The project still creates reusable `train` and `validation` partitions from that source split.

The registered TinyStories corpus uses the Hugging Face dataset `roneneldan/TinyStories`,
whose source splits are `train` and `validation`. Because source splits are input
shards in this project, omitting `--source-split` merges both TinyStories source splits
before creating reusable project partitions. Pass `--source-split train` when you want
to train and evaluate only from the original TinyStories training split.

The registered Europarl corpus uses the Hugging Face dataset `Helsinki-NLP/europarl`,
defaulting to the `en-fr` language-pair configuration and the English text path
`translation.en`. Europarl provides only a `train` source split for each language pair,
so the project creates reusable `train` and `validation` partitions from that source
split. Pass `--text-column translation.fr` to train on the French side of the default
pair.

To add another corpus, add a loader module under `src/corpora/` and register a new `CorpusDefinition` in `src/corpora/registry.py`.

## Models

The model training, query, and evaluation CLIs are model-generic. `bigram`, `trigram-add-k`, `trigram-jelinek-mercer`, `trigram-absolute-discount`, `trigram-good-turing`, and `trigram-kneser-ney` are currently registered.

See [MODELS.md](MODELS.md) for the probability formulas and brief model descriptions.

To add another model, add its code under `src/models/` and expose module-level `train`, `load`, and `format_summary` functions; the model registry discovers concrete model modules automatically.

## ClearML

ClearML is required for experiment tracking, model storage, and durable artifact storage. Start a local ClearML server:

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

For most experiments, refresh tokenizer training only when the corpus/tokenizer settings change, then run model training:

```powershell
uv run python -m src.cli.tokenizer_training --streaming
uv run python -m src.cli.model_training --model bigram --streaming
```

You can also stop and resume the same model-training PipelineController run at stage boundaries. Start the model-training DAG through model training:

```powershell
uv run python -m src.cli.model_training --run-until-stage train_model --model bigram --streaming
```

Then resume the newest eligible controller run for evaluation or run repeatable queries from the completed model stage:

```powershell
uv run python -m src.cli.train --model bigram --streaming --limit 1000
uv run python -m src.cli.evaluate --model bigram --streaming --limit 1000
uv run python -m src.cli.query --model bigram --prompt "Once upon"
```

Stage resume commands re-enqueue the controller task on the controller queue, so run a ClearML agent for queued continuation. The query command does not re-enqueue the model-training controller; it creates a new single-stage query pipeline and links it to the resolved model stage. You can disambiguate model-training runs with `--model-training-controller-id`, or use the model-training CLI directly:

```powershell
uv run python -m src.cli.model_training --run-until-stage train_model --model bigram --streaming
uv run python -m src.cli.model_training --pipeline-queued --no-wait --run-stage evaluate --model bigram --streaming
```

The pipeline, stage, and query CLIs connect options as grouped ClearML hyperparameter sections, report final metrics, upload useful non-model artifacts, and register trained tokenizer/model files as ClearML Models. Pipeline stage identity comes from the controller task plus child task names and `parent` links, not custom stage tags. Evaluation metrics in ClearML are partition-prefixed, split plans are uploaded as `data-split-plan-json`, and query stages publish generated-text samples under Debug Samples. Use `--clearml-project`, `--pipeline-name`, `--tokenizer-training-name`, `--model-training-name`, `--pipeline-version`, `--clearml-config-file`, `--clearml-output-uri`, and repeated `--clearml-tag` options to customize the pipeline runs.

### ClearML Smoke Test

On a new machine, this is the shortest end-to-end ClearML check:

```powershell
uv sync
docker compose -f docker-compose.clearml.yml up -d
Copy-Item clearml.local.conf.example clearml.conf

uv run python -m src.cli.tokenizer_training --streaming --limit 50 --vocab-size 100 --artifact-name tinystories-sentencepiece-smoke --no-sentencepiece-hard-vocab-limit --clearml-tag smoke
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-sentencepiece-smoke --streaming --limit 50 --clearml-tag smoke
```

Expected result:

```text
The ClearML tokenizer-training and model-training controller task IDs are printed in the terminal.
The ClearML UI shows completed tokenizer-training and model-training controllers tagged smoke.
The tokenizer-training pipeline has a train_tokenizer stage task; the model-training pipeline has train_model, evaluate, and query stage tasks.
The stage tasks have grouped hyperparameter sections, tokenizer/model/evaluation/query scalar metrics, tokenizer and language-model records on the Models page, data split plan artifacts, evaluation summary artifacts, query result artifacts, and query generated-text Debug Samples. Additional `src.cli.query` runs create separate query PipelineController runs with their own query parameters, `query-result` artifacts, and Debug Samples entries.
The Models page contains registered model records for the tokenizer and n-gram model files.
The uploaded files are also visible under .clearml/fileserver/.
```

To stop the local ClearML server:

```powershell
docker compose -f docker-compose.clearml.yml down
```

## Generated Files

Generated models, tokenizer files, evaluation summaries, query results, and other experiment outputs belong in ClearML. The CLIs create temporary local staging files under ignored `artifacts/staging/` while they run, then clean those files up after upload. Corpus caches and `.clearml/` Docker/ClearML server state remain local and are ignored by git.

## License

This project is licensed under the GNU General Public License v3.0. See `LICENSE`.
