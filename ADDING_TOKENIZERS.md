# Adding A New Tokenizer

This project treats tokenizers as reusable trained artifacts. A tokenizer
algorithm has two sides:

- training code that produces a tokenizer model file and vocabulary artifact
- a runtime codec that language models can use for encoding, decoding, and
  special-token metadata

Current tokenizer algorithms live under `src/tokenizers` and are selected with
`--tokenizer-algo`.

```text
sentencepiece
wordlevel-whitespace
```

## Discovery Contract

Tokenizer algorithms are not auto-discovered. Add the algorithm name to
`src/tokenizers/core.py`, then wire its training and loading branches through
`src/tokenizers/registry.py` and `src/tokenizers/core.py`.

The stable algorithm name is stored in trained model JSON files as
`tokenizer_algo`, logged to ClearML, and used later by model training, query,
and evaluation. Choose a short kebab-case name such as `bpe-bytelevel` and keep
it stable.

Because tokenizer modules are imported during CLI startup, keep module top-level
code side-effect free. Do not train, read large artifacts, or call external
services at import time.

## Usual File Layout

For a new algorithm, start with one focused training module:

```text
src/tokenizers/my_tokenizer_training.py
```

Then update the shared registry and runtime codec code:

```text
src/tokenizers/core.py
src/tokenizers/registry.py
src/cli/tokenizer_training.py
src/pipelines/language_model/tokenizer_training.py
src/pipelines/language_model/tokenizer_stage.py
```

If the algorithm uses an existing runtime format that `HfTokenizerCodec` can
load, reuse it instead of adding a new codec class. Add a new `TokenizerCodec`
subclass only when the model file cannot be handled by SentencePiece or the
Hugging Face `tokenizers` runtime.

## Required Pieces

`src/tokenizers/core.py`

Add a constant for the algorithm name and include it in `TOKENIZER_ALGOS`.

```python
MY_TOKENIZER_ALGO = "my-tokenizer"
TOKENIZER_ALGOS = (
    SENTENCEPIECE_ALGO,
    WORDLEVEL_WHITESPACE_ALGO,
    MY_TOKENIZER_ALGO,
)
```

Make `load_tokenizer(...)` return a codec for the algorithm. The codec must
implement the `TokenizerCodec` surface:

- `encode(text) -> list[int]`
- `decode(token_ids) -> str`
- `id_to_piece(token_id) -> str`
- `vocab_size`
- `bos_id`
- `eos_id`
- `unk_id`

The n-gram models call only this shared surface. Keep algorithm-specific logic
inside the codec.

`src/tokenizers/my_tokenizer_training.py`

Expose a lean trainer function that consumes an iterable of raw texts and
returns two paths:

```python
def train_my_tokenizer(
    texts: Iterable[str],
    *,
    output_prefix: Path,
    vocab_size: int,
    text_normalization: normalization.TextNormalization = "none",
) -> tuple[Path, Path]:
    ...
```

The first returned path is the tokenizer model file. The second is the
vocabulary artifact. Use `tok_core.iter_normalized_sentences(...)` unless the
algorithm needs a different corpus stream.

`src/tokenizers/registry.py`

Add a `train_tokenizer(...)` branch that calls the trainer and returns
`TokenizerTrainingOutput`. Add a matching branch in `tokenizer_options(...)`.

Keep algorithm-specific options in the options mapping instead of widening the
trainer dispatch with many unrelated arguments. The registry is the translation
layer between CLI/pipeline option names and the concrete trainer signature.

`src/cli/tokenizer_training.py`

The `--tokenizer-algo` choices come from `tokenizer_registry.tokenizer_algo_names()`.
If the new algorithm needs new CLI options, add them here and pass them into
`tokenizer_pipeline.add_pipeline_steps(...)`.

Prefer algorithm-prefixed option names when the option is not generic:

```text
--my-tokenizer-min-frequency
--my-tokenizer-pre-tokenizer
```

`src/pipelines/language_model/tokenizer_training.py`

Add new tokenizer options to `add_pipeline_steps(...)` and pass them through to
the `train_tokenizer` stage function kwargs. This keeps ClearML pipeline runs
reproducible.

`src/pipelines/language_model/tokenizer_stage.py`

Accept the new stage kwargs, include them in the `"Tokenizer Options"` ClearML
parameter section via `tokenizer_registry.tokenizer_options(...)`, and pass them
to `train_tokenizer(...)`.

## Runtime Metadata

All trained language models embed tokenizer metadata through
`tok_core.tokenizer_payload(...)`. A codec should report accurate values for:

- `tokenizer_algo`
- `vocab_size`
- `bos_id`
- `eos_id`
- `unk_id`
- `pieces`

Use the shared special tokens when the algorithm supports them:

```text
[UNK]
[BOS]
[EOS]
```

If an algorithm does not have a BOS or EOS token, return `-1`. Existing model
helpers already treat negative special-token IDs as absent.

## Artifact Files

The tokenizer-training stage registers the tokenizer model file as a ClearML
model and uploads the vocabulary file as the `tokenizer-vocabulary` artifact.

Use predictable suffixes:

```text
output_prefix.model      SentencePiece model
output_prefix.vocab      SentencePiece vocabulary
output_prefix.json       Hugging Face tokenizer model
output_prefix.vocab.json Hugging Face vocabulary
```

For a new format, choose suffixes that make the artifact obvious and document
them in the trainer module.

## Detection

`detect_tokenizer_algo(...)` currently detects SentencePiece by fallback and
the WordLevel whitespace tokenizer from the Hugging Face JSON payload. If the
new tokenizer model is JSON, write `"tokenizer_algo": "my-tokenizer"` into the
model file or teach `detect_tokenizer_algo(...)` how to recognize it.

Prefer writing explicit metadata when the file format allows it.

## Smoke Test Checklist

After adding the algorithm, run a small tokenizer-training job:

```powershell
uv run python -m src.cli.tokenizer_training --streaming --limit 50 --vocab-size 100 --tokenizer-algo my-tokenizer --artifact-name tinystories-my-tokenizer-smoke
```

Then train a tiny model from the new tokenizer:

```powershell
uv run python -m src.cli.model_training --model bigram --tokenizer-model-name tinystories-my-tokenizer-smoke --streaming --limit 50
```

Check that:

- the tokenizer stage stores a ClearML model file
- the `tokenizer-vocabulary` artifact exists
- the model-training stage resolves the tokenizer by `corpus` and
  `tokenizer_model_name`
- query output decodes readable text
- the trained model JSON contains the expected `tokenizer_algo`

## Documentation Checklist

When a real tokenizer algorithm is added, update:

- `README.md` with the user-facing command example when the algorithm should be
  visible to normal users
- this file with any algorithm-specific caveats
- `config.toml` only when the new algorithm should become the default
