"""Importable ClearML pipeline stage functions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.cli.data_splits import (
    build_cli_split_plan,
    inherited_split_plan_from_task,
    split_plan_parameter_sections,
    upload_split_plan_artifact,
)
from src.cli.evaluate import (
    evaluation_metrics_for_partition,
    evaluation_payload,
    stage_model_artifacts as stage_evaluation_model_artifacts,
)
from src.cli.output import iter_with_progress
from src.cli.query import (
    query_metrics,
    query_payload,
    stage_model_artifacts as stage_query_model_artifacts,
)
from src.cli.train import stage_tokenizer_model, training_summary_metrics
from src.corpora.registry import get_corpus
from src.corpora.splits import (
    TRAIN_PARTITION,
    attach_split_plan_to_json_model,
    count_partition_rows,
    iter_partition_rows,
    load_partition_texts,
    read_model_split_plan,
    source_split_label,
)
from src.corpora.text import iter_text_column
from src.models.registry import get_model
from src.tokenizers.sentencepiece_training import train_sentencepiece
from src.tracking.clearml import ClearMLRun, configure_clearml_config_file


TOKENIZER_STAGE = "train_tokenizer"
MODEL_STAGE = "train_model"
EVALUATION_STAGE = "evaluate"
QUERY_STAGE = "query"


def train_tokenizer_pipeline_step(
    *,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    vocab_size: int,
    artifact_name: str,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
    text_normalization: str,
    clearml_output_uri: str | None = None,
    clearml_tags: str | list[str] | tuple[str, ...] | None = None,
    clearml_config_file: str | None = None,
) -> str:
    """Train and publish the tokenizer step artifacts."""
    stage = "train_tokenizer"
    _configure_step_clearml(clearml_config_file)
    corpus_definition = get_corpus(corpus)
    split_plan = build_cli_split_plan(
        corpus_definition,
        corpus=corpus,
        dataset_id=dataset_id,
        source_split=source_split,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )

    with TemporaryDirectory(prefix="lme-pipeline-tokenizer-") as staging_root:
        staging_dir = Path(staging_root)
        output_prefix = staging_dir / artifact_name
        clearml_run = _current_step_run(
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            stage=stage,
        )
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.pipeline",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": dataset_id,
                    "source_split": source_split_label(source_split),
                    "training_partition": TRAIN_PARTITION,
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Tokenizer": {
                    "vocab_size": vocab_size,
                    "artifact_name": artifact_name,
                    "model_type": model_type,
                    "character_coverage": character_coverage,
                    "hard_vocab_limit": hard_vocab_limit,
                    "max_sentence_length": max_sentence_length,
                },
                **split_plan_parameter_sections(split_plan),
            }
        )

        texts = load_partition_texts(
            corpus_definition,
            dataset_id=dataset_id,
            plan=split_plan,
            partition=TRAIN_PARTITION,
            streaming=streaming,
            text_column=text_column,
            limit=limit,
        )
        model_path, vocab_path = train_sentencepiece(
            texts,
            output_prefix=output_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            hard_vocab_limit=hard_vocab_limit,
            max_sentence_length=max_sentence_length,
            text_normalization=text_normalization,
        )

        clearml_run.log_metrics(
            "Tokenizer training",
            {
                "vocab_size": vocab_size,
                "character_coverage": character_coverage,
                "hard_vocab_limit": hard_vocab_limit,
                "limit": limit,
            },
        )
        upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"corpus": corpus, "stage": stage},
        )
        clearml_run.upload_artifact(
            "sentencepiece-model",
            model_path,
            metadata={"corpus": corpus, "vocab_size": vocab_size},
        )
        clearml_run.upload_artifact(
            "sentencepiece-vocabulary",
            vocab_path,
            metadata={"corpus": corpus, "vocab_size": vocab_size},
        )
        clearml_run.register_model(
            name=model_path.stem,
            model_path=model_path,
            framework="custom",
            tags=("tokenizer", corpus),
            comment="SentencePiece tokenizer model.",
        )
        return _require_task_id(clearml_run)


def train_model_pipeline_step(
    *,
    tokenizer_task_id: str,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    discount: float,
    text_normalization: str,
    clearml_output_uri: str | None = None,
    clearml_tags: str | list[str] | tuple[str, ...] | None = None,
    clearml_config_file: str | None = None,
) -> str:
    """Train the language model from the tokenizer step artifact."""
    stage = "train_model"
    _configure_step_clearml(clearml_config_file)
    corpus_definition = get_corpus(corpus)
    model_definition = get_model(model_name)

    with TemporaryDirectory(prefix="lme-pipeline-model-") as staging_root:
        staging_dir = Path(staging_root)
        staged_tokenizer_model = stage_tokenizer_model(
            tokenizer_task_id=tokenizer_task_id,
            tokenizer_model=None,
            staging_dir=staging_dir,
        )
        inherited_plan = inherited_split_plan_from_task(
            task_id=tokenizer_task_id,
            staging_dir=staging_dir,
        )
        if inherited_plan is not None:
            dataset_id = inherited_plan.dataset_id
            source_split = inherited_plan.source_split
            train_ratio = inherited_plan.train_ratio
            split_seed = inherited_plan.split_seed

        split_plan = build_cli_split_plan(
            corpus_definition,
            corpus=corpus,
            dataset_id=dataset_id,
            source_split=source_split,
            train_ratio=train_ratio,
            split_seed=split_seed,
        )
        output_path = staging_dir / f"{corpus}-sentencepiece-{model_definition.name}.json"
        model_options = {
            "corpus": corpus,
            "tokenizer_model": staged_tokenizer_model,
            "output": output_path,
            "stored_tokenizer_model": Path(staged_tokenizer_model.name),
            "smoothing": smoothing,
            "unigram_weight": unigram_weight,
            "bigram_weight": bigram_weight,
            "trigram_weight": trigram_weight,
            "discount": discount,
            "text_normalization": text_normalization,
        }
        model_definition.validate_options(model_options)

        clearml_run = _current_step_run(
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            stage=stage,
        )
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.pipeline",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                    "tokenizer_task_id": tokenizer_task_id,
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": dataset_id,
                    "source_split": source_split_label(source_split),
                    "training_partition": TRAIN_PARTITION,
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Model": {
                    "model": model_definition.name,
                    "smoothing": smoothing,
                    "unigram_weight": unigram_weight,
                    "bigram_weight": bigram_weight,
                    "trigram_weight": trigram_weight,
                    "discount": discount,
                },
                "Tokenizer": {
                    "tokenizer_task_id": tokenizer_task_id,
                    "tokenizer_artifact": "sentencepiece-model",
                    "tokenizer_artifact_file": staged_tokenizer_model.name,
                },
                **split_plan_parameter_sections(split_plan),
                "Artifacts": {
                    "output_artifact_file": output_path.name,
                },
            }
        )

        texts = load_partition_texts(
            corpus_definition,
            dataset_id=dataset_id,
            plan=split_plan,
            partition=TRAIN_PARTITION,
            streaming=streaming,
            text_column=text_column,
            limit=limit,
        )
        summary = model_definition.train(texts, model_options)
        attach_split_plan_to_json_model(summary.output_path, split_plan)

        clearml_run.log_metrics("Model training", training_summary_metrics(summary))
        upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"model": model_definition.name, "corpus": corpus, "stage": stage},
        )
        clearml_run.upload_artifact(
            "input-tokenizer-model",
            summary.tokenizer_model,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "trained-model-json",
            summary.output_path,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.register_model(
            name=summary.output_path.stem,
            model_path=summary.output_path,
            framework="custom",
            tags=("language-model", model_definition.name, corpus),
            comment="Token n-gram language model JSON.",
        )
        return _require_task_id(clearml_run)


def evaluate_pipeline_step(
    *,
    model_task_id: str,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    top_k: int,
    clearml_output_uri: str | None = None,
    clearml_tags: str | list[str] | tuple[str, ...] | None = None,
    clearml_config_file: str | None = None,
) -> str:
    """Evaluate the trained model step artifact."""
    stage = "evaluate"
    _configure_step_clearml(clearml_config_file)
    corpus_definition = get_corpus(corpus)
    model_definition = get_model(model_name)
    if model_definition.evaluate is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")

    click.echo(
        f"Evaluation stage started: model={model_definition.name}, corpus={corpus}, "
        f"partition={evaluation_partition}, top_k={top_k}"
    )
    if limit is not None:
        click.echo(f"Evaluation row limit: first {limit:,} selected rows")

    with TemporaryDirectory(prefix="lme-pipeline-evaluate-") as staging_root:
        staging_dir = Path(staging_root)
        click.echo(f"Staging model artifacts from ClearML task {model_task_id}...")
        staged_model_path = stage_evaluation_model_artifacts(
            model_task_id=model_task_id,
            model_path=None,
            staging_dir=staging_dir,
        )
        click.echo(f"Staged model artifact: {staged_model_path.name}")
        inherited_plan = read_model_split_plan(staged_model_path)
        if inherited_plan is not None:
            dataset_id = inherited_plan.dataset_id
            source_split = inherited_plan.source_split
            train_ratio = inherited_plan.train_ratio
            split_seed = inherited_plan.split_seed
            click.echo(f"Using inherited data split plan: {inherited_plan.split_id}")

        split_plan = build_cli_split_plan(
            corpus_definition,
            corpus=corpus,
            dataset_id=dataset_id,
            source_split=source_split,
            train_ratio=train_ratio,
            split_seed=split_seed,
        )
        evaluation_options = {
            "corpus": corpus,
            "model_path": staged_model_path,
            "top_k": top_k,
        }
        if model_definition.validate_evaluation_options is not None:
            model_definition.validate_evaluation_options(evaluation_options)
        click.echo(
            f"Evaluation data: dataset={dataset_id}, "
            f"source_split={source_split_label(source_split)}, text_column={text_column}"
        )

        clearml_run = _current_step_run(
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            stage=stage,
        )
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.pipeline",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                    "model_task_id": model_task_id,
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": dataset_id,
                    "source_split": source_split_label(source_split),
                    "evaluation_partition": evaluation_partition,
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                },
                "Model": {
                    "model": model_definition.name,
                    "model_task_id": model_task_id,
                    "model_artifact": "trained-model-json",
                    "tokenizer_artifact": "input-tokenizer-model",
                    "model_artifact_file": staged_model_path.name,
                },
                "Evaluation": {
                    "top_k": top_k,
                },
                **split_plan_parameter_sections(split_plan),
            }
        )

        click.echo(f"Loading dataset rows for {evaluation_partition} evaluation...")
        dataset = corpus_definition.load(
            dataset_id=dataset_id,
            split=split_plan.source_split,
            streaming=streaming,
        )
        click.echo("Counting selected evaluation rows...")
        total_rows = count_partition_rows(
            dataset,
            partition=evaluation_partition,
            plan=split_plan,
            limit=limit,
        )
        if total_rows is None:
            click.echo("Evaluation row total is unknown; progress will report processed rows.")
        else:
            click.echo(f"Evaluation rows selected: {total_rows:,}")
        rows = iter_partition_rows(
            dataset,
            partition=evaluation_partition,
            plan=split_plan,
        )
        texts = iter_text_column(
            rows,
            text_column=text_column,
            limit=limit,
        )

        click.echo("Running model evaluation...")
        summary = model_definition.evaluate(
            iter_with_progress(
                texts,
                label=f"Evaluating {evaluation_partition} rows",
                total=total_rows,
                unit="rows",
            ),
            evaluation_options,
        )
        click.echo(
            f"Evaluation complete: {summary.sequence_count:,} sequences, "
            f"{summary.token_count:,} tokens, {summary.transition_count:,} transitions"
        )

        click.echo("Uploading evaluation metrics and artifacts...")
        clearml_run.log_metrics(
            "Evaluation",
            evaluation_metrics_for_partition(summary, partition=evaluation_partition),
        )
        upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"model": model_definition.name, "corpus": corpus, "stage": stage},
        )
        clearml_run.upload_artifact(
            "evaluation-summary",
            {
                **evaluation_payload(summary),
                "evaluation_partition": evaluation_partition,
                "evaluation_limit": limit,
                "data_split": split_plan.to_payload(),
            },
            metadata={
                "model": model_definition.name,
                "corpus": corpus,
                "evaluation_partition": evaluation_partition,
                "split_id": split_plan.split_id,
            },
        )
        clearml_run.upload_artifact(
            "evaluated-model",
            summary.model_path,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "tokenizer-model",
            summary.tokenizer_model,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        click.echo("Evaluation artifacts uploaded.")
        return _require_task_id(clearml_run)


def query_pipeline_step(
    *,
    model_task_id: str,
    model_name: str,
    corpus: str,
    prompt: str,
    max_tokens: int,
    top_k: int,
    decoding: str,
    temperature: float,
    seed: int | None,
    clearml_output_uri: str | None = None,
    clearml_tags: str | list[str] | tuple[str, ...] | None = None,
    clearml_config_file: str | None = None,
) -> str:
    """Query the trained model step artifact."""
    stage = "query"
    _configure_step_clearml(clearml_config_file)
    model_definition = get_model(model_name)
    if model_definition.query is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")

    with TemporaryDirectory(prefix="lme-pipeline-query-") as staging_root:
        staged_model_path = stage_query_model_artifacts(
            model_task_id=model_task_id,
            model_path=None,
            staging_dir=Path(staging_root),
        )
        query_options = {
            "corpus": corpus,
            "model_path": staged_model_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "decoding": decoding,
            "temperature": temperature,
            "seed": seed,
        }
        if model_definition.validate_query_options is not None:
            model_definition.validate_query_options(query_options)

        clearml_run = _current_step_run(
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            stage=stage,
        )
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.pipeline",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                    "model_task_id": model_task_id,
                },
                "Data": {
                    "corpus": corpus,
                },
                "Model": {
                    "model": model_definition.name,
                    "model_task_id": model_task_id,
                    "model_artifact": "trained-model-json",
                    "tokenizer_artifact": "input-tokenizer-model",
                    "model_artifact_file": staged_model_path.name,
                },
                "Query": {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "top_k": top_k,
                    "decoding": decoding,
                    "temperature": temperature,
                    "seed": seed,
                },
            }
        )

        result = model_definition.query(query_options)
        clearml_run.log_metrics("Query", query_metrics(result))
        clearml_run.upload_artifact(
            "query-result",
            query_payload(result),
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "queried-model",
            result.model_path,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "tokenizer-model",
            result.tokenizer_model,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        return _require_task_id(clearml_run)


def _configure_step_clearml(clearml_config_file: str | None) -> None:
    if clearml_config_file is not None:
        configure_clearml_config_file(Path(clearml_config_file))


def _current_step_run(
    *,
    clearml_output_uri: str | None,
    clearml_tags: str | list[str] | tuple[str, ...] | None,
    stage: str,
) -> ClearMLRun:
    try:
        from clearml import OutputModel, Task
    except ImportError as error:
        raise click.ClickException(
            "ClearML pipeline steps require the clearml Python package. "
            "Run `uv sync` before using the pipeline CLI."
        ) from error

    task = Task.current_task()
    if task is None:
        raise click.ClickException(
            "This pipeline step must run inside a ClearML task created by PipelineController."
        )

    tags = tuple(dict.fromkeys(_normalize_tags(clearml_tags)))
    if tags:
        task.add_tags(list(tags))
    return ClearMLRun(
        task=task,
        output_model_type=OutputModel,
        output_uri=clearml_output_uri,
        task_tags=tags,
    )


def _require_task_id(clearml_run: ClearMLRun) -> str:
    task_id = clearml_run.task_id
    if task_id is None:
        raise click.ClickException("ClearML step task ID is not available.")
    return task_id


def _normalize_tags(clearml_tags: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if clearml_tags is None:
        return ()
    if isinstance(clearml_tags, str):
        return tuple(tag for tag in clearml_tags.splitlines() if tag)
    return tuple(clearml_tags)


PIPELINE_STEP_HELPERS = (
    _configure_step_clearml,
    _current_step_run,
    _require_task_id,
    _normalize_tags,
)


def pipeline_artifact_monitors() -> dict[str, list[str | tuple[str, str]]]:
    return {
        TOKENIZER_STAGE: [
            "sentencepiece-model",
            "sentencepiece-vocabulary",
        ],
        MODEL_STAGE: [
            "input-tokenizer-model",
            "trained-model-json",
            "data-split-plan-json",
        ],
        EVALUATION_STAGE: [
            "evaluation-summary",
        ],
        QUERY_STAGE: [
            "query-result",
        ],
    }


def pipeline_metric_monitors(evaluation_partition: str) -> dict[str, list[tuple[str, str]]]:
    return {
        TOKENIZER_STAGE: [
            ("Tokenizer training", "vocab_size"),
            ("Tokenizer training", "limit"),
        ],
        MODEL_STAGE: [
            ("Model training", "sequence_count"),
            ("Model training", "token_count"),
            ("Model training", "transition_count"),
        ],
        EVALUATION_STAGE: [
            ("Evaluation", f"{evaluation_partition}/next_token_accuracy"),
            ("Evaluation", f"{evaluation_partition}/top_k_accuracy"),
            ("Evaluation", f"{evaluation_partition}/perplexity"),
        ],
        QUERY_STAGE: [
            ("Query", "generated_token_count"),
            ("Query", "top_next_token_probability"),
        ],
    }


def output_uri_value(clearml_output_uri: str | None) -> str | bool:
    return clearml_output_uri if clearml_output_uri is not None else True
