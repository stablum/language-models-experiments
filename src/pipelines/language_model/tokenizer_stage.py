"""Tokenizer-training ClearML function step."""

from __future__ import annotations

from src.corpora import registry as corpora_registry
from src.corpora import splits as corpus_splits
from src.ml_core.cli import staging
from src.ml_core.data import split_artifacts
from src.ml_core.data import splits as data_splits
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import stage_runtime
from src.tokenizers import registry as tokenizer_registry


def train_tokenizer_step(
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
    tokenizer_algo: str,
    sentencepiece_model_type: str,
    sentencepiece_character_coverage: float,
    sentencepiece_hard_vocab_limit: bool,
    sentencepiece_max_sentence_length: int | None,
    text_normalization: str,
    clearml_output_uri: str | None = None,
    clearml_tags: stage_runtime.ClearmlTags = None,
    clearml_config_file: str | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_total: int | None = None,
    pipeline_stage_title: str | None = None,
) -> str:
    """Train and publish the tokenizer step artifacts."""

    stage = lm_def.TOKENIZER_STAGE
    corpus_definition = corpora_registry.get_corpus(corpus)
    split_plan = split_artifacts.build_cli_split_plan(
        corpus_definition,
        corpus=corpus,
        dataset_id=dataset_id,
        source_split=source_split,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )

    with staging.temporary_staging_directory(
        prefix="lme-pipeline-tokenizer-"
    ) as staging_dir:
        clearml_run = stage_runtime.start_step(
            stage_runtime.StepRuntimeCfg(
                stage=stage,
                clearml_output_uri=clearml_output_uri,
                clearml_tags=clearml_tags,
                clearml_config_file=clearml_config_file,
                pipeline_stage_index=pipeline_stage_index,
                pipeline_stage_total=pipeline_stage_total,
                pipeline_stage_title=pipeline_stage_title,
            )
        )
        output_prefix = staging_dir / artifact_name
        tokenizer_options = tokenizer_registry.tokenizer_options(
            tokenizer_algo=tokenizer_algo,
            sentencepiece_model_type=sentencepiece_model_type,
            sentencepiece_character_coverage=sentencepiece_character_coverage,
            sentencepiece_hard_vocab_limit=sentencepiece_hard_vocab_limit,
            sentencepiece_max_sentence_length=sentencepiece_max_sentence_length,
        )
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.tokenizer_training",
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": dataset_id,
                    "dataset_revision": split_plan.dataset_revision or "",
                    "source_split": data_splits.source_split_label(source_split),
                    "training_partition": data_splits.TRAIN_PARTITION,
                    "text_column": text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Tokenizer": {
                    "tokenizer_algo": tokenizer_algo,
                    "vocab_size": vocab_size,
                    "artifact_name": artifact_name,
                },
                "Tokenizer Options": {
                    **tokenizer_options,
                },
                **split_artifacts.split_plan_parameter_sections(split_plan),
            }
        )

        texts = corpus_splits.load_partition_texts(
            corpus_definition,
            dataset_id=dataset_id,
            plan=split_plan,
            partition=data_splits.TRAIN_PARTITION,
            streaming=streaming,
            text_column=text_column,
            limit=limit,
        )
        tokenizer_output = tokenizer_registry.train_tokenizer(
            texts,
            tokenizer_algo=tokenizer_algo,
            output_prefix=output_prefix,
            vocab_size=vocab_size,
            tokenizer_options=tokenizer_options,
            text_normalization=text_normalization,
        )

        clearml_run.log_metrics(
            "Tokenizer training",
            {
                "vocab_size": vocab_size,
                "limit": limit,
            },
        )
        split_artifacts.upload_split_plan_artifact(
            clearml_run,
            staging_dir=staging_dir,
            plan=split_plan,
            metadata={"corpus": corpus, "stage": stage},
        )
        clearml_run.upload_artifact(
            tokenizer_registry.TOKENIZER_VOCAB_ARTIFACT,
            tokenizer_output.vocab_path,
            metadata={
                "corpus": corpus,
                "tokenizer_algo": tokenizer_algo,
                "vocab_size": vocab_size,
            },
        )
        clearml_run.register_model(
            name=artifact_name,
            model_path=tokenizer_output.model_path,
            framework="custom",
            tags=("tokenizer", corpus),
            comment=f"{tokenizer_algo} tokenizer model.",
        )
        return stage_runtime.require_task_id(clearml_run)
