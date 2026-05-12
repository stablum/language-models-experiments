"""Definition for the tokenizer-training ClearML pipeline."""

from __future__ import annotations

from pathlib import Path

from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import stage_entries
from src.pipelines.language_model import step_config


TOKENIZER_TRAINING_PIPELINE = lm_def.PipelineDefinition(
    default_name=lm_def.DEFAULT_TOKENIZER_TRAINING_NAME,
    stages=lm_def.TOKENIZER_TRAINING_STAGES,
    stage_dependencies=lm_def.TOKENIZER_TRAINING_STAGE_DEPENDENCIES,
)


def add_pipeline_steps(
    pipeline: object,
    *,
    clearml_project: str,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    clearml_config_file: Path | None,
    execution_queue: str | None,
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
) -> None:
    cfg = step_config.StepCfg(
        pipeline_definition=TOKENIZER_TRAINING_PIPELINE,
        project_name=clearml_project,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
        config_file=clearml_config_file,
        queue=execution_queue,
    )
    cfg.add(
        pipeline,
        name=lm_def.TOKENIZER_STAGE,
        function=stage_entries.train_tokenizer_stage_entry,
        function_kwargs={
            "corpus": corpus,
            "dataset_id": dataset_id,
            "source_split": source_split,
            "text_column": text_column,
            "streaming": streaming,
            "limit": limit,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "vocab_size": vocab_size,
            "artifact_name": artifact_name,
            "tokenizer_algo": tokenizer_algo,
            "sentencepiece_model_type": sentencepiece_model_type,
            "sentencepiece_character_coverage": sentencepiece_character_coverage,
            "sentencepiece_hard_vocab_limit": sentencepiece_hard_vocab_limit,
            "sentencepiece_max_sentence_length": sentencepiece_max_sentence_length,
            "text_normalization": text_normalization,
        },
        task_type="training",
    )


__all__ = (
    "TOKENIZER_TRAINING_PIPELINE",
    "add_pipeline_steps",
)
