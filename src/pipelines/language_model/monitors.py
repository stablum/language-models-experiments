"""ClearML monitor definitions for language-model pipeline stages."""

from __future__ import annotations

from src.ml_core.data import splits as data_splits
from src.pipelines.language_model import definition as lm_def


def pipeline_artifact_monitors() -> dict[str, list[str | tuple[str, str]]]:
    return {
        lm_def.TOKENIZER_STAGE: [
            "sentencepiece-vocabulary",
            data_splits.SPLIT_PLAN_ARTIFACT,
        ],
        lm_def.MODEL_STAGE: [
            data_splits.SPLIT_PLAN_ARTIFACT,
        ],
        lm_def.EVALUATION_STAGE: [
            "evaluation-summary",
        ],
        lm_def.QUERY_STAGE: [
            "query-result",
        ],
    }


def pipeline_metric_monitors(
    evaluation_partition: str | None = None,
) -> dict[str, list[tuple[str, str]]]:
    if evaluation_partition in data_splits.PROJECT_PARTITIONS:
        evaluation_partitions = tuple(
            dict.fromkeys((evaluation_partition, *data_splits.PROJECT_PARTITIONS))
        )
    else:
        evaluation_partitions = tuple(data_splits.PROJECT_PARTITIONS)

    return {
        lm_def.TOKENIZER_STAGE: [
            ("Tokenizer training", "vocab_size"),
            ("Tokenizer training", "limit"),
        ],
        lm_def.MODEL_STAGE: [
            ("Model training", "sequence_count"),
            ("Model training", "token_count"),
            ("Model training", "transition_count"),
        ],
        lm_def.EVALUATION_STAGE: [
            ("Evaluation", f"{partition}/{metric}")
            for partition in evaluation_partitions
            for metric in (
                "next_token_accuracy",
                "top_k_accuracy",
                "perplexity",
            )
        ],
        lm_def.QUERY_STAGE: [
            ("Query", "generated_token_count"),
            ("Query", "top_next_token_probability"),
        ],
    }
