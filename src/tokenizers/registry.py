"""Tokenizer algorithm registry and training dispatch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from src.corpora import normalization
from src.ml_core import cfg as core_cfg
from src.tokenizers import core as tok_core
from src.tokenizers import sentencepiece_training
from src.tokenizers import wordlevel_training


DEFAULT_TOKENIZER_ALGO = tok_core.SENTENCEPIECE_ALGO
TOKENIZER_VOCAB_ARTIFACT = "tokenizer-vocabulary"


class TokenizerTrainingOutput(core_cfg.FrozenBaseCfg):
    model_path: Path
    vocab_path: Path
    tokenizer_algo: str


def tokenizer_algo_names() -> tuple[str, ...]:
    return tok_core.TOKENIZER_ALGOS


def default_artifact_name(
    *,
    corpus: str,
    tokenizer_algo: str,
    vocab_size: int,
) -> str:
    return f"{corpus}-{tokenizer_algo}-{vocab_size}"


def train_tokenizer(
    texts: Iterable[str],
    *,
    tokenizer_algo: str,
    output_prefix: Path,
    vocab_size: int,
    tokenizer_options: Mapping[str, object],
    text_normalization: normalization.TextNormalization,
) -> TokenizerTrainingOutput:
    if tokenizer_algo == tok_core.SENTENCEPIECE_ALGO:
        model_path, vocab_path = sentencepiece_training.train_sentencepiece(
            texts,
            output_prefix=output_prefix,
            vocab_size=vocab_size,
            model_type=str(tokenizer_options["model_type"]),
            character_coverage=float(tokenizer_options["character_coverage"]),
            hard_vocab_limit=bool(tokenizer_options["hard_vocab_limit"]),
            max_sentence_length=_optional_int(
                tokenizer_options.get("max_sentence_length")
            ),
            text_normalization=text_normalization,
        )
        return TokenizerTrainingOutput(
            model_path=model_path,
            vocab_path=vocab_path,
            tokenizer_algo=tokenizer_algo,
        )

    if tokenizer_algo == tok_core.WORDLEVEL_WHITESPACE_ALGO:
        model_path, vocab_path = wordlevel_training.train_wordlevel_whitespace(
            texts,
            output_prefix=output_prefix,
            vocab_size=vocab_size,
            text_normalization=text_normalization,
        )
        return TokenizerTrainingOutput(
            model_path=model_path,
            vocab_path=vocab_path,
            tokenizer_algo=tokenizer_algo,
        )

    raise ValueError(f"Unsupported tokenizer algorithm: {tokenizer_algo}")


def tokenizer_options(
    *,
    tokenizer_algo: str,
    sentencepiece_model_type: str,
    sentencepiece_character_coverage: float,
    sentencepiece_hard_vocab_limit: bool,
    sentencepiece_max_sentence_length: int | None,
) -> dict[str, object]:
    if tokenizer_algo == tok_core.SENTENCEPIECE_ALGO:
        return {
            "model_type": sentencepiece_model_type,
            "character_coverage": sentencepiece_character_coverage,
            "hard_vocab_limit": sentencepiece_hard_vocab_limit,
            "max_sentence_length": sentencepiece_max_sentence_length,
        }
    if tokenizer_algo == tok_core.WORDLEVEL_WHITESPACE_ALGO:
        return {}
    raise ValueError(f"Unsupported tokenizer algorithm: {tokenizer_algo}")


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
