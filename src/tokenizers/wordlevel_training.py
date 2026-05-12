"""Hugging Face WordLevel tokenizer training helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import tokenizers as hf_tok
import tokenizers.models as hf_models
import tokenizers.pre_tokenizers as hf_pre_tokenizers
import tokenizers.trainers as hf_trainers

from src.corpora import normalization
from src.ml_core import json_io
from src.tokenizers import core as tok_core


def train_wordlevel_whitespace(
    texts: Iterable[str],
    *,
    output_prefix: Path,
    vocab_size: int = 1000,
    text_normalization: normalization.TextNormalization = "none",
) -> tuple[Path, Path]:
    if vocab_size < len(tok_core.SPECIAL_TOKENS):
        raise ValueError(
            "WordLevel tokenizer vocab_size must reserve room for "
            f"{len(tok_core.SPECIAL_TOKENS)} special tokens."
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = hf_tok.Tokenizer(
        hf_models.WordLevel(unk_token=tok_core.UNK_TOKEN)
    )
    tokenizer.pre_tokenizer = hf_pre_tokenizers.Whitespace()
    trainer = hf_trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=list(tok_core.SPECIAL_TOKENS),
    )
    tokenizer.train_from_iterator(
        tok_core.iter_normalized_sentences(
            texts,
            text_normalization=text_normalization,
        ),
        trainer=trainer,
    )

    model_path = output_prefix.with_suffix(".json")
    vocab_path = output_prefix.with_suffix(".vocab.json")
    tokenizer.save(str(model_path))
    write_vocab(tokenizer, vocab_path)
    return model_path, vocab_path


def write_vocab(tokenizer: hf_tok.Tokenizer, vocab_path: Path) -> None:
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    tokens = [
        {"id": token_id, "token": token}
        for token, token_id in sorted(vocab.items(), key=lambda item: item[1])
    ]
    json_io.write_json(
        vocab_path,
        {
            "tokenizer_algo": tok_core.WORDLEVEL_WHITESPACE_ALGO,
            "tokens": tokens,
        },
    )
