"""Runtime tokenizer codecs shared by language-model code."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import sentencepiece as spm
import tokenizers as hf_tok

from src.corpora import normalization
from src.ml_core import json_io


SENTENCEPIECE_ALGO = "sentencepiece"
WORDLEVEL_WHITESPACE_ALGO = "wordlevel-whitespace"
TOKENIZER_ALGOS = (SENTENCEPIECE_ALGO, WORDLEVEL_WHITESPACE_ALGO)

UNK_TOKEN = "[UNK]"  # unknown token
BOS_TOKEN = "[BOS]"  # beginning of sequence
EOS_TOKEN = "[EOS]"  # end of sequence
SPECIAL_TOKENS = (UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


class TokenizerCodec:
    """Small common surface used by n-gram models."""

    algo: ClassVar[str]

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    def id_to_piece(self, token_id: int) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def bos_id(self) -> int:
        raise NotImplementedError

    @property
    def eos_id(self) -> int:
        raise NotImplementedError

    @property
    def unk_id(self) -> int:
        raise NotImplementedError


class SentencePieceCodec(TokenizerCodec):
    algo = SENTENCEPIECE_ALGO

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.processor.decode(list(token_ids))

    def id_to_piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(token_id)

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()


class HfTokenizerCodec(TokenizerCodec):
    def __init__(self, model_path: Path, *, algo: str) -> None:
        self.model_path = model_path
        self.algo = algo
        self.tokenizer = hf_tok.Tokenizer.from_file(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def id_to_piece(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(token_id)
        return token if token is not None else f"<id:{token_id}>"

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def bos_id(self) -> int:
        return self._token_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self._token_id(EOS_TOKEN)

    @property
    def unk_id(self) -> int:
        return self._token_id(UNK_TOKEN)

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return int(token_id) if token_id is not None else -1


def load_tokenizer(
    model_path: Path,
    *,
    tokenizer_algo: str | None = None,
) -> TokenizerCodec:
    algo = tokenizer_algo or detect_tokenizer_algo(model_path)
    if algo == SENTENCEPIECE_ALGO:
        return SentencePieceCodec(model_path)
    if algo == WORDLEVEL_WHITESPACE_ALGO:
        return HfTokenizerCodec(model_path, algo=algo)
    raise ValueError(f"Unsupported tokenizer algorithm: {algo}")


def detect_tokenizer_algo(model_path: Path) -> str:
    payload = json_io.maybe_read_mapping(model_path)
    if payload is None:
        return SENTENCEPIECE_ALGO

    algo = payload.get("tokenizer_algo")
    if algo in TOKENIZER_ALGOS:
        return str(algo)

    model = payload.get("model")
    pre_tokenizer = payload.get("pre_tokenizer")
    model_type = _mapping_value(model, "type")
    pre_tokenizer_type = _mapping_value(pre_tokenizer, "type")
    if model_type == "WordLevel" and pre_tokenizer_type == "Whitespace":
        return WORDLEVEL_WHITESPACE_ALGO

    return SENTENCEPIECE_ALGO


def iter_normalized_sentences(
    texts: Iterable[str],
    *,
    text_normalization: normalization.TextNormalization = "none",
) -> Iterator[str]:
    for text in texts:
        text = normalization.normalize_text(text, text_normalization)
        for line in text.splitlines():
            sentence = line.strip()
            if sentence:
                yield sentence


def encode_prompt(
    tokenizer: TokenizerCodec,
    prompt: str,
    *,
    text_normalization: normalization.TextNormalization = "none",
) -> list[int]:
    prompt = normalization.normalize_text(prompt, text_normalization)
    if not prompt:
        return []
    return tokenizer.encode(prompt)


def decode_continuation(
    tokenizer: TokenizerCodec,
    *,
    generated_text: str,
    prompt_text: str,
    generated_token_ids: list[int],
) -> str:
    if prompt_text and generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):]
    return tokenizer.decode(generated_token_ids)


def iter_token_sequences(
    texts: Iterable[str],
    tokenizer: TokenizerCodec,
    *,
    bos_count: int,
    min_length: int,
    text_normalization: normalization.TextNormalization = "none",
) -> Iterator[list[int]]:
    for sentence in iter_normalized_sentences(
        texts,
        text_normalization=text_normalization,
    ):
        token_ids = tokenizer.encode(sentence)
        if tokenizer.bos_id >= 0:
            token_ids = [tokenizer.bos_id] * bos_count + token_ids
        if tokenizer.eos_id >= 0:
            token_ids.append(tokenizer.eos_id)

        if len(token_ids) >= min_length:
            yield token_ids


def tokenizer_payload(
    tokenizer: TokenizerCodec,
    *,
    tokenizer_model: Path,
    stored_tokenizer_model: Path | None,
    text_normalization: normalization.TextNormalization,
) -> dict[str, object]:
    return {
        "tokenizer_model": str(stored_tokenizer_model or tokenizer_model),
        "tokenizer_algo": tokenizer.algo,
        "vocab_size": tokenizer.vocab_size,
        "text_normalization": text_normalization,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "unk_id": tokenizer.unk_id,
        "pieces": [
            tokenizer.id_to_piece(token_id)
            for token_id in range(tokenizer.vocab_size)
        ],
    }


def _mapping_value(value: Any, key: str) -> str | None:
    return str(value.get(key)) if isinstance(value, dict) and key in value else None
