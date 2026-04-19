"""Text normalization policies for corpus and prompt text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from typing import Literal


TextNormalization = Literal["none", "lossy-ascii"]
TEXT_NORMALIZATION_MODES: tuple[TextNormalization, ...] = ("none", "lossy-ascii")
DEFAULT_TEXT_NORMALIZATION: TextNormalization = "lossy-ascii"


LOSSY_ASCII_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u1680": " ",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u2028": " ",
        "\u2029": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
        "\u00ad": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u2035": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2036": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00b7": ".",
        "\u2022": "*",
        "\u2044": "/",
        "\u00d7": "x",
        "\u00f7": "/",
    }
)


def normalize_text(
    text: str,
    mode: TextNormalization = DEFAULT_TEXT_NORMALIZATION,
) -> str:
    if mode == "none":
        return text
    if mode == "lossy-ascii":
        return normalize_lossy_ascii(text)
    raise ValueError(f"Unknown text normalization mode: {mode}")


def normalize_texts(
    texts: Iterable[str],
    mode: TextNormalization = DEFAULT_TEXT_NORMALIZATION,
) -> Iterator[str]:
    for text in texts:
        yield normalize_text(text, mode)


def normalize_lossy_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(LOSSY_ASCII_TRANSLATION)
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = "".join(
        character
        if character in "\t\n\r" or (character.isascii() and character.isprintable())
        else " "
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()
