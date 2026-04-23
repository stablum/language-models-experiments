"""Output helpers shared by CLI entry points."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import TextIO


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ANSI_RESET = "\033[0m"
ANSI_TIMESTAMP = "\033[90m"
ANSI_COLORS = {
    "error": "\033[31m",
    "stage": "\033[1;36m",
    "warning": "\033[33m",
}
ERROR_MARKERS = ("error", "failed", "failure", "exception", "traceback")
STAGE_TITLE_MARKERS = (
    "corpus stats:",
    "evaluation:",
    "model training:",
    "query:",
    "tokenizer training:",
)
WARNING_MARKERS = ("warning", "warn:")


class TimestampedLineWriter:
    """Text stream wrapper that prepends a timestamp to each output line."""

    def __init__(self, stream: TextIO, *, default_level: str | None = None) -> None:
        self._stream = stream
        self._at_line_start = True
        self._default_level = default_level
        self._line_level: str | None = None
        self._enable_color = stream_supports_color(stream)

    def write(self, text: object) -> int:
        output = self._coerce_text(text)
        if not output:
            return 0

        for segment in output.splitlines(keepends=True):
            if self._at_line_start:
                self._line_level = self._classify(segment)
                self._write_timestamp()

            color = self._color()
            if color:
                self._stream.write(color)

            self._stream.write(segment)
            if color:
                self._stream.write(ANSI_RESET)

            self._at_line_start = segment.endswith(("\n", "\r"))
            if self._at_line_start:
                self.flush()
                self._line_level = None

        return len(text) if isinstance(text, bytes | str) else len(output)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)

    def _coerce_text(self, text: object) -> str:
        if isinstance(text, bytes):
            encoding = self._stream.encoding or "utf-8"
            return text.decode(encoding, errors="replace")
        return str(text)

    def _write_timestamp(self) -> None:
        timestamp = f"[{datetime.now().strftime(TIMESTAMP_FORMAT)}] "
        if not self._enable_color:
            self._stream.write(timestamp)
            return

        self._stream.write(ANSI_TIMESTAMP)
        self._stream.write(timestamp)
        self._stream.write(ANSI_RESET)

    def _classify(self, text: str) -> str | None:
        lowered = text.lower()
        stripped = lowered.strip()
        if stripped.startswith("stage ") or stripped in STAGE_TITLE_MARKERS:
            return "stage"
        if any(marker in lowered for marker in WARNING_MARKERS):
            return "warning"
        if any(marker in lowered for marker in ERROR_MARKERS):
            return "error"
        return self._default_level

    def _color(self) -> str | None:
        if not self._enable_color or self._line_level is None:
            return None
        return ANSI_COLORS.get(self._line_level)


def stream_supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    color_policy = os.environ.get("LME_COLOR", "always").lower()
    if color_policy in {"0", "false", "never", "no", "off"}:
        return False
    if color_policy in {"1", "always", "force", "true", "yes", "on"}:
        return True
    try:
        return stream.isatty()
    except OSError:
        return False


def stage_title(index: int, total: int, title: str) -> str:
    return f"Stage {index}/{total} - {title}:"


def prepare_terminal_colors() -> None:
    if os.name != "nt":
        return

    try:
        from colorama import just_fix_windows_console
    except ImportError:
        return
    just_fix_windows_console()


@contextmanager
def timestamped_cli_output() -> Iterator[None]:
    prepare_terminal_colors()
    stdout = TimestampedLineWriter(sys.stdout)
    stderr = TimestampedLineWriter(sys.stderr, default_level="error")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = stdout  # type: ignore[assignment]
    sys.stderr = stderr  # type: ignore[assignment]
    try:
        yield
    finally:
        stdout.flush()
        stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
