"""Output helpers shared by CLI entry points."""

from __future__ import annotations

import codecs
import os
import sys
import threading
from dataclasses import dataclass
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
from typing import TextIO


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ANSI_RESET = "\033[0m"
ANSI_TIMESTAMP = "\033[90m"
ANSI_COLORS = {
    "error": "\033[31m",
    "stage": "\033[1;36m",
    "warning": "\033[33m",
}
ANSI_DELTA = ANSI_COLORS["warning"]
ERROR_MARKERS = ("error", "failed", "failure", "exception", "traceback")
WARNING_MARKERS = ("warning", "warn:")


@dataclass
class LineTimingState:
    last_emission_monotonic: float | None = None


class FileDescriptorCapture:
    """Forward low-level file-descriptor writes through a timestamped writer."""

    def __init__(
        self,
        fd: int,
        writer: TimestampedLineWriter,
        *,
        encoding: str,
        errors: str,
    ) -> None:
        self._fd = fd
        self._writer = writer
        self._encoding = encoding
        self._errors = errors
        self._saved_fd: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._saved_fd = os.dup(self._fd)
        read_fd, write_fd = os.pipe()
        self._read_fd = read_fd
        os.dup2(write_fd, self._fd)
        os.close(write_fd)
        self._thread = threading.Thread(target=self._forward, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._saved_fd is None:
            return

        os.dup2(self._saved_fd, self._fd)
        os.close(self._saved_fd)
        self._saved_fd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._writer.flush()

    def _forward(self) -> None:
        assert self._read_fd is not None
        decoder = codecs.getincrementaldecoder(self._encoding)(errors=self._errors)
        try:
            while True:
                chunk = os.read(self._read_fd, 4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    self._writer.write(text)

            tail = decoder.decode(b"", final=True)
            if tail:
                self._writer.write(tail)
        finally:
            os.close(self._read_fd)
            self._read_fd = None


class TimestampedLineWriter:
    """Text stream wrapper that prepends a timestamp to each output line."""

    def __init__(
        self,
        stream: TextIO,
        *,
        default_level: str | None = None,
        timing_state: LineTimingState | None = None,
    ) -> None:
        self._stream = stream
        self._at_line_start = True
        self._default_level = default_level
        self._line_level: str | None = None
        self._enable_color = stream_supports_color(stream)
        self._timing_state = timing_state or LineTimingState()
        self._lock = threading.Lock()

    def write(self, text: object) -> int:
        output = self._coerce_text(text)
        if not output:
            return 0

        with self._lock:
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
                    self._stream.flush()
                    self._line_level = None

        return len(text) if isinstance(text, bytes | str) else len(output)

    def flush(self) -> None:
        with self._lock:
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
        delta = f"[{format_delta(self._mark_emission())}] "
        if not self._enable_color:
            self._stream.write(timestamp)
            self._stream.write(delta)
            return

        self._stream.write(ANSI_TIMESTAMP)
        self._stream.write(timestamp)
        self._stream.write(ANSI_RESET)
        self._stream.write(ANSI_DELTA)
        self._stream.write(delta)
        self._stream.write(ANSI_RESET)

    def _classify(self, text: str) -> str | None:
        lowered = text.lower()
        if any(marker in lowered for marker in WARNING_MARKERS):
            return "warning"
        if any(marker in lowered for marker in ERROR_MARKERS):
            return "error"
        return self._default_level

    def _color(self) -> str | None:
        if not self._enable_color or self._line_level is None:
            return None
        return ANSI_COLORS.get(self._line_level)

    def _mark_emission(self) -> float:
        current_monotonic = perf_counter()
        previous_monotonic = self._timing_state.last_emission_monotonic
        self._timing_state.last_emission_monotonic = current_monotonic
        if previous_monotonic is None:
            return 0.0
        return max(0.0, current_monotonic - previous_monotonic)


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
    return highlight_stage_title(f"Stage {index}/{total} - {title}:")


def highlight_stage_title(text: str) -> str:
    if not stream_supports_color(sys.stdout):
        return text
    return f"{ANSI_COLORS['stage']}{text}{ANSI_RESET}"


def format_delta(delta_seconds: float) -> str:
    return f"+{delta_seconds:.3f}s"


def prepare_terminal_colors() -> None:
    if os.name != "nt":
        return

    try:
        from colorama import just_fix_windows_console
    except ImportError:
        return
    just_fix_windows_console()


def duplicate_stream(stream: TextIO) -> TextIO:
    duplicate_fd = os.dup(stream.fileno())
    return os.fdopen(
        duplicate_fd,
        mode="w",
        buffering=1,
        encoding=stream.encoding or "utf-8",
        errors=getattr(stream, "errors", None) or "replace",
        closefd=True,
    )


@contextmanager
def timestamped_cli_output() -> Iterator[None]:
    prepare_terminal_colors()
    timing_state = LineTimingState()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_sink = duplicate_stream(original_stdout)
    stderr_sink = duplicate_stream(original_stderr)
    stdout = TimestampedLineWriter(stdout_sink, timing_state=timing_state)
    stderr = TimestampedLineWriter(
        stderr_sink,
        default_level="error",
        timing_state=timing_state,
    )
    stdout_capture = FileDescriptorCapture(
        original_stdout.fileno(),
        stdout,
        encoding=original_stdout.encoding or "utf-8",
        errors=getattr(original_stdout, "errors", None) or "replace",
    )
    stderr_capture = FileDescriptorCapture(
        original_stderr.fileno(),
        stderr,
        encoding=original_stderr.encoding or "utf-8",
        errors=getattr(original_stderr, "errors", None) or "replace",
    )

    try:
        stdout_capture.start()
        stderr_capture.start()
        sys.stdout = stdout  # type: ignore[assignment]
        sys.stderr = stderr  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_capture.stop()
        stderr_capture.stop()
        stdout.flush()
        stderr.flush()
        stdout_sink.close()
        stderr_sink.close()
