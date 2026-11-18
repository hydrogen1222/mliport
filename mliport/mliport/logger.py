"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Logging utilities for mliport.

One append-only ``run.log`` writer for every run (``LiveRunLogger``), with
buffered high-frequency records and a follow-command helper.
"""

from __future__ import annotations

import os
import shlex
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def follow_log_command(log_path: Path | str) -> str:
    """Return a copy-paste command for following a live log."""
    resolved = Path(log_path).resolve()
    if os.name == "nt":
        escaped = str(resolved).replace("'", "''")
        return f"Get-Content -Wait -Tail 20 '{escaped}'"
    return f"tail -f {shlex.quote(str(resolved))}"


class LiveRunLogger:
    """Thread-safe run log with buffered high-frequency record support.

    Ordinary status messages are flushed immediately and forwarded to the
    active interface. High-frequency records (for example one line per MD
    step) can use :meth:`write_buffered`: every record is retained in
    ``run.log``, while filesystem flushes are coalesced and the UI callback is
    not flooded.
    """

    BUFFERED_FLUSH_RECORDS = 100
    BUFFERED_FLUSH_SECONDS = 1.0

    def __init__(
        self,
        log_path: Path | str,
        callback: Any | None = None,
        *,
        append: bool = False,
    ) -> None:
        self.path = Path(log_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self._lock = threading.Lock()
        # ``append=True`` keeps an existing log (a resume or an extension must
        # add to the run's history, never truncate it) and still creates the
        # file on the first write, so it exists from the first line.
        # Held for the logger's lifetime and closed in close(), so a context
        # manager cannot be used here.
        self._handle = open(  # noqa: SIM115
            self.path,
            "a" if append else "w",
            encoding="utf-8",
            buffering=1024 * 1024,
        )
        self._buffered_records = 0
        self._last_flush = time.monotonic()

    def _write_locked(self, message: str, level: str) -> int:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = message.splitlines() or [""]
        for line in lines:
            self._handle.write(f"{timestamp} [{level.upper()}] {line}\n")
        return len(lines)

    def _flush_locked(self) -> None:
        self._handle.flush()
        self._buffered_records = 0
        self._last_flush = time.monotonic()

    def __call__(self, message: str, level: str = "info") -> None:
        """Write and flush a message, then forward it to the active interface."""
        text = str(message)
        with self._lock:
            self._write_locked(text, level)
            self._flush_locked()

        if self.callback is not None:
            self.callback(text, level)

    def write_buffered(self, message: str, level: str = "info") -> None:
        """Retain a high-frequency record without a per-record disk flush.

        Buffered records deliberately bypass the UI callback. They are flushed
        after a bounded number of records or elapsed wall time, and always on
        the next ordinary log message or :meth:`close`.
        """
        text = str(message)
        now = time.monotonic()
        with self._lock:
            self._buffered_records += self._write_locked(text, level)
            if (
                self._buffered_records >= self.BUFFERED_FLUSH_RECORDS
                or now - self._last_flush >= self.BUFFERED_FLUSH_SECONDS
            ):
                self._flush_locked()

    def flush(self) -> None:
        """Make all buffered records visible to readers."""
        with self._lock:
            if not self._handle.closed:
                self._flush_locked()

    def close(self) -> None:
        """Flush and close the log file."""
        with self._lock:
            if not self._handle.closed:
                self._flush_locked()
                self._handle.close()

    def __enter__(self) -> LiveRunLogger:
        return self

    def __exit__(self, exc_type, _exc_value, traceback) -> None:
        self.close()
