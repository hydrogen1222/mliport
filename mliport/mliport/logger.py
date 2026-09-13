"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Logging utilities for mliport.

Provides structured logging for all calculations with support for
file output and console output.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any


def setup_logger(
    name: str = "mliport",
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    console: bool = True,
) -> Logger:
    """Setup a logger with file and/or console output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        console: Whether to output to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class CalculationLogger:
    """Logger for calculation runs.

    Provides structured logging for a single calculation run
    with timing and progress tracking.
    """

    def __init__(
        self,
        name: str,
        output_dir: Path | str,
        verbose: bool = True,
    ):
        """Initialize calculation logger.

        Args:
            name: Calculation name
            output_dir: Output directory for logs
            verbose: Whether to print to console
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.output_dir / "calculation.log"
        self.logger = setup_logger(
            name=f"mliport.{name}",
            log_file=log_file,
            console=verbose,
        )

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


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
    ) -> None:
        self.path = Path(log_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self._lock = threading.Lock()
        self._handle = open(
            self.path,
            "w",
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
