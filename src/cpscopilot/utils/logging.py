"""Logging configuration (console + per-run log file, UTC timestamps)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

__all__ = ["close_log_files", "get_logger", "setup_logging"]

_FORMAT = "%(asctime)sZ | %(levelname)-7s | %(name)s | %(message)s"
_ROOT = "cpscopilot"


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime


def setup_logging(log_file: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure the package logger.

    Handlers from a previous call are removed so repeated runs in one process
    (e.g. a multi-seed sweep) each write to their own log file.
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = _UTCFormatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def close_log_files() -> None:
    """Detach and close file handlers (call before a run directory is sealed)."""
    logger = logging.getLogger(_ROOT)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger."""
    if name.startswith(_ROOT):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT}.{name}")
