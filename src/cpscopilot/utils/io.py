"""I/O helpers: hashing, JSON encoding, exclusive (never-overwrite) writes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "environment_info",
    "read_json",
    "sha256_bytes",
    "sha256_file",
    "to_jsonable",
    "write_bytes_exclusive",
    "write_json_exclusive",
    "write_text_exclusive",
]

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "matplotlib",
    "pydantic",
    "PyYAML",
    "anthropic",
    "torch",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas/dataclass objects into JSON-safe values.

    Non-finite floats become ``None`` so the output is strict JSON.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return str(pd.Timestamp(obj).isoformat())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict(orient="records")]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Iterable):
        return [to_jsonable(x) for x in obj]
    return str(obj)


def write_bytes_exclusive(path: Path, data: bytes) -> Path:
    """Write ``data`` to a *new* file; raise ``FileExistsError`` if it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        fh.write(data)
    return path


def write_text_exclusive(path: Path, text: str) -> Path:
    return write_bytes_exclusive(path, text.encode("utf-8"))


def write_json_exclusive(path: Path, obj: Any, indent: int = 2) -> Path:
    text = json.dumps(to_jsonable(obj), indent=indent, sort_keys=False, allow_nan=False)
    return write_text_exclusive(path, text + "\n")


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _git_info(cwd: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def environment_info(project_root: Path | None = None) -> dict[str, Any]:
    """Collect interpreter, platform, package and git provenance information."""
    packages: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "git": _git_info(project_root or Path.cwd()),
        "argv": list(sys.argv),
    }
