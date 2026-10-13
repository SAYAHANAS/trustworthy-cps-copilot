"""Immutable experiment run directories.

Every run gets a fresh directory named
``<UTC timestamp>_<experiment>_s<seed>_<config hash[:8]>``. Files are written
with exclusive-create semantics (``open(..., "x")``), so nothing is ever
overwritten. :meth:`RunContext.finalize` writes a SHA-256 manifest of every
artifact and marks the files read-only; :func:`verify_run` re-checks a run
against its manifest.
"""

from __future__ import annotations

import io
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cpscopilot.config import ExperimentConfig, config_hash
from cpscopilot.utils.io import (
    read_json,
    sha256_file,
    to_jsonable,
    write_bytes_exclusive,
    write_json_exclusive,
    write_text_exclusive,
)

__all__ = ["RunContext", "create_run", "slugify", "utc_stamp", "verify_run"]

MANIFEST = "manifest.json"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug or "run"


@dataclass
class RunContext:
    """Handle on one run directory with never-overwrite write helpers."""

    run_id: str
    run_dir: Path
    config: ExperimentConfig
    config_hash: str
    full_hash: str
    started_at: str
    finalized: bool = field(default=False, init=False)

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def _guard(self) -> None:
        if self.finalized:
            raise RuntimeError(f"run {self.run_id} is finalized and read-only")

    def save_json(self, name: str, obj: Any) -> Path:
        self._guard()
        return write_json_exclusive(self.path(name), obj)

    def save_yaml(self, name: str, obj: Any) -> Path:
        self._guard()
        text = yaml.safe_dump(to_jsonable(obj), sort_keys=False, allow_unicode=True)
        return write_text_exclusive(self.path(name), text)

    def save_text(self, name: str, text: str) -> Path:
        self._guard()
        return write_text_exclusive(self.path(name), text)

    def save_jsonl(self, name: str, records: list[Any]) -> Path:
        import json

        self._guard()
        lines = [json.dumps(to_jsonable(r), allow_nan=False) for r in records]
        return write_text_exclusive(self.path(name), "\n".join(lines) + ("\n" if lines else ""))

    def save_frame(self, name: str, frame: pd.DataFrame) -> Path:
        """Save a DataFrame as (optionally gzip-compressed) CSV."""
        self._guard()
        # mtime=0 keeps gzip output byte-identical across reruns.
        compression: Any = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        buffer = io.BytesIO()
        frame.to_csv(buffer, index=False, compression=compression)
        return write_bytes_exclusive(self.path(name), buffer.getvalue())

    def save_arrays(self, name: str, **arrays: np.ndarray) -> Path:
        self._guard()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return write_bytes_exclusive(self.path(name), buffer.getvalue())

    def save_bytes(self, name: str, data: bytes) -> Path:
        self._guard()
        return write_bytes_exclusive(self.path(name), data)

    def finalize(self, make_read_only: bool = True) -> dict[str, Any]:
        """Write the SHA-256 manifest and (optionally) make all files read-only."""
        self._guard()
        files = sorted(
            p for p in self.run_dir.rglob("*") if p.is_file() and p.name != MANIFEST
        )
        manifest = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "full_hash": self.full_hash,
            "started_at": self.started_at,
            "finalized_at": datetime.now(UTC).isoformat(),
            "files": {
                str(p.relative_to(self.run_dir)): {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in files
            },
        }
        write_json_exclusive(self.path(MANIFEST), manifest)
        if make_read_only:
            for p in [*files, self.path(MANIFEST)]:
                mode = p.stat().st_mode
                os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        self.finalized = True
        return manifest


def create_run(config: ExperimentConfig, output_root: Path | None = None) -> RunContext:
    """Create a brand-new run directory (never reuses an existing one)."""
    root = Path(output_root) if output_root else config.resolve_path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    chash = config_hash(config)
    fhash = config_hash(config, include_seed=True)
    stamp = utc_stamp()
    base = f"{stamp}_{slugify(config.name)}_s{config.seed}_{chash[:8]}"
    for attempt in range(1000):
        run_id = base if attempt == 0 else f"{base}_{attempt}"
        run_dir = root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return RunContext(
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            config_hash=chash,
            full_hash=fhash,
            started_at=datetime.now(UTC).isoformat(),
        )
    raise RuntimeError(f"could not allocate a unique run directory under {root}")


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Re-hash every file listed in a run's manifest.

    Returns a report with ``ok`` plus lists of modified, missing and untracked files.
    """
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / MANIFEST)
    modified, missing = [], []
    for rel, info in manifest["files"].items():
        p = run_dir / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != info["sha256"]:
            modified.append(rel)
    tracked = set(manifest["files"]) | {MANIFEST}
    untracked = sorted(
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*")
        if p.is_file() and str(p.relative_to(run_dir)) not in tracked
    )
    return {
        "run_id": manifest["run_id"],
        "ok": not (modified or missing),
        "modified": modified,
        "missing": missing,
        "untracked": untracked,
    }
