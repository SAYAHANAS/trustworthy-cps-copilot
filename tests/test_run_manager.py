"""Immutable, hashed run directories."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from cpscopilot.config import load_config
from cpscopilot.utils.run_manager import create_run, verify_run


def test_runs_never_share_a_directory(tmp_path: Path) -> None:
    cfg = load_config(None, ["name=demo"])
    dirs = {create_run(cfg, tmp_path).run_dir for _ in range(3)}
    assert len(dirs) == 3
    assert all("demo_s42_" in d.name for d in dirs)


def test_files_are_written_exclusively(tmp_path: Path) -> None:
    ctx = create_run(load_config(None), tmp_path)
    ctx.save_json("metrics.json", {"a": 1})
    with pytest.raises(FileExistsError):
        ctx.save_json("metrics.json", {"a": 2})
    ctx.save_frame("table.csv.gz", pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(FileExistsError):
        ctx.save_frame("table.csv.gz", pd.DataFrame({"x": [3]}))


def test_finalize_writes_manifest_and_locks_files(tmp_path: Path) -> None:
    ctx = create_run(load_config(None), tmp_path)
    ctx.save_json("metrics.json", {"a": 1, "nan": float("nan")})
    manifest = ctx.finalize()
    assert "metrics.json" in manifest["files"]
    target = ctx.path("metrics.json")
    assert not os.stat(target).st_mode & stat.S_IWUSR
    assert verify_run(ctx.run_dir)["ok"]
    with pytest.raises(RuntimeError):
        ctx.save_json("late.json", {})
    os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)  # simulate tampering
    target.write_text('{"a": 999}')
    report = verify_run(ctx.run_dir)
    assert not report["ok"] and report["modified"] == ["metrics.json"]
