"""Aggregate metrics of several runs (typically one configuration, many seeds).

Every numeric leaf of ``metrics.json`` is flattened into a dotted path and
summarised by mean, sample standard deviation and a Student-t confidence
interval across runs. Runs with different ``config_hash`` values are refused
unless ``allow_mixed=True`` because averaging different configurations is
almost always a mistake.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cpscopilot.utils.io import read_json

__all__ = ["aggregate_runs", "check_independent_runs", "flatten_numeric"]

_SKIP_PREFIXES = ("timings_seconds", "feature_importance", "dataset.", "preparation.")


def flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested dicts into ``{"a.b.c": value}`` for finite numeric leaves."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten_numeric(value, f"{prefix}{key}."))
    elif isinstance(obj, bool):
        out[prefix[:-1]] = float(obj)
    elif isinstance(obj, (int, float)) and obj is not None and math.isfinite(float(obj)):
        out[prefix[:-1]] = float(obj)
    return out


def check_independent_runs(metrics: list[dict[str, Any]], allow_duplicates: bool = False) -> None:
    """Refuse repeated runs, and repeated seeds within one configuration.

    The same run passed twice is always refused. Two runs of one configuration with
    the same seed are exact replicas that would shrink the s.d. and CI; they are
    accepted only with ``allow_duplicates`` (e.g. a determinism check). Different
    configurations may share seeds (``allow_mixed`` comparisons).
    """
    run_ids = [m["run_id"] for m in metrics]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"the same run was passed more than once: {sorted(run_ids)}")
    if allow_duplicates:
        return
    keys = [(m["config_hash"], m["seed"]) for m in metrics]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise ValueError(
            f"seeds repeat within one configuration {[(h[:12], s) for h, s in repeated]}; "
            "repeated seeds are exact replicas and would understate the across-seed variance")


def aggregate_runs(run_dirs: list[Path], confidence: float = 0.95, allow_mixed: bool = False,
                   allow_duplicates: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return ``(summary, table)`` where ``table`` has one row per metric.

    Runs must share a configuration hash (unless ``allow_mixed``) and be
    independent: distinct runs, and distinct seeds within each configuration
    (unless ``allow_duplicates``).
    """
    if not run_dirs:
        raise ValueError("no runs to aggregate")
    metrics = [read_json(Path(d) / "metrics.json") for d in run_dirs]
    hashes = sorted({m["config_hash"] for m in metrics})
    if len(hashes) > 1 and not allow_mixed:
        raise ValueError(f"runs come from different configurations: {hashes}")
    check_independent_runs(metrics, allow_duplicates)
    flat = [flatten_numeric(m) for m in metrics]
    keys = sorted(set().union(*flat))
    rows = []
    for key in keys:
        if key.startswith(_SKIP_PREFIXES) or key in ("seed",):
            continue
        values = np.array([f[key] for f in flat if key in f], dtype=np.float64)
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * std / math.sqrt(n))
        else:
            half = float("nan")
        rows.append({"metric": key, "n": n, "mean": mean, "std": std,
                     "ci_lower": mean - half, "ci_upper": mean + half,
                     "min": float(values.min()), "max": float(values.max())})
    table = pd.DataFrame(rows)
    summary = {
        "config_hashes": hashes,
        "runs": [m["run_id"] for m in metrics],
        "seeds": [m["seed"] for m in metrics],
        "confidence": confidence,
        "n_metrics": len(table),
    }
    return summary, table
