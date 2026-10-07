"""Moving-block bootstrap confidence intervals for time-series metrics.

Rows of a time series are autocorrelated, so the i.i.d. bootstrap understates
uncertainty. The circular moving-block bootstrap resamples contiguous blocks
of length ``L`` and keeps short-range dependence intact. Resamples in which a
metric is undefined (e.g. no attack rows) are skipped and counted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from cpscopilot.models.ensemble import moving_block_indices

__all__ = ["block_bootstrap_ci", "default_block_size"]


def default_block_size(n: int) -> int:
    """``ceil(n^(1/3))`` - a standard rate for block length - but at least 10."""
    return max(10, int(np.ceil(n ** (1.0 / 3.0))))


def block_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_resamples: int,
    block_size: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Percentile intervals for several metrics from shared resamples."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    point = {name: float(fn(y, p)) for name, fn in metrics.items()}
    if n_resamples <= 0 or n == 0:
        return {k: {"estimate": v, "lower": None, "upper": None, "std": None, "valid": 0}
                for k, v in point.items()}
    samples: dict[str, list[float]] = {k: [] for k in metrics}
    for _ in range(n_resamples):
        idx = moving_block_indices(n, block_size, rng)
        ys, ps = y[idx], p[idx]
        for name, fn in metrics.items():
            value = fn(ys, ps)
            if np.isfinite(value):
                samples[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    out: dict[str, dict[str, Any]] = {}
    for name, values in samples.items():
        arr = np.asarray(values)
        if len(arr) == 0:
            out[name] = {"estimate": point[name], "lower": None, "upper": None, "std": None,
                         "valid": 0}
            continue
        out[name] = {
            "estimate": point[name],
            "lower": float(np.quantile(arr, alpha)),
            "upper": float(np.quantile(arr, 1.0 - alpha)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "valid": int(len(arr)),
        }
    return out
