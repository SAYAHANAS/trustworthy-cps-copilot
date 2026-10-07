"""Current-state and future-horizon targets.

For a binary attack indicator ``y`` and horizon ``h`` steps:

* ``any``  : ``y_h[t] = max(y[t+1], ..., y[t+h])`` - an attack is active at
  some point in the *strictly future* window ``(t, t+h]``.
* ``onset``: defined only where ``y[t] == 0``; ``y_h[t] = 1`` iff an attack
  starts in ``(t, t+h]``. Rows already under attack are undefined (NaN) because
  they belong to the detection task.

Rows whose horizon extends beyond the end of their split/segment are NaN and
are excluded, so a label never looks across a split boundary.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["future_labels", "horizon_to_steps", "segmentwise_future_labels"]


def horizon_to_steps(minutes: float, sampling_seconds: float) -> int:
    """Convert a horizon in minutes into an integer number of rows."""
    steps = minutes * 60.0 / sampling_seconds
    rounded = int(round(steps))
    if rounded < 1 or abs(steps - rounded) > 1e-6:
        raise ValueError(
            f"horizon of {minutes} min is not a positive whole number of "
            f"{sampling_seconds}s steps ({steps:.4f})"
        )
    return rounded


def future_labels(y: np.ndarray, horizon: int, mode: Literal["any", "onset"] = "any") -> np.ndarray:
    """Future target for one contiguous sequence (float array, NaN = undefined)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.full(n, np.nan)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n <= horizon:
        return out
    # fwd[t] = max(y[t], ..., y[t + h - 1]) computed on the reversed series.
    fwd = pd.Series(y[::-1]).rolling(horizon, min_periods=horizon).max().to_numpy()[::-1]
    out[:-1] = fwd[1:]  # shift by one -> strictly future window (t, t+h]
    if mode == "onset":
        out = np.where(y == 1, np.nan, out)
    elif mode != "any":
        raise ValueError(f"unknown future label mode {mode!r}")
    return out


def segmentwise_future_labels(
    y: np.ndarray,
    segments: np.ndarray,
    horizon: int,
    mode: Literal["any", "onset"] = "any",
) -> np.ndarray:
    """Apply :func:`future_labels` independently within each contiguous segment."""
    out = np.full(len(y), np.nan)
    boundaries = np.flatnonzero(np.diff(segments)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(y)]):
        out[start:stop] = future_labels(y[start:stop], horizon, mode)
    return out
