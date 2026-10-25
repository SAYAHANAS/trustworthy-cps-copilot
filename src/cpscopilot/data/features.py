"""Causal rolling-window feature engineering.

Every feature at time ``t`` is a function of observations at times ``<= t``
only:

* raw values are forward-filled (a causal imputation: it only copies the past),
* differences use ``x[t] - x[t - lag]``,
* rolling statistics use trailing windows ``[t - w + 1, t]`` (``center=False``)
  and require a full window (``min_periods = w``).

Features are computed independently inside each split and inside each
contiguous segment (a new segment starts after a timestamp gap larger than
``max_gap_factor * sampling_seconds``), so no window spans a split boundary or
a data outage. The first ``warmup_steps`` rows of each segment have incomplete
history and are flagged invalid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cpscopilot.config import FeatureConfig
from cpscopilot.data.schema import TIMESTAMP

__all__ = ["CausalWindowFeaturizer", "base_feature_name", "segment_ids"]

SEP = "__"


def base_feature_name(feature: str) -> str:
    """``"LIT101__mean_w30"`` -> ``"LIT101"``."""
    return feature.split(SEP, 1)[0]


def segment_ids(timestamps: pd.Series, sampling_seconds: float,
                max_gap_factor: float) -> np.ndarray:
    """Integer id of the contiguous segment each row belongs to."""
    deltas = timestamps.diff().dt.total_seconds().to_numpy()
    breaks = np.nan_to_num(deltas, nan=0.0) > max_gap_factor * sampling_seconds
    return np.cumsum(breaks).astype(np.int64)


class CausalWindowFeaturizer:
    """Stateless transformer producing trailing-window features.

    It has no fitted parameters, so applying it per split cannot leak
    statistics between splits.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg

    @property
    def warmup_steps(self) -> int:
        return self.cfg.warmup_steps

    def feature_names(self, base_cols: list[str]) -> list[str]:
        names: list[str] = []
        if self.cfg.include_raw:
            names += [f"{c}{SEP}raw" for c in base_cols]
        for lag in self.cfg.diff_lags:
            names += [f"{c}{SEP}diff{lag}" for c in base_cols]
        for w in self.cfg.windows:
            for stat in self.cfg.stats:
                names += [f"{c}{SEP}{stat}_w{w}" for c in base_cols]
        return names

    def _segment(self, values: pd.DataFrame) -> list[np.ndarray]:
        x = values.ffill(limit=self.cfg.ffill_limit)
        blocks: list[np.ndarray] = []
        if self.cfg.include_raw:
            blocks.append(x.to_numpy(dtype=np.float64))
        for lag in self.cfg.diff_lags:
            blocks.append(x.diff(lag).to_numpy(dtype=np.float64))
        position = pd.Series(np.arange(len(x), dtype=np.float64), index=x.index)
        for w in self.cfg.windows:
            roll = x.rolling(window=w, min_periods=w, center=False)
            cache: dict[str, np.ndarray] = {}
            for stat in self.cfg.stats:
                if stat == "mean":
                    arr = roll.mean().to_numpy()
                elif stat == "std":
                    arr = roll.std(ddof=0).to_numpy()
                elif stat == "min":
                    arr = cache.setdefault("min", roll.min().to_numpy())
                elif stat == "max":
                    arr = cache.setdefault("max", roll.max().to_numpy())
                elif stat == "range":
                    mx = cache.setdefault("max", roll.max().to_numpy())
                    mn = cache.setdefault("min", roll.min().to_numpy())
                    arr = mx - mn
                elif stat == "slope":
                    # OLS slope of x on time inside the trailing window:
                    # cov(x, k) / var(k), with var(k) = (w^2 - 1) / 12 for k = 0..w-1.
                    mean_x = roll.mean().to_numpy()
                    mean_kx = x.mul(position, axis=0).rolling(w, min_periods=w).mean().to_numpy()
                    mean_k = position.rolling(w, min_periods=w).mean().to_numpy()[:, None]
                    arr = (mean_kx - mean_x * mean_k) / ((w * w - 1) / 12.0)
                else:  # pragma: no cover - guarded by config validation
                    raise ValueError(stat)
                blocks.append(np.asarray(arr, dtype=np.float64))
        return blocks

    def transform(
        self,
        frame: pd.DataFrame,
        base_cols: list[str],
        segments: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Compute features for one split.

        Args:
            frame: rows of a single split, sorted by time.
            base_cols: numeric input columns.
            segments: contiguous-segment id per row (``None`` = one segment).

        Returns:
            ``(features, valid)`` where ``features`` is float32 with the same
            index as ``frame`` and ``valid`` marks rows with complete history.
        """
        if TIMESTAMP in frame and not frame[TIMESTAMP].is_monotonic_increasing:
            raise ValueError("featurizer input must be sorted by time")
        segments = np.zeros(len(frame), dtype=np.int64) if segments is None else segments
        names = self.feature_names(base_cols)
        out = np.empty((len(frame), len(names)), dtype=np.float32)
        valid = np.zeros(len(frame), dtype=bool)
        values = frame[base_cols].astype(np.float64)
        boundaries = np.flatnonzero(np.diff(segments)) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, len(frame)]
        for start, stop in zip(starts, stops):
            blocks = self._segment(values.iloc[start:stop])
            out[start:stop] = np.hstack(blocks).astype(np.float32)
            valid[start + self.warmup_steps: stop] = True
        return pd.DataFrame(out, index=frame.index, columns=names), valid
