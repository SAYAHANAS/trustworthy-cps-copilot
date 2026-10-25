"""Strict chronological train / calibration / test splitting with embargo gaps.

The splitter never shuffles. Each split is one contiguous block of the
time-sorted frame and consecutive blocks are separated by ``embargo_steps``
discarded rows (purging), which

* removes autocorrelated near-duplicates at the boundaries, and
* guarantees that no rolling window or future-label horizon of one split can
  reach into the next one, even before the per-split feature/label computation
  (which already confines both to their own split).

The split *order* is train -> calibration -> test, so models are always fitted
on the past, calibrated on a later period and evaluated on the most recent one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cpscopilot.config import SplitConfig
from cpscopilot.data.schema import LABEL, TIMESTAMP

__all__ = ["SPLIT_ORDER", "SplitError", "SplitSpec", "TemporalSplit", "TemporalSplitter"]

SPLIT_ORDER = ("train", "calibration", "test")


class SplitError(ValueError):
    """Raised when a split cannot satisfy the temporal or class constraints."""


@dataclass(frozen=True)
class SplitSpec:
    """Row-position bounds ``[start, stop)`` of one split in the sorted frame."""

    name: str
    start: int
    stop: int
    start_ts: str
    end_ts: str
    n_rows: int
    n_positive: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TemporalSplit:
    """Three chronologically ordered, non-overlapping segments."""

    frames: dict[str, pd.DataFrame]
    specs: dict[str, SplitSpec]
    embargo_steps: int
    n_total: int

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.frames[name]

    def audit(self, sampling_seconds: float) -> dict[str, Any]:
        """Explicit boundary checks; every entry must be ``True`` for a valid split."""
        checks: dict[str, Any] = {}
        specs = [self.specs[n] for n in SPLIT_ORDER]
        frames = [self.frames[n] for n in SPLIT_ORDER]
        index_sets = [set(f.index) for f in frames]
        checks["no_row_overlap"] = all(
            not (index_sets[i] & index_sets[j]) for i in range(3) for j in range(i + 1, 3)
        )
        checks["position_order"] = all(
            specs[i].stop + self.embargo_steps <= specs[i + 1].start for i in range(2)
        )
        checks["strictly_increasing_time"] = all(
            f[TIMESTAMP].is_monotonic_increasing and not f[TIMESTAMP].duplicated().any()
            for f in frames
        )
        gaps_s = []
        ok_time = True
        for a, b in zip(frames[:-1], frames[1:]):
            gap = (b[TIMESTAMP].iloc[0] - a[TIMESTAMP].iloc[-1]).total_seconds()
            gaps_s.append(gap)
            ok_time &= gap > 0
            ok_time &= gap >= self.embargo_steps * sampling_seconds * 0.999
        checks["boundary_gaps_seconds"] = gaps_s
        checks["temporal_order_with_embargo"] = bool(ok_time)
        checks["train_max_before_calibration_min"] = bool(
            frames[0][TIMESTAMP].max() < frames[1][TIMESTAMP].min())
        checks["calibration_max_before_test_min"] = bool(
            frames[1][TIMESTAMP].max() < frames[2][TIMESTAMP].min())
        checks["passed"] = bool(
            checks["no_row_overlap"] and checks["position_order"]
            and checks["strictly_increasing_time"] and checks["temporal_order_with_embargo"]
            and checks["train_max_before_calibration_min"]
            and checks["calibration_max_before_test_min"]
        )
        checks["embargo_steps"] = self.embargo_steps
        checks["discarded_rows"] = self.n_total - sum(s.n_rows for s in specs)
        checks["splits"] = {s.name: s.as_dict() for s in specs}
        return checks


class TemporalSplitter:
    """Chronological splitter (fraction- or timestamp-based)."""

    def __init__(self, cfg: SplitConfig, embargo_steps: int) -> None:
        if embargo_steps < 0:
            raise SplitError("embargo_steps must be non-negative")
        self.cfg = cfg
        self.embargo_steps = int(embargo_steps)

    def _bounds_fraction(self, n: int) -> list[tuple[int, int]]:
        e = self.embargo_steps
        usable = n - 2 * e
        if usable < 3:
            raise SplitError(f"{n} rows cannot host 3 splits with embargo {e}")
        n_train = int(np.floor(self.cfg.train_frac * usable))
        n_cal = int(np.floor(self.cfg.calibration_frac * usable))
        n_test = usable - n_train - n_cal
        if min(n_train, n_cal, n_test) < 1:
            raise SplitError("a split would be empty; adjust fractions or embargo")
        a = (0, n_train)
        b = (a[1] + e, a[1] + e + n_cal)
        c = (b[1] + e, n)
        return [a, b, c]

    def _bounds_timestamp(self, ts: pd.Series) -> list[tuple[int, int]]:
        e = self.embargo_steps
        t_train_end = pd.Timestamp(self.cfg.train_end)
        t_cal_end = pd.Timestamp(self.cfg.calibration_end)
        if not t_train_end < t_cal_end:
            raise SplitError("train_end must precede calibration_end")
        values = ts.to_numpy()
        i_train = int(np.searchsorted(values, np.datetime64(t_train_end), side="left"))
        i_cal = int(np.searchsorted(values, np.datetime64(t_cal_end), side="left"))
        bounds = [(0, i_train), (i_train + e, i_cal), (i_cal + e, len(values))]
        if any(stop - start < 1 for start, stop in bounds):
            raise SplitError(f"timestamp boundaries yield an empty split: {bounds}")
        return bounds

    def split(self, frame: pd.DataFrame) -> TemporalSplit:
        """Split a time-sorted frame. The frame's index is preserved as row id."""
        ts = frame[TIMESTAMP]
        if not ts.is_monotonic_increasing:
            raise SplitError("frame must be sorted by timestamp before splitting")
        n = len(frame)
        if self.cfg.strategy == "fraction":
            bounds = self._bounds_fraction(n)
        else:
            bounds = self._bounds_timestamp(ts)
        frames: dict[str, pd.DataFrame] = {}
        specs: dict[str, SplitSpec] = {}
        for name, (start, stop) in zip(SPLIT_ORDER, bounds):
            part = frame.iloc[start:stop]
            n_pos = int(part[LABEL].sum())
            if n_pos < self.cfg.min_positive_rows:
                raise SplitError(
                    f"{name} split has {n_pos} attack rows (< min_positive_rows="
                    f"{self.cfg.min_positive_rows}); supervised training/calibration/evaluation "
                    "needs attacks in every split - adjust split fractions or boundaries"
                )
            frames[name] = part
            specs[name] = SplitSpec(
                name=name, start=start, stop=stop,
                start_ts=str(part[TIMESTAMP].iloc[0]), end_ts=str(part[TIMESTAMP].iloc[-1]),
                n_rows=stop - start, n_positive=n_pos,
            )
        return TemporalSplit(frames=frames, specs=specs, embargo_steps=self.embargo_steps,
                             n_total=n)
