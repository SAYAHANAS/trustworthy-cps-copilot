"""Canonical in-memory representation of a CPS telemetry dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["CPSDataset", "TIMESTAMP", "LABEL", "ATTACK_TYPE", "BENIGN_TYPE"]

TIMESTAMP = "timestamp"
LABEL = "label"
ATTACK_TYPE = "attack_type"
BENIGN_TYPE = "normal"


@dataclass
class CPSDataset:
    """A chronologically sorted telemetry table plus its metadata.

    Attributes:
        frame: one row per time step; contains ``timestamp``, the feature
            columns, the binary ``label`` (1 = attack in progress) and optional
            metadata columns (attack type, phase, campaign id).
        feature_cols: model inputs. Metadata columns are *never* features; the
            leakage audit verifies this.
        sampling_seconds: nominal sampling period (after any resampling).
        source: dataset identifier.
        fingerprint: provenance (raw-file SHA-256 digests or generator hash).
        meta_cols: label-derived/diagnostic columns kept for stratified
            evaluation only.
        categorical_cols: string-valued inputs; they are one-hot encoded with a
            vocabulary learnt on the training split only.
    """

    frame: pd.DataFrame
    feature_cols: list[str]
    sampling_seconds: float
    source: str
    fingerprint: dict[str, Any] = field(default_factory=dict)
    meta_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        missing = [c for c in [TIMESTAMP, LABEL, *self.feature_cols] if c not in self.frame]
        if missing:
            raise ValueError(f"dataset is missing required columns: {missing[:10]}")
        overlap = set(self.feature_cols) & ({TIMESTAMP, LABEL} | set(self.meta_cols))
        if overlap:
            raise ValueError(f"label/meta columns cannot be features: {sorted(overlap)}")
        ts = self.frame[TIMESTAMP]
        if not ts.is_monotonic_increasing:
            raise ValueError("dataset frame must be sorted by timestamp")
        if ts.isna().any():
            raise ValueError("timestamps must not contain NaT")

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def timestamps(self) -> np.ndarray:
        return self.frame[TIMESTAMP].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.frame[LABEL].to_numpy(dtype=np.int8)

    def summary(self) -> dict[str, Any]:
        y = self.labels
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_features": len(self.feature_cols),
            "sampling_seconds": self.sampling_seconds,
            "start": str(self.frame[TIMESTAMP].iloc[0]),
            "end": str(self.frame[TIMESTAMP].iloc[-1]),
            "attack_rows": int(y.sum()),
            "attack_prevalence": float(y.mean()) if len(y) else float("nan"),
            "attack_episodes": int(np.sum(np.diff(np.r_[0, y]) == 1)),
            "meta_cols": list(self.meta_cols),
        }
