"""Independent evidence sources and their consistency.

:class:`RobustAnomalyScorer` is an unsupervised, physics-agnostic second
opinion on the *current* state: robust z-scores of the current sensor values
relative to **benign training rows only**. Its score is mapped to an evidence
level in ``[0, 1]`` that is 0 for typical behaviour and approaches 1 in the
extreme tail of the benign reference distribution.

:func:`evidence_consistency` measures agreement between sources:

* cross-source agreement ``1 - |p_detect - anomaly_evidence|`` (supervised
  detector vs. unsupervised deviation),
* temporal coherence ``1 - max_i max(0, p_{h_i} - p_{h_{i+1}})``: for nested
  horizons ``h_i < h_{i+1}`` a coherent predictor satisfies
  ``P(attack in (t, t+h_i]) <= P(attack in (t, t+h_{i+1}])``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cpscopilot.data.leakage import FitLedger, LeakageError, index_fingerprint

__all__ = ["RobustAnomalyScorer", "evidence_consistency"]

_Z_CAP = 10.0


class RobustAnomalyScorer:
    """Robust z-score deviation from benign training behaviour."""

    def __init__(self, tail_quantile: float = 0.95) -> None:
        self.tail_quantile = tail_quantile
        self.columns_: list[str] = []
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.reference_: np.ndarray | None = None

    def fit(self, current: pd.DataFrame, y_now: np.ndarray, split: str,
            ledger: FitLedger) -> RobustAnomalyScorer:
        if split != "train":
            raise LeakageError(f"anomaly scorer must be fitted on train, got {split!r}")
        benign = current[np.asarray(y_now) == 0]
        values = benign.to_numpy(dtype=np.float64)
        center = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - center), axis=0) * 1.4826
        std = np.nanstd(values, axis=0)
        scale = np.maximum(mad, std)
        self.columns_ = list(current.columns)
        self.center_ = np.nan_to_num(center)
        self.scale_ = np.where(np.nan_to_num(scale) > 1e-9, np.nan_to_num(scale), 1.0)
        self.reference_ = np.sort(self._raw_score(self.z(benign)))
        ledger.record("anomaly_scorer", split, len(benign), index_fingerprint(benign.index),
                      reference_size=len(self.reference_))
        return self

    def z(self, current: pd.DataFrame) -> np.ndarray:
        """Absolute robust z-scores (NaN -> 0), capped at 10."""
        if self.center_ is None:
            raise RuntimeError("anomaly scorer is not fitted")
        values = current[self.columns_].to_numpy(dtype=np.float64)
        z = np.abs((values - self.center_) / self.scale_)
        return np.clip(np.nan_to_num(z, nan=0.0), 0.0, _Z_CAP)

    @staticmethod
    def _raw_score(z: np.ndarray) -> np.ndarray:
        return np.mean(z**2, axis=1)

    def evidence(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(raw_score, evidence in [0, 1])`` for a z-score matrix."""
        assert self.reference_ is not None
        score = self._raw_score(z)
        ecdf = np.searchsorted(self.reference_, score, side="right") / len(self.reference_)
        tau = self.tail_quantile
        return score, np.clip((ecdf - tau) / (1.0 - tau), 0.0, 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {"tail_quantile": self.tail_quantile, "n_columns": len(self.columns_),
                "reference_size": 0 if self.reference_ is None else len(self.reference_)}


def evidence_consistency(
    p_detect: np.ndarray,
    anomaly_evidence: np.ndarray,
    p_future_ordered: np.ndarray | None,
    weights: dict[str, float],
) -> dict[str, np.ndarray]:
    """Combine cross-source agreement and temporal coherence into ``[0, 1]``.

    Args:
        p_detect: calibrated current-attack probability ``(n,)``.
        anomaly_evidence: unsupervised evidence ``(n,)``.
        p_future_ordered: calibrated future probabilities ``(H, n)`` ordered by
            increasing horizon, or ``None``.
        weights: ``{"cross_source": w1, "temporal_coherence": w2}``.
    """
    cross = 1.0 - np.abs(np.asarray(p_detect) - np.asarray(anomaly_evidence))
    if p_future_ordered is not None and len(p_future_ordered) >= 2:
        violations = np.clip(p_future_ordered[:-1] - p_future_ordered[1:], 0.0, None)
        temporal = 1.0 - violations.max(axis=0)
    else:
        temporal = np.ones_like(cross)
    w_c = float(weights.get("cross_source", 0.7))
    w_t = float(weights.get("temporal_coherence", 0.3))
    total = w_c + w_t
    if total <= 0:
        raise ValueError("consistency weights must sum to a positive value")
    combined = (w_c * cross + w_t * temporal) / total
    return {"cross_source": cross, "temporal_coherence": temporal,
            "consistency": np.clip(combined, 0.0, 1.0)}
