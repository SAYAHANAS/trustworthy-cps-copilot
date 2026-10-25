"""Stage 2 - detection of the *current* security state.

The detector estimates ``P(attack in progress at t | features up to t)``. Its
calibrated output is discretised by the trust layer into the operational
states benign / suspicious / malicious using thresholds chosen on the
calibration split.
"""

from __future__ import annotations

import numpy as np

from cpscopilot.config import ModelConfig
from cpscopilot.data.leakage import FitLedger
from cpscopilot.models.ensemble import ProbabilisticEnsemble

__all__ = ["DetectionModule"]


class DetectionModule:
    """Ensemble classifier for the current state ``y_now``."""

    name = "detection"

    def __init__(self, cfg: ModelConfig, seed: int, n_jobs: int = 1) -> None:
        self.ensemble = ProbabilisticEnsemble(self.name, cfg, seed, n_jobs)

    def fit(self, X: np.ndarray, y_now: np.ndarray, split: str, ledger: FitLedger,
            row_ids: np.ndarray | None = None) -> DetectionModule:
        self.ensemble.fit(X, y_now, split, ledger, row_ids)
        return self

    def predict_members(self, X: np.ndarray) -> np.ndarray:
        """Per-member attack probabilities, shape ``(M, n)``."""
        return self.ensemble.predict_member_proba(X)
