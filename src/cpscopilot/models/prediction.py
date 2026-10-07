"""Stage 3 - multi-horizon future threat prediction.

One ensemble per horizon estimates ``P(attack within (t, t+h] | features up
to t)``. Each horizon's model is trained only on training rows whose full
horizon lies inside the training split (labels that would cross the split
boundary are undefined and dropped).
"""

from __future__ import annotations

import numpy as np

from cpscopilot.config import ModelConfig
from cpscopilot.data.leakage import FitLedger
from cpscopilot.models.ensemble import ProbabilisticEnsemble

__all__ = ["FutureThreatPredictor", "horizon_key"]


def horizon_key(minutes: float) -> str:
    """Stable string key for a horizon, e.g. ``15.0 -> "h15"``."""
    return f"h{minutes:g}"


class FutureThreatPredictor:
    """Independent ensemble per prediction horizon."""

    def __init__(self, horizons_minutes: list[float], cfg: ModelConfig, seed: int,
                 n_jobs: int = 1) -> None:
        self.horizons = list(horizons_minutes)
        self.models = {
            horizon_key(h): ProbabilisticEnsemble(f"prediction_{horizon_key(h)}", cfg, seed, n_jobs)
            for h in self.horizons
        }

    def fit(self, X: np.ndarray, y_future: dict[str, np.ndarray], split: str, ledger: FitLedger,
            row_ids: np.ndarray | None = None) -> FutureThreatPredictor:
        """Fit each horizon on rows where its label is defined.

        Args:
            y_future: horizon key -> float labels with NaN for undefined rows.
        """
        for key, model in self.models.items():
            y = y_future[key]
            mask = ~np.isnan(y)
            ids = row_ids[mask] if row_ids is not None else None
            model.fit(X[mask], y[mask].astype(np.int8), split, ledger, ids)
        return self

    def predict_members(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Horizon key -> per-member probabilities ``(M, n)``."""
        return {key: model.predict_member_proba(X) for key, model in self.models.items()}
