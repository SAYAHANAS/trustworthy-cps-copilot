"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpscopilot.config import ExperimentConfig, load_config
from cpscopilot.data.schema import LABEL, TIMESTAMP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE = PROJECT_ROOT / "configs" / "experiments" / "synthetic_smoke.yaml"

# Small enough for CI, large enough that every split contains attacks.
TINY_OVERRIDES = [
    "dataset.synthetic.days=1.5",
    "dataset.synthetic.n_attacks=12",
    "detection.ensemble_size=2",
    "prediction.ensemble_size=2",
    "detection.params.n_estimators=30",
    "prediction.params.n_estimators=30",
    "evaluation.bootstrap_resamples=8",
    "copilot.max_events=10",
    "runtime.leakage_probes=3",
    "runtime.n_jobs=1",
]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def tiny_config(tmp_path: Path) -> ExperimentConfig:
    cfg = load_config(SMOKE, TINY_OVERRIDES)
    return cfg.model_copy(update={"output_root": str(tmp_path / "runs")})


def make_series_frame(n: int = 600, n_cols: int = 3, seed: int = 0, period_s: int = 10,
                      attack_every: int = 97, attack_len: int = 9) -> pd.DataFrame:
    """Regularly sampled random-walk telemetry with periodic labelled attacks."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        TIMESTAMP: pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(n) * period_s, "s")
    })
    for j in range(n_cols):
        values = np.cumsum(rng.normal(0, 1, n))
        values[rng.random(n) < 0.02] = np.nan
        frame[f"S{j}"] = values
    y = np.zeros(n, dtype=np.int8)
    for start in range(attack_every // 2, n, attack_every):
        y[start: start + attack_len] = 1
    frame[LABEL] = y
    return frame


@pytest.fixture()
def series_frame() -> pd.DataFrame:
    return make_series_frame()
