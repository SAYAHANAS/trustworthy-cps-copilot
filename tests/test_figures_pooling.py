"""Figure helpers: pooled reliability bins and band dispersion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpscopilot.evaluation.calibration_metrics import expected_calibration_error, reliability_bins
from cpscopilot.plotting.figures import RunArtifacts, _mean_curve, _pooled_reliability


def fake_run(y: np.ndarray, p: np.ndarray, roc_tpr: list[float]) -> RunArtifacts:
    bins = {k: v.tolist() for k, v in reliability_bins(y, p, 10).items()}
    curves = {"detection": {"temperature": {"available": True,
                                            "roc": {"fpr": [0.0, 0.5, 1.0], "tpr": roc_tpr},
                                            "pr": {"recall": [0, 1], "precision": [1, 0]}}}}
    return RunArtifacts(run_dir=Path("."), metrics={}, curves=curves, copilot_audit=None,
                        reliability={"detection": {"temperature": bins}})


def test_pooled_reliability_equals_reliability_of_the_pooled_rows() -> None:
    rng = np.random.default_rng(0)
    p1, p2 = rng.random(500), rng.random(800) ** 2
    y1, y2 = (rng.random(500) < p1).astype(int), (rng.random(800) < 0.5).astype(int)
    pooled = _pooled_reliability([fake_run(y1, p1, [0, .5, 1]), fake_run(y2, p2, [0, .5, 1])],
                                 "detection", "temperature")
    joint = reliability_bins(np.r_[y1, y2], np.r_[p1, p2], 10)
    np.testing.assert_array_equal(pooled["count"], joint["count"])
    np.testing.assert_allclose(pooled["frequency"], joint["frequency"], equal_nan=True)
    assert pooled["ece"] == pytest.approx(expected_calibration_error(np.r_[y1, y2],
                                                                     np.r_[p1, p2], 10))


def test_curve_band_uses_sample_standard_deviation() -> None:
    y, p = np.array([0, 1]), np.array([0.2, 0.8])
    runs = [fake_run(y, p, [0, 0.4, 1]), fake_run(y, p, [0, 0.6, 1])]
    _, mean, sd = _mean_curve(runs, "detection", "temperature", "roc")
    assert mean[1] == pytest.approx(0.5)
    assert sd[1] == pytest.approx(np.std([0.4, 0.6], ddof=1))
