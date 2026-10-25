"""Calibrators and calibration/discrimination metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, softmax

from cpscopilot.data.leakage import FitLedger, LeakageError
from cpscopilot.evaluation.bootstrap import block_bootstrap_ci
from cpscopilot.evaluation.calibration_metrics import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)
from cpscopilot.evaluation.metrics import classification_metrics, compact_curves
from cpscopilot.trust.calibration import (
    TemperatureScaler,
    fit_temperature_multiclass,
    make_calibrator,
)


def overconfident(n: int, true_t: float, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 4, n)
    y = (rng.random(n) < expit(z / true_t)).astype(int)
    return expit(z), y  # the "model" reports σ(z): too confident by factor true_t


def test_temperature_scaling_recovers_known_temperature() -> None:
    p, y = overconfident(40000, 2.5, 0)
    ts = TemperatureScaler().fit(p, y, "calibration", FitLedger(), "m")
    assert ts.temperature_ == pytest.approx(2.5, rel=0.06)
    assert ts.nll_after_ < ts.nll_before_


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_calibrators_reduce_held_out_ece(method: str) -> None:
    p_cal, y_cal = overconfident(20000, 3.0, 1)
    p_test, y_test = overconfident(20000, 3.0, 2)
    cal = make_calibrator(method).fit(p_cal, y_cal, "calibration", FitLedger(), "m")
    after = cal.transform(p_test)
    assert ((after >= 0) & (after <= 1)).all()
    assert expected_calibration_error(y_test, after) < 0.5 * expected_calibration_error(
        y_test, p_test)


def test_temperature_scaling_preserves_ranking() -> None:
    p, y = overconfident(2000, 2.0, 3)
    ts = TemperatureScaler().fit(p, y, "calibration", FitLedger(), "m")
    assert np.array_equal(np.argsort(p, kind="mergesort"),
                          np.argsort(ts.transform(p), kind="mergesort"))


@pytest.mark.parametrize("split", ["train", "test"])
def test_calibrators_refuse_wrong_split(split: str) -> None:
    p, y = overconfident(100, 2.0, 4)
    for method in ("none", "temperature", "platt", "isotonic"):
        with pytest.raises(LeakageError):
            make_calibrator(method).fit(p, y, split, FitLedger(), "m")


def test_multiclass_temperature() -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(0, 3, (30000, 4))
    probs = softmax(logits / 2.0, axis=1)
    y = np.array([rng.choice(4, p=row) for row in probs])
    assert fit_temperature_multiclass(logits, y) == pytest.approx(2.0, rel=0.08)


def test_ece_mce_brier_on_a_hand_computed_example() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    # 2 bins: [0, .5) -> conf .1 vs freq .5 ; [.5, 1] -> conf .9 vs freq 1.0
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    assert maximum_calibration_error(y, p, n_bins=2) == pytest.approx(0.4)
    assert brier_score(y, p) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)


def test_probability_one_falls_in_the_last_bin() -> None:
    bins = reliability_bins(np.array([1, 0]), np.array([1.0, 0.0]), n_bins=10)
    assert bins["count"][-1] == 1 and bins["count"][0] == 1


def test_perfectly_calibrated_scores_have_small_ece() -> None:
    rng = np.random.default_rng(6)
    p = rng.random(100000)
    y = (rng.random(100000) < p).astype(int)
    report = calibration_report(y, p, 15)
    assert report["ece"] < 0.01 and report["adaptive_ece"] < 0.01
    assert report["ece_top_label"] < 0.01


def test_imbalance_aware_classification_metrics() -> None:
    y = np.array([0] * 95 + [1] * 5)
    p = np.r_[np.linspace(0, 0.4, 95), np.linspace(0.6, 1, 5)]
    m = classification_metrics(y, p, threshold=0.5)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    always_benign = classification_metrics(y, np.zeros(100), 0.5)
    assert always_benign["balanced_accuracy"] == pytest.approx(0.5)  # accuracy would say 0.95
    curves = compact_curves(y, p, 50)
    assert curves["available"] and len(curves["pr"]["precision"]) == 50


def test_block_bootstrap_interval_contains_estimate() -> None:
    rng = np.random.default_rng(7)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(int)
    ci = block_bootstrap_ci(y, p, {"brier": brier_score}, 200, 30, np.random.default_rng(0))
    assert ci["brier"]["lower"] <= ci["brier"]["estimate"] <= ci["brier"]["upper"]
    assert ci["brier"]["valid"] == 200
