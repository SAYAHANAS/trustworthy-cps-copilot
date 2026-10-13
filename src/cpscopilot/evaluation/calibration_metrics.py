"""Calibration metrics, computed separately from discrimination metrics.

For binary attack probabilities ``p`` and outcomes ``y``:

* **positive-class ECE** - bins on ``p``; per bin compare the mean predicted
  probability with the observed attack frequency (reliability diagram),
* **top-label ECE** - bins on the confidence ``max(p, 1 - p)``; compare with
  the accuracy of the implied decision,
* **MCE** - maximum per-bin gap (worst-case miscalibration),
* **adaptive ECE** - equal-mass (quantile) bins, robust to the skewed score
  distributions typical of rare attacks,
* **Brier score** and **NLL** - proper scoring rules.

``ECE = Σ_b (n_b / N) |acc_b - conf_b|`` and ``MCE = max_b |acc_b - conf_b|``
over non-empty bins.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = [
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "reliability_bins",
]

_EPS = 1e-12


def reliability_bins(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 15,
    mode: Literal["positive", "top_label"] = "positive",
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> dict[str, np.ndarray]:
    """Per-bin confidence, observed frequency/accuracy and counts.

    Uniform bins are ``[0, 1/B), ..., [(B-1)/B, 1]`` (the last bin is closed).
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    if mode == "positive":
        conf, outcome = p, y
    else:
        conf = np.maximum(p, 1.0 - p)
        outcome = ((p >= 0.5).astype(float) == y).astype(float)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    k = len(edges) - 1
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    sum_conf = np.bincount(idx, weights=conf, minlength=k)
    sum_out = np.bincount(idx, weights=outcome, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / counts, np.nan)
        frequency = np.where(counts > 0, sum_out / counts, np.nan)
    return {"lower": edges[:-1], "upper": edges[1:], "confidence": mean_conf,
            "frequency": frequency, "count": counts}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
                               mode: Literal["positive", "top_label"] = "positive",
                               strategy: Literal["uniform", "quantile"] = "uniform") -> float:
    bins = reliability_bins(y, p, n_bins, mode, strategy)
    n = bins["count"].sum()
    if n == 0:
        return float("nan")
    gap = np.abs(bins["frequency"] - bins["confidence"])
    return float(np.nansum(bins["count"] * gap) / n)


def maximum_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
                              mode: Literal["positive", "top_label"] = "positive") -> float:
    bins = reliability_bins(y, p, n_bins, mode, "uniform")
    gap = np.abs(bins["frequency"] - bins["confidence"])
    return float(np.nanmax(gap)) if np.any(bins["count"] > 0) else float("nan")


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))


def negative_log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    """All calibration metrics plus the positive-class reliability-diagram data."""
    bins = reliability_bins(y, p, n_bins, "positive", "uniform")
    return {
        "ece": expected_calibration_error(y, p, n_bins, "positive"),
        "mce": maximum_calibration_error(y, p, n_bins, "positive"),
        "ece_top_label": expected_calibration_error(y, p, n_bins, "top_label"),
        "mce_top_label": maximum_calibration_error(y, p, n_bins, "top_label"),
        "adaptive_ece": expected_calibration_error(y, p, n_bins, "positive", "quantile"),
        "brier": brier_score(y, p),
        "nll": negative_log_likelihood(y, p),
        "mean_probability": float(np.mean(p)),
        "observed_frequency": float(np.mean(y)),
        "reliability": {k: v.tolist() for k, v in bins.items()},
    }
