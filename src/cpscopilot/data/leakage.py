"""Explicit anti-leakage instrumentation.

* :class:`FitLedger` records which split every stateful component was fitted
  on. The runner asserts that nothing was ever fitted on the test split and
  that train-only components (imputer, scaler, encoders, models, anomaly
  scorer) were fitted on exactly the training rows.
* :func:`audit_feature_causality` re-computes features on *truncated* histories
  (everything after ``t`` removed) and on *perturbed futures* (everything after
  ``t`` replaced by noise) and requires the features at ``t`` to be unchanged.
* :func:`audit_future_labels` re-derives future labels by brute force.
* :func:`audit_feature_columns` rejects label/meta columns among the inputs and
  flags suspicious label proxies (near-perfect correlation with the target).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cpscopilot.data.features import CausalWindowFeaturizer
from cpscopilot.data.labels import future_labels

__all__ = [
    "FitLedger",
    "LeakageError",
    "LeakageReport",
    "audit_feature_causality",
    "audit_feature_columns",
    "audit_future_labels",
    "index_fingerprint",
]


class LeakageError(RuntimeError):
    """A temporal/data leakage invariant was violated."""


def index_fingerprint(index: pd.Index | np.ndarray) -> str:
    """Order-sensitive SHA-256 of row identifiers."""
    values = np.asarray(index, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass
class FitLedger:
    """Append-only record of every ``fit`` call in a run."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, component: str, split: str, n_rows: int,
               fingerprint: str | None = None, **extra: Any) -> None:
        self.entries.append({"component": component, "split": split, "n_rows": int(n_rows),
                             "index_sha256": fingerprint, **extra})

    def assert_not_fitted_on(self, split: str) -> None:
        bad = [e["component"] for e in self.entries if e["split"] == split]
        if bad:
            raise LeakageError(f"components fitted on the {split} split: {bad}")

    def components_on(self, split: str) -> list[str]:
        return [e["component"] for e in self.entries if e["split"] == split]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": list(self.entries)}


@dataclass
class LeakageReport:
    """Collection of named checks; ``passed`` is False if any critical check failed."""

    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, passed: bool, critical: bool = True, **detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "critical": critical, **detail})

    def add_result(self, name: str, result: dict[str, Any], critical: bool = True) -> None:
        """Add an audit-function result dict that carries its own ``passed`` flag."""
        detail = {k: v for k, v in result.items() if k != "passed"}
        self.add(name, bool(result["passed"]), critical, **detail)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks if c["critical"])

    def failures(self) -> list[str]:
        return [c["name"] for c in self.checks if c["critical"] and not c["passed"]]

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise LeakageError(f"leakage audit failed: {self.failures()}")

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": self.failures(), "checks": self.checks}


def audit_feature_causality(
    featurizer: CausalWindowFeaturizer,
    frame: pd.DataFrame,
    base_cols: list[str],
    n_probes: int,
    rng: np.random.Generator,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Truncation and future-perturbation tests on random probe positions.

    For each probe ``t`` (with complete history) the features at ``t`` must be
    identical when (a) every row after ``t`` is removed and (b) every row after
    ``t`` is replaced by random noise and NaNs.
    """
    full, valid = featurizer.transform(frame, base_cols)
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        return {"passed": True, "probes": [], "note": "no valid rows"}
    probes = np.sort(rng.choice(candidates, size=min(n_probes, len(candidates)), replace=False))
    max_trunc_err = 0.0
    max_perturb_err = 0.0
    for t in probes:
        truncated, _ = featurizer.transform(frame.iloc[: t + 1], base_cols)
        a = full.iloc[t].to_numpy(dtype=np.float64)
        b = truncated.iloc[t].to_numpy(dtype=np.float64)
        max_trunc_err = max(max_trunc_err, float(np.nanmax(np.abs(a - b), initial=0.0)))
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            max_trunc_err = float("inf")
        perturbed = frame.copy()
        tail = perturbed.index[t + 1:]
        if len(tail):
            noise = rng.normal(0.0, 1e6, size=(len(tail), len(base_cols)))
            noise[rng.random(noise.shape) < 0.1] = np.nan
            perturbed.loc[tail, base_cols] = noise
        pert, _ = featurizer.transform(perturbed, base_cols)
        c = pert.iloc[t].to_numpy(dtype=np.float64)
        max_perturb_err = max(max_perturb_err, float(np.nanmax(np.abs(a - c), initial=0.0)))
        if not np.array_equal(np.isnan(a), np.isnan(c)):
            max_perturb_err = float("inf")
    passed = max_trunc_err <= atol and max_perturb_err <= atol
    return {
        "passed": bool(passed),
        "probes": probes.tolist(),
        "max_truncation_abs_error": max_trunc_err,
        "max_future_perturbation_abs_error": max_perturb_err,
    }


def audit_future_labels(
    y_now: np.ndarray,
    y_future: np.ndarray,
    horizon: int,
    mode: str,
    n_probes: int,
    rng: np.random.Generator,
    segments: np.ndarray | None = None,
) -> dict[str, Any]:
    """Brute-force re-derivation of future labels at probe positions.

    Verifies, for one split, that every defined label equals the maximum of the
    *strictly future* in-segment window and that labels whose horizon would
    leave the segment (and hence the split) are undefined.
    """
    n = len(y_now)
    segments = np.zeros(n, dtype=np.int64) if segments is None else np.asarray(segments)
    seg_end = np.empty(n, dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(segments)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, n]):
        seg_end[start:stop] = stop
    probes = rng.choice(n, size=min(n_probes, n), replace=False) if n else np.array([], int)
    probes = np.unique(np.r_[probes, np.arange(max(0, n - horizon - 1), n)])
    mismatches = 0
    for t in probes:
        if t + horizon >= seg_end[t] or (mode == "onset" and y_now[t] == 1):
            expected = np.nan
        else:
            expected = float(np.max(y_now[t + 1: t + horizon + 1]))
        got = y_future[t]
        if not ((np.isnan(expected) and np.isnan(got)) or expected == got):
            mismatches += 1
    tail_nan = bool(np.isnan(y_future[-horizon:]).all()) if n >= horizon else True
    reference = future_labels(y_now[: seg_end[0]], horizon, mode)  # type: ignore[arg-type]
    first_segment_ok = np.allclose(np.nan_to_num(reference, nan=-1),
                                   np.nan_to_num(y_future[: seg_end[0]], nan=-1))
    return {"passed": bool(mismatches == 0 and tail_nan and first_segment_ok),
            "tail_undefined": tail_nan, "brute_force_mismatches": int(mismatches),
            "n_probes": int(len(probes))}


def audit_feature_columns(
    feature_cols: list[str],
    forbidden: list[str],
    X_train: pd.DataFrame | None = None,
    y_train: np.ndarray | None = None,
    proxy_threshold: float = 0.995,
) -> dict[str, Any]:
    """Reject forbidden columns and flag near-perfect label proxies on train."""
    forbidden_set = {f.lower() for f in forbidden}
    hits = [c for c in feature_cols if c.split("__", 1)[0].lower() in forbidden_set]
    proxies: list[dict[str, Any]] = []
    if X_train is not None and y_train is not None and len(np.unique(y_train)) == 2:
        y = (y_train - y_train.mean()) / (y_train.std() + 1e-12)
        X = X_train.to_numpy(dtype=np.float64)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        Z = np.where(np.isnan(X), 0.0, (X - mu) / np.where(sd > 0, sd, 1.0))
        corr = np.abs(Z.T @ y) / len(y)
        for j in np.flatnonzero(corr >= proxy_threshold):
            proxies.append({"feature": feature_cols[j], "abs_corr": float(corr[j])})
    return {"passed": not hits, "forbidden_columns_found": hits, "label_proxies": proxies}
