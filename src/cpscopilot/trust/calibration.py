"""Post-hoc probability calibration.

All calibrators are fitted on the **calibration split only** (fitting on the
training split would calibrate against in-sample, over-confident scores;
fitting on test would leak). ``fit`` refuses any other split.

* :class:`TemperatureScaler` - one parameter ``T > 0``; ``p' = σ(logit(p) / T)``
  (binary) or ``softmax(z / T)`` (multi-class). Minimises the negative
  log-likelihood with a bounded, deterministic 1-D search over ``log T``.
  Accuracy/ranking is unchanged because the map is monotone.
* :class:`PlattScaler` - ``p' = σ(a·logit(p) + b)`` (adds a bias term, which
  can also correct prior shift caused by class re-weighting).
* :class:`IsotonicCalibrator` - non-parametric monotone map.
* :class:`IdentityCalibrator` - no-op baseline.

Calibrators are fitted on the *ensemble-mean* probability. ``memberwise`` says
whether the same map may also be applied to individual members (needed for the
uncertainty decomposition): true for smooth monotone parametric maps
(temperature, Platt, identity), false for isotonic regression, whose plateaus
would collapse or tear apart member disagreement.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_softmax

from cpscopilot.data.leakage import FitLedger, LeakageError

__all__ = [
    "Calibrator",
    "EPS",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattScaler",
    "TemperatureScaler",
    "binary_nll",
    "fit_temperature_multiclass",
    "logit",
    "make_calibrator",
]

EPS = 1e-7
_LOG_T_BOUNDS = (np.log(1e-2), np.log(1e2))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def binary_nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _check_split(split: str, name: str) -> None:
    if split != "calibration":
        raise LeakageError(f"{name} must be fitted on the calibration split, got {split!r}")


class Calibrator(Protocol):
    name: str
    memberwise: bool

    def fit(self, p: np.ndarray, y: np.ndarray, split: str, ledger: FitLedger,
            component: str) -> Calibrator: ...

    def transform(self, p: np.ndarray) -> np.ndarray: ...

    def state_dict(self) -> dict[str, Any]: ...


class _Base:
    name = "base"
    memberwise = True

    def _record(self, ledger: FitLedger, split: str, n: int, component: str) -> None:
        ledger.record(f"calibrator:{component}:{self.name}", split, n)

    def transform(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class IdentityCalibrator(_Base):
    name = "none"

    def fit(self, p: np.ndarray, y: np.ndarray, split: str, ledger: FitLedger,
            component: str) -> IdentityCalibrator:
        _check_split(split, "IdentityCalibrator")
        self._record(ledger, split, len(p), component)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {"method": self.name}


class TemperatureScaler(_Base):
    """Binary temperature scaling on logits derived from probabilities."""

    name = "temperature"

    def __init__(self) -> None:
        self.temperature_: float = 1.0
        self.nll_before_: float | None = None
        self.nll_after_: float | None = None

    def fit(self, p: np.ndarray, y: np.ndarray, split: str, ledger: FitLedger,
            component: str) -> TemperatureScaler:
        _check_split(split, "TemperatureScaler")
        z = logit(p)
        y = np.asarray(y, dtype=np.float64)

        def nll(log_t: float) -> float:
            return binary_nll(expit(z / np.exp(log_t)), y)

        res = minimize_scalar(nll, bounds=_LOG_T_BOUNDS, method="bounded",
                              options={"xatol": 1e-8})
        self.temperature_ = float(np.exp(res.x))
        self.nll_before_ = binary_nll(p, y)
        self.nll_after_ = float(res.fun)
        self._record(ledger, split, len(p), component)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return expit(logit(p) / self.temperature_)

    def state_dict(self) -> dict[str, Any]:
        return {"method": self.name, "temperature": self.temperature_,
                "nll_before": self.nll_before_, "nll_after": self.nll_after_}


def fit_temperature_multiclass(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the multi-class NLL of ``softmax(logits / T)``."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    def nll(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-np.mean(lp[np.arange(len(y)), y]))

    res = minimize_scalar(nll, bounds=_LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-8})
    return float(np.exp(res.x))


class PlattScaler(_Base):
    """Logistic recalibration of the logit (slope and intercept)."""

    name = "platt"

    def __init__(self) -> None:
        self.a_: float = 1.0
        self.b_: float = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray, split: str, ledger: FitLedger,
            component: str) -> PlattScaler:
        _check_split(split, "PlattScaler")
        z = logit(p)
        y = np.asarray(y, dtype=np.float64)

        def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
            q = np.clip(expit(theta[0] * z + theta[1]), EPS, 1 - EPS)
            value = -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))
            grad_common = q - y
            return float(value), np.array([np.mean(grad_common * z), np.mean(grad_common)])

        res = minimize(loss, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        self.a_, self.b_ = float(res.x[0]), float(res.x[1])
        self._record(ledger, split, len(p), component)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return expit(self.a_ * logit(p) + self.b_)

    def state_dict(self) -> dict[str, Any]:
        return {"method": self.name, "a": self.a_, "b": self.b_}


class IsotonicCalibrator(_Base):
    """Monotone non-parametric calibration (sklearn ``IsotonicRegression``)."""

    name = "isotonic"
    memberwise = False

    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip",
                                        increasing=True)

    def fit(self, p: np.ndarray, y: np.ndarray, split: str, ledger: FitLedger,
            component: str) -> IsotonicCalibrator:
        _check_split(split, "IsotonicCalibrator")
        self.model.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self._record(ledger, split, len(p), component)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        shape = np.shape(p)
        out = self.model.predict(np.asarray(p, dtype=np.float64).ravel())
        return np.clip(out, EPS, 1 - EPS).reshape(shape)

    def state_dict(self) -> dict[str, Any]:
        return {"method": self.name, "n_thresholds": int(len(self.model.X_thresholds_))}


def make_calibrator(method: str) -> Calibrator:
    """Factory for ``none`` / ``temperature`` / ``platt`` / ``isotonic``."""
    table: dict[str, type] = {
        "none": IdentityCalibrator,
        "temperature": TemperatureScaler,
        "platt": PlattScaler,
        "isotonic": IsotonicCalibrator,
    }
    if method not in table:
        raise ValueError(f"unknown calibration method {method!r}")
    return table[method]()
