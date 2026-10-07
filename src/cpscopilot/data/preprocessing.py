"""Train-only preprocessing: categorical encoding, imputation, scaling.

All statistics are learnt from the training split and frozen. ``fit`` refuses
any split other than ``"train"`` and records itself in the
:class:`~cpscopilot.data.leakage.FitLedger`; ``transform`` never updates state.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from cpscopilot.data.leakage import FitLedger, LeakageError, index_fingerprint

__all__ = ["LeakageSafePreprocessor", "TrainOnlyOneHotEncoder"]


def _require_train(split: str, component: str) -> None:
    if split != "train":
        raise LeakageError(f"{component} may only be fitted on the train split, got {split!r}")


class TrainOnlyOneHotEncoder:
    """One-hot encoding with a vocabulary learnt on the training split.

    Categories unseen during training map to a shared ``__other__`` column, so
    the test vocabulary can never influence the feature space.
    """

    def __init__(self, columns: list[str], max_categories: int = 20) -> None:
        self.columns = list(columns)
        self.max_categories = max_categories
        self.vocabulary_: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame, split: str, ledger: FitLedger) -> TrainOnlyOneHotEncoder:
        _require_train(split, "TrainOnlyOneHotEncoder")
        for col in self.columns:
            counts = frame[col].astype(str).value_counts()
            top = counts.sort_index().sort_values(ascending=False, kind="mergesort")
            self.vocabulary_[col] = list(top.index[: self.max_categories])
        ledger.record("categorical_encoder", split, len(frame), index_fingerprint(frame.index))
        return self

    def output_columns(self) -> list[str]:
        return [f"{c}={v}" for c in self.columns for v in [*self.vocabulary_[c], "__other__"]]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.vocabulary_:
            raise RuntimeError("encoder is not fitted")
        blocks = {}
        for col in self.columns:
            values = frame[col].astype(str)
            vocab = self.vocabulary_[col]
            for v in vocab:
                blocks[f"{col}={v}"] = (values == v).astype(np.float32).to_numpy()
            blocks[f"{col}=__other__"] = (~values.isin(vocab)).astype(np.float32).to_numpy()
        return pd.DataFrame(blocks, index=frame.index)


class LeakageSafePreprocessor:
    """Median imputation + standardisation + constant-column removal.

    Args:
        scaler: ``standard`` (mean/std), ``robust`` (median/IQR) or ``none``.
    """

    def __init__(self, scaler: Literal["standard", "robust", "none"] = "standard") -> None:
        self.scaler = scaler
        self.columns_: list[str] = []
        self.kept_: list[str] = []
        self.median_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fit_fingerprint_: str | None = None
        self.fit_rows_: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.median_ is not None

    def fit(self, X: pd.DataFrame, split: str, ledger: FitLedger) -> LeakageSafePreprocessor:
        _require_train(split, "LeakageSafePreprocessor")
        values = X.to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        all_nan = np.all(np.isnan(values), axis=0)
        median = np.where(all_nan, 0.0, np.nanmedian(np.where(all_nan, 0.0, values), axis=0))
        filled = np.where(np.isnan(values), median, values)
        if self.scaler == "standard":
            center, scale = filled.mean(axis=0), filled.std(axis=0)
        elif self.scaler == "robust":
            q75, q25 = np.percentile(filled, [75, 25], axis=0)
            center, scale = np.median(filled, axis=0), q75 - q25
        else:
            center, scale = np.zeros(filled.shape[1]), np.ones(filled.shape[1])
        constant = (np.ptp(filled, axis=0) <= 1e-12) | all_nan
        keep = ~constant
        self.columns_ = list(X.columns)
        self.kept_ = [c for c, k in zip(X.columns, keep) if k]
        self.median_ = median[keep]
        self.center_ = center[keep]
        self.scale_ = np.where(scale[keep] > 1e-12, scale[keep], 1.0)
        self._keep_mask = keep
        self.fit_fingerprint_ = index_fingerprint(X.index)
        self.fit_rows_ = len(X)
        ledger.record("preprocessor", split, len(X), self.fit_fingerprint_,
                      n_inputs=len(self.columns_), n_kept=len(self.kept_),
                      dropped_constant=int(constant.sum()))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("preprocessor is not fitted")
        if list(X.columns) != self.columns_:
            raise ValueError("column mismatch between fit and transform")
        values = X.to_numpy(dtype=np.float64, copy=True)[:, self._keep_mask]
        values[~np.isfinite(values)] = np.nan
        values = np.where(np.isnan(values), self.median_, values)
        return ((values - self.center_) / self.scale_).astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return {
            "scaler": self.scaler,
            "n_inputs": len(self.columns_),
            "n_kept": len(self.kept_),
            "dropped_constant": [c for c in self.columns_ if c not in set(self.kept_)],
            "fit_rows": self.fit_rows_,
            "fit_index_sha256": self.fit_fingerprint_,
        }
