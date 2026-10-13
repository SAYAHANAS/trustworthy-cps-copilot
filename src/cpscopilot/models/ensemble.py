"""Seeded ensembles of probabilistic binary classifiers.

An ensemble of ``M`` independently seeded members provides the per-member
probabilities needed for epistemic-uncertainty estimation (deep-ensemble
style mutual information, see :mod:`cpscopilot.trust.uncertainty`). Diversity
comes from seeded row/column subsampling and, optionally, from a temporal
moving-block bootstrap that preserves autocorrelation inside blocks.

Class imbalance is handled with class weights computed on each member's own
training sample (``scale_pos_weight`` for XGBoost, sample weights otherwise).
Re-weighting distorts probabilities, which is one reason the trust layer
re-calibrates them on a held-out split.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cpscopilot.config import ModelConfig
from cpscopilot.data.leakage import FitLedger, LeakageError, index_fingerprint
from cpscopilot.utils.logging import get_logger
from cpscopilot.utils.seed import derive_seed

__all__ = ["ProbabilisticEnsemble", "build_estimator", "has_internal_randomness",
           "moving_block_indices"]

log = get_logger("models.ensemble")

_XGB_KEYS = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
             "min_child_weight", "reg_lambda", "reg_alpha", "gamma", "max_bin"}


def moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap sample of ``n`` row positions."""
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
    return np.sort(idx.ravel()[:n])


def build_estimator(cfg: ModelConfig, seed: int, n_jobs: int, pos_weight: float) -> Any:
    """Instantiate one ensemble member for the configured backend."""
    params = dict(cfg.params)
    if cfg.backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("backend 'xgboost' requires the xgboost package") from exc
        unknown = set(params) - _XGB_KEYS
        if unknown:
            raise ValueError(f"unsupported xgboost params: {sorted(unknown)}")
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=n_jobs,
            scale_pos_weight=pos_weight,
            verbosity=0,
        )
    if cfg.backend == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=params.get("max_depth"),
            l2_regularization=float(params.get("reg_lambda", 0.0)),
            early_stopping=False,
            random_state=seed,
        )
    if cfg.backend == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=n_jobs,
            random_state=seed,
        )
    if cfg.backend == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=float(params.get("C", 1.0)), max_iter=2000,
                                  random_state=seed)
    raise ValueError(f"unknown backend {cfg.backend!r}")


def has_internal_randomness(cfg: ModelConfig) -> bool:
    """Whether differently seeded members of this backend differ without resampling.

    ``hist_gb`` (no early stopping, < 200k rows) and ``logistic`` are deterministic,
    and so is XGBoost without row/column subsampling; seeds alone would then give
    identical members and a meaningless (zero) epistemic uncertainty.
    """
    if cfg.backend in ("hist_gb", "logistic"):
        return False
    if cfg.backend == "xgboost":
        p = cfg.params
        return float(p.get("subsample", 1.0)) < 1.0 or float(p.get("colsample_bytree", 1.0)) < 1.0
    return True  # random_forest bootstraps rows internally


class ProbabilisticEnsemble:
    """``M`` seeded members; exposes per-member and mean positive-class probabilities.

    Member diversity is guaranteed: for backends without internal randomness the
    seeded moving-block bootstrap is used even when ``bootstrap: none``, and an
    ensemble whose members still predict identically is rejected.
    """

    def __init__(self, name: str, cfg: ModelConfig, base_seed: int, n_jobs: int = 1) -> None:
        self.name = name
        self.cfg = cfg
        self.base_seed = int(base_seed)
        self.n_jobs = n_jobs
        self.members: list[Any] = []
        self.member_seeds: list[int] = []
        self.train_info: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, split: str, ledger: FitLedger,
            row_ids: np.ndarray | None = None) -> ProbabilisticEnsemble:
        """Fit all members on the training split (any other split is refused)."""
        if split != "train":
            raise LeakageError(f"{self.name}: models may only be fitted on train, got {split!r}")
        y = np.asarray(y).astype(np.int8)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"{self.name}: training labels contain a single class {classes}")
        self.members, self.member_seeds = [], []
        n = len(y)
        bootstrap = self.cfg.bootstrap
        if self.cfg.ensemble_size > 1 and bootstrap == "none" and not has_internal_randomness(
                self.cfg):
            bootstrap = "block"
            log.warning("%s: backend %r has no internal randomness; using the seeded block "
                        "bootstrap so that ensemble members differ", self.name, self.cfg.backend)
        for m in range(self.cfg.ensemble_size):
            seed = derive_seed(self.base_seed, self.name, "member", m)
            rng = np.random.default_rng(seed)
            idx = np.arange(n)
            if bootstrap == "block":
                for _attempt in range(10):
                    cand = moving_block_indices(n, self.cfg.block_size, rng)
                    if len(np.unique(y[cand])) == 2:
                        idx = cand
                        break
                else:
                    log.warning("%s member %d: block bootstrap lacked both classes; "
                                "using all rows", self.name, m)
            Xm, ym = X[idx], y[idx]
            n_pos = int(ym.sum())
            pos_weight = (len(ym) - n_pos) / max(n_pos, 1) if self.cfg.class_weight == "balanced" \
                else 1.0
            est = build_estimator(self.cfg, seed, self.n_jobs, pos_weight)
            if self.cfg.backend == "xgboost":
                est.fit(Xm, ym)
            else:
                weights = np.where(ym == 1, pos_weight, 1.0)
                est.fit(Xm, ym, sample_weight=weights)
            self.members.append(est)
            self.member_seeds.append(seed)
        if self.cfg.ensemble_size > 1:
            probe = self.predict_member_proba(X[: min(n, 5000)])
            if float(np.ptp(probe, axis=0).max()) < 1e-12:
                raise ValueError(
                    f"{self.name}: all {self.cfg.ensemble_size} ensemble members predict "
                    "identically, so epistemic uncertainty is undefined; use bootstrap: block "
                    "or a stochastic backend configuration")
        prevalence = float(y.mean())
        self.train_info = {"n_rows": n, "prevalence": prevalence,
                           "member_seeds": self.member_seeds, "backend": self.cfg.backend,
                           "effective_bootstrap": bootstrap}
        ledger.record(f"model:{self.name}", split, n,
                      index_fingerprint(row_ids) if row_ids is not None else None,
                      prevalence=prevalence, members=len(self.members))
        return self

    def predict_member_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability of every member, shape ``(M, n)``."""
        if not self.members:
            raise RuntimeError(f"{self.name} is not fitted")
        return np.vstack([m.predict_proba(X)[:, 1] for m in self.members]).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble-mean positive-class probability, shape ``(n,)``."""
        return self.predict_member_proba(X).mean(axis=0)

    def feature_importance(self, feature_names: list[str], top_k: int = 20) -> list[dict[str, Any]]:
        """Mean gain/impurity importance across members (tree backends only)."""
        mats = [getattr(m, "feature_importances_", None) for m in self.members]
        if any(v is None for v in mats):
            return []
        mean = np.mean(np.vstack(mats), axis=0)
        order = np.argsort(-mean, kind="mergesort")[:top_k]
        return [{"feature": feature_names[i], "importance": float(mean[i])} for i in order]
