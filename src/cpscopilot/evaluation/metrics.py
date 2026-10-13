"""Imbalance-aware discrimination metrics and compact ROC/PR curves.

Accuracy is never reported alone for imbalanced attack data: the headline
metrics are PR-AUC (average precision), balanced accuracy, macro-F1 and MCC,
reported next to the attack prevalence (the PR-AUC of a no-skill classifier).
Decision thresholds are always selected on the calibration split and only
*applied* to the test split.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = ["classification_metrics", "compact_curves", "pr_auc", "roc_auc"]


def _two_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) == 2


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if _two_classes(y) else float("nan")


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if _two_classes(y) else float("nan")


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Threshold-free and thresholded metrics for binary attack detection."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out = {
            "n": int(len(y)),
            "n_positive": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else float("nan"),
            "pr_auc": pr_auc(y, p),
            "roc_auc": roc_auc(y, p),
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if _two_classes(y)
            else float("nan"),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "f1_attack": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)) if _two_classes(y) else float("nan"),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    return out


def compact_curves(y: np.ndarray, p: np.ndarray, n_points: int = 400) -> dict[str, Any]:
    """ROC and (interpolated) PR curves resampled onto fixed grids for storage."""
    y = np.asarray(y).astype(int)
    if not _two_classes(y):
        return {"available": False}
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.linspace(0.0, 1.0, n_points)
    tpr_grid = np.interp(grid, fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, p)
    order = np.argsort(recall, kind="mergesort")
    rec_sorted, prec_sorted = recall[order], precision[order]
    # interpolated precision: best precision achievable at recall >= r
    best_from_right = np.maximum.accumulate(prec_sorted[::-1])[::-1]
    idx = np.searchsorted(rec_sorted, grid, side="left").clip(0, len(rec_sorted) - 1)
    return {
        "available": True,
        "roc": {"fpr": grid.tolist(), "tpr": tpr_grid.tolist()},
        "pr": {"recall": grid.tolist(), "precision": best_from_right[idx].tolist()},
        "prevalence": float(y.mean()),
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
    }
