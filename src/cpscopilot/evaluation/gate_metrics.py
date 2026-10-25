"""Operational metrics of the assistance gate and selective prediction.

Ground-truth categories per time step:

* ``ongoing``  - an attack is in progress (``y_now = 1``),
* ``imminent`` - no attack now, but one occurs within the longest horizon,
* ``benign``   - neither.

Reported quantities include the mode distribution per category, attack
coverage (attack rows routed to any mode other than *monitor*), the alert
burden on analysts (approval/abstain rows and alert episodes per hour), the
abstention quality (decision error rate inside vs. outside abstentions) and the
risk-coverage curve / AURC of a confidence score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cpscopilot.gate.policy import AssistanceMode

__all__ = ["gate_metrics", "ground_truth_category", "risk_coverage"]

ATTENTION_MODES = {AssistanceMode.RECOMMEND_WITH_APPROVAL.value, AssistanceMode.ABSTAIN.value}


def ground_truth_category(y_now: np.ndarray, y_future_any: np.ndarray) -> np.ndarray:
    return np.where(y_now == 1, "ongoing", np.where(y_future_any == 1, "imminent", "benign"))


def risk_coverage(errors: np.ndarray, confidence: np.ndarray,
                  n_points: int = 101) -> dict[str, Any]:
    """Selective risk (error rate among the most confident) vs. coverage; AURC."""
    errors = np.asarray(errors, dtype=np.float64)
    order = np.argsort(-np.asarray(confidence), kind="mergesort")
    cum_err = np.cumsum(errors[order])
    n = len(errors)
    counts = np.arange(1, n + 1)
    selective = cum_err / counts
    aurc = float(selective.mean()) if n else float("nan")
    grid = np.linspace(1.0 / max(n, 1), 1.0, n_points)
    idx = np.clip(np.ceil(grid * n).astype(int) - 1, 0, max(n - 1, 0))
    optimal = np.sort(errors)  # oracle ordering: correct first
    optimal_aurc = float((np.cumsum(optimal) / counts).mean()) if n else float("nan")
    return {"coverage": grid.tolist(), "selective_risk": selective[idx].tolist(),
            "aurc": aurc, "optimal_aurc": optimal_aurc, "excess_aurc": aurc - optimal_aurc,
            "full_coverage_risk": float(errors.mean()) if n else float("nan")}


def _episodes(active: np.ndarray) -> int:
    a = active.astype(int)
    return int(np.sum(np.diff(np.r_[0, a]) == 1))


def gate_metrics(
    modes: np.ndarray,
    y_now: np.ndarray,
    y_future_any: np.ndarray,
    decision_errors: np.ndarray,
    hours: float,
) -> dict[str, Any]:
    """Distribution, coverage, burden and abstention quality of gate decisions."""
    modes = np.asarray(modes).astype(str)
    category = ground_truth_category(np.asarray(y_now), np.asarray(y_future_any))
    order = [m.value for m in AssistanceMode.ordered()]
    table = pd.crosstab(pd.Series(category, name="category"), pd.Series(modes, name="mode"))
    table = table.reindex(index=["benign", "imminent", "ongoing"], columns=order, fill_value=0)
    attack = category != "benign"
    routed = modes != AssistanceMode.MONITOR.value
    attention = np.isin(modes, list(ATTENTION_MODES))
    abstain = modes == AssistanceMode.ABSTAIN.value
    errors = np.asarray(decision_errors, dtype=bool)
    counts = {m: int((modes == m).sum()) for m in order}
    n = len(modes)

    def frac(mask: np.ndarray, within: np.ndarray) -> float:
        return float(mask[within].mean()) if within.any() else float("nan")

    precision_by_mode = {m: frac(attack, modes == m) for m in order}
    err_abstain = frac(errors, abstain)
    err_other = frac(errors, ~abstain)
    return {
        "n": n,
        "mode_counts": counts,
        "mode_fractions": {m: c / n if n else float("nan") for m, c in counts.items()},
        "crosstab": {cat: {m: int(table.loc[cat, m]) for m in order} for cat in table.index},
        "attack_coverage": frac(routed, attack),
        "ongoing_coverage": frac(routed, category == "ongoing"),
        "imminent_coverage": frac(routed, category == "imminent"),
        "missed_attack_rate": frac(~routed, attack),
        "benign_attention_rate": frac(attention, ~attack),
        "attack_precision_by_mode": precision_by_mode,
        "attention_rows_per_hour": float(attention.sum() / hours) if hours > 0 else float("nan"),
        "alert_episodes_per_hour": float(_episodes(routed) / hours) if hours > 0
        else float("nan"),
        "attention_episodes_per_hour": float(_episodes(attention) / hours) if hours > 0
        else float("nan"),
        "abstain_rate": float(abstain.mean()) if n else float("nan"),
        "decision_error_rate_abstained": err_abstain,
        "decision_error_rate_not_abstained": err_other,
        "abstention_error_ratio": err_abstain / err_other
        if err_other and np.isfinite(err_abstain) and err_other > 0 else float("nan"),
    }
