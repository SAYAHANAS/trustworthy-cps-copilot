"""Assistance gate: selects how much the copilot may do for each event.

Modes, from least to most assistance:

``monitor``          log only, no copilot output beyond a status line
``explain``          grounded summary + explanation + evidence, no actions
``recommend``        adds recommended investigation steps
``recommend_with_approval``  steps that each require explicit analyst approval
``abstain``          the copilot declines (insufficient trust); the raw
                     evidence is routed to the analyst for manual review

Rules are evaluated in order; the first match wins:

=====  ============================  ===========================================
R1     abstain_epistemic             epistemic > max (model outside competence)
R2     abstain_inconsistent          consistency < min and risk >= explain_min
R3     approval_high_risk            risk >= approval_min_risk
R4     approval_critical_asset       criticality >= min and risk >= recommend_min
R5     recommend                     risk >= recommend_min and confidence >= min
R6     explain                       risk >= explain_min or state == malicious
R7     monitor                       otherwise
=====  ============================  ===========================================

*Suspicious* rows below ``explain_min_risk`` stay in monitor (they are logged
with their state); explaining every high-recall *suspicious* row would flood
analysts with low-value alerts.

Analyst context adapts the thresholds transparently (``adapt``), e.g. junior
analysts receive more explanations and every recommendation requires approval;
overloaded analysts receive fewer low-risk explanations. The batch and
per-event paths share one vectorised implementation, so they cannot diverge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from cpscopilot.config import AnalystProfile, GateConfig

__all__ = ["AssistanceGate", "AssistanceMode", "GateDecision", "GateThresholds", "RULES"]


class AssistanceMode(StrEnum):
    MONITOR = "monitor"
    EXPLAIN = "explain"
    RECOMMEND = "recommend"
    RECOMMEND_WITH_APPROVAL = "recommend_with_approval"
    ABSTAIN = "abstain"

    @classmethod
    def ordered(cls) -> list[AssistanceMode]:
        return [cls.MONITOR, cls.EXPLAIN, cls.RECOMMEND, cls.RECOMMEND_WITH_APPROVAL, cls.ABSTAIN]

    @property
    def allows_recommendations(self) -> bool:
        return self in (AssistanceMode.RECOMMEND, AssistanceMode.RECOMMEND_WITH_APPROVAL)

    @property
    def allows_explanation(self) -> bool:
        return self not in (AssistanceMode.MONITOR, AssistanceMode.ABSTAIN)


RULES: dict[str, AssistanceMode] = {
    "R1_abstain_epistemic": AssistanceMode.ABSTAIN,
    "R2_abstain_inconsistent": AssistanceMode.ABSTAIN,
    "R3_approval_high_risk": AssistanceMode.RECOMMEND_WITH_APPROVAL,
    "R4_approval_critical_asset": AssistanceMode.RECOMMEND_WITH_APPROVAL,
    "R5_recommend": AssistanceMode.RECOMMEND,
    "R5b_recommend_junior_approval": AssistanceMode.RECOMMEND_WITH_APPROVAL,
    "R6_explain": AssistanceMode.EXPLAIN,
    "R7_monitor": AssistanceMode.MONITOR,
}


@dataclass(frozen=True)
class GateThresholds:
    """Resolved numeric thresholds (after data-driven and analyst adaptation)."""

    abstain_max_epistemic: float
    abstain_min_consistency: float
    approval_min_risk: float
    approval_min_criticality: float
    recommend_min_risk: float
    recommend_min_confidence: float
    explain_min_risk: float
    recommendations_require_approval: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    """Mode for one event plus the rule that fired and human-readable reasons."""

    mode: AssistanceMode
    rule_id: str
    reasons: tuple[str, ...]
    thresholds: GateThresholds

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "rule_id": self.rule_id, "reasons": list(self.reasons),
                "thresholds": self.thresholds.as_dict()}


class AssistanceGate:
    """Ordered, auditable policy over trust-layer outputs."""

    REQUIRED = ("epistemic", "consistency", "risk", "criticality", "confidence", "state")

    def __init__(self, cfg: GateConfig) -> None:
        self.cfg = cfg
        self.base_: GateThresholds | None = None
        self.resolution_: dict[str, Any] = {}

    # ------------------------------------------------------------ thresholds
    def fit(self, trust_cal: pd.DataFrame, benign_mask: np.ndarray | None = None) -> AssistanceGate:
        """Resolve data-driven thresholds on the calibration split.

        * the epistemic abstention threshold is a quantile of calibration epistemic
          uncertainty (unless fixed in the config);
        * with ``benign_mask`` (calibration rows with no attack now or within the
          longest horizon), the explain/recommend risk thresholds are raised to the
          quantiles that keep at most ``*_benign_budget`` of benign rows at or above
          that mode - an explicit analyst alert budget. Configured values act as floors.
        """
        cfg = self.cfg
        max_epi = cfg.abstain_max_epistemic
        if max_epi is None:
            max_epi = float(np.quantile(trust_cal["epistemic"], cfg.abstain_epistemic_quantile))
        explain, recommend, approval = (cfg.explain_min_risk, cfg.recommend_min_risk,
                                        cfg.approval_min_risk)
        resolution: dict[str, Any] = {"explain": "floor", "recommend": "floor"}
        if benign_mask is not None and np.any(benign_mask):
            benign_risk = trust_cal["risk"].to_numpy()[np.asarray(benign_mask, dtype=bool)]
            for name, budget in (("explain", cfg.explain_benign_budget),
                                 ("recommend", cfg.recommend_benign_budget)):
                if budget is None:
                    continue
                q = float(np.quantile(benign_risk, 1.0 - budget))
                if name == "explain" and q > explain:
                    explain, resolution[name] = q, f"benign budget {budget:g}"
                if name == "recommend" and q > recommend:
                    recommend, resolution[name] = q, f"benign budget {budget:g}"
        recommend = max(recommend, explain)
        approval = max(approval, recommend)
        self.resolution_ = {**resolution, "n_benign_calibration_rows":
                            int(np.sum(benign_mask)) if benign_mask is not None else None}
        self.base_ = GateThresholds(
            abstain_max_epistemic=float(max_epi),
            abstain_min_consistency=cfg.abstain_min_consistency,
            approval_min_risk=float(approval),
            approval_min_criticality=cfg.approval_min_criticality,
            recommend_min_risk=float(recommend),
            recommend_min_confidence=cfg.recommend_min_confidence,
            explain_min_risk=float(explain),
        )
        return self

    def adapt(self, ctx: AnalystProfile) -> tuple[GateThresholds, list[str]]:
        """Analyst-adaptive thresholds and a log of the adjustments made."""
        if self.base_ is None:
            raise RuntimeError("gate is not fitted")
        t = self.base_
        notes: list[str] = []
        explain, recommend_conf = t.explain_min_risk, t.recommend_min_confidence
        approval, require_approval = t.approval_min_risk, False
        if ctx.expertise == "junior":
            explain *= self.cfg.junior_explain_factor
            notes.append(f"junior: explain_min_risk x{self.cfg.junior_explain_factor}")
            if self.cfg.junior_requires_approval:
                require_approval = True
                notes.append("junior: every recommendation requires approval")
        elif ctx.expertise == "senior":
            recommend_conf += self.cfg.senior_confidence_offset
            notes.append(f"senior: recommend_min_confidence {self.cfg.senior_confidence_offset:+}")
        if ctx.workload >= self.cfg.high_workload_threshold:
            explain *= self.cfg.high_workload_explain_factor
            notes.append(f"workload {ctx.workload:.2f}: explain_min_risk "
                         f"x{self.cfg.high_workload_explain_factor}")
        if ctx.shift == "night":
            approval *= self.cfg.night_approval_factor
            notes.append(f"night shift: approval_min_risk x{self.cfg.night_approval_factor}")
        if ctx.override_rate >= self.cfg.override_rate_threshold:
            recommend_conf += self.cfg.override_confidence_offset
            notes.append(f"override rate {ctx.override_rate:.2f}: recommend_min_confidence "
                         f"+{self.cfg.override_confidence_offset}")
        recommend = t.recommend_min_risk
        approval = float(np.clip(max(approval, recommend), 0.0, 1.0))
        explain = float(np.clip(min(explain, recommend), 0.0, 1.0))
        adapted = replace(
            t,
            explain_min_risk=explain,
            approval_min_risk=approval,
            recommend_min_confidence=float(np.clip(recommend_conf, 0.0, 1.0)),
            recommendations_require_approval=require_approval,
        )
        return adapted, notes

    # -------------------------------------------------------------- decisions
    @staticmethod
    def _rules(frame: pd.DataFrame, t: GateThresholds) -> np.ndarray:
        epi = frame["epistemic"].to_numpy(dtype=np.float64)
        cons = frame["consistency"].to_numpy(dtype=np.float64)
        risk = frame["risk"].to_numpy(dtype=np.float64)
        crit = frame["criticality"].to_numpy(dtype=np.float64)
        conf = frame["confidence"].to_numpy(dtype=np.float64)
        malicious = frame["state"].to_numpy() == "malicious"
        recommend_id = "R5b_recommend_junior_approval" if t.recommendations_require_approval \
            else "R5_recommend"
        conditions = [
            epi > t.abstain_max_epistemic,
            (cons < t.abstain_min_consistency) & (risk >= t.explain_min_risk),
            risk >= t.approval_min_risk,
            (crit >= t.approval_min_criticality) & (risk >= t.recommend_min_risk),
            (risk >= t.recommend_min_risk) & (conf >= t.recommend_min_confidence),
            (risk >= t.explain_min_risk) | malicious,
        ]
        choices = ["R1_abstain_epistemic", "R2_abstain_inconsistent", "R3_approval_high_risk",
                   "R4_approval_critical_asset", recommend_id, "R6_explain"]
        return np.select(conditions, choices, default="R7_monitor")

    def decide_batch(self, trust: pd.DataFrame, ctx: AnalystProfile) -> pd.DataFrame:
        """Vectorised decisions; returns ``mode`` and ``rule_id`` per row."""
        missing = [c for c in self.REQUIRED if c not in trust]
        if missing:
            raise KeyError(f"trust frame lacks columns {missing}")
        thresholds, _ = self.adapt(ctx)
        rule_ids = self._rules(trust, thresholds)
        modes = np.array([RULES[r].value for r in rule_ids], dtype=object)
        return pd.DataFrame({"mode": modes, "rule_id": rule_ids}, index=trust.index)

    def decide(self, event: pd.Series | dict[str, Any], ctx: AnalystProfile) -> GateDecision:
        """Decision for one event, with reasons naming the values and thresholds."""
        row = pd.DataFrame([dict(event)])
        thresholds, notes = self.adapt(ctx)
        rule_id = str(self._rules(row, thresholds)[0])
        r = row.iloc[0]
        t = thresholds
        reasons = {
            "R1_abstain_epistemic": f"epistemic uncertainty {r.epistemic:.3f} bits exceeds "
                                    f"{t.abstain_max_epistemic:.3f}",
            "R2_abstain_inconsistent": f"evidence consistency {r.consistency:.2f} below "
                                       f"{t.abstain_min_consistency:.2f} at risk {r.risk:.2f}",
            "R3_approval_high_risk": f"risk {r.risk:.2f} >= {t.approval_min_risk:.2f}",
            "R4_approval_critical_asset": f"asset criticality {r.criticality:.2f} >= "
                                          f"{t.approval_min_criticality:.2f} and risk "
                                          f"{r.risk:.2f} >= {t.recommend_min_risk:.2f}",
            "R5_recommend": f"risk {r.risk:.2f} >= {t.recommend_min_risk:.2f} and confidence "
                            f"{r.confidence:.2f} >= {t.recommend_min_confidence:.2f}",
            "R5b_recommend_junior_approval": f"risk {r.risk:.2f} >= {t.recommend_min_risk:.2f} "
                                             "and analyst profile requires approval",
            "R6_explain": f"risk {r.risk:.2f} >= {t.explain_min_risk:.2f} or state "
                          f"{r.state!r} is malicious",
            "R7_monitor": f"risk {r.risk:.2f} below {t.explain_min_risk:.2f} and state "
                          f"{r.state!r} is not malicious",
        }
        return GateDecision(mode=RULES[rule_id], rule_id=rule_id,
                            reasons=(reasons[rule_id], *notes), thresholds=thresholds)
