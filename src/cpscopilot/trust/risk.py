"""Trust & risk layer: turns raw model outputs into an auditable assessment.

For every time step it produces (all calibrated on the calibration split):

* ``p_detect`` - calibrated probability that an attack is in progress,
* ``p_h<k>`` - calibrated probability of an attack within each horizon,
* uncertainty: epistemic (ensemble mutual information) and aleatoric,
* ``anomaly_evidence`` and ``consistency`` (cross-source agreement),
* ``criticality`` of the most critical implicated asset,
* ``risk = max(p_detect, max_h w_h * p_h) * criticality``,
* ``confidence = max(p*, 1 - p*) * (1 - epistemic_rel) * consistency`` where
  ``p*`` is the probability of the signal driving the risk,
* ``state`` in {benign, suspicious, malicious} from thresholds chosen on the
  calibration split (F1-optimal for *malicious*, recall-targeted for
  *suspicious*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cpscopilot.config import CalibrationConfig, TrustConfig
from cpscopilot.data.leakage import FitLedger
from cpscopilot.trust.asset import AssetRegistry
from cpscopilot.trust.calibration import Calibrator, make_calibrator
from cpscopilot.trust.evidence import RobustAnomalyScorer, evidence_consistency
from cpscopilot.trust.uncertainty import decompose_uncertainty

__all__ = ["ModelOutputs", "TrustRiskLayer", "select_state_thresholds", "STATES"]

DETECTION = "detection"
STATES = ("benign", "suspicious", "malicious")


@dataclass
class ModelOutputs:
    """Uncalibrated per-member probabilities for one split."""

    detect_members: np.ndarray  # (M, n)
    future_members: dict[str, np.ndarray]  # horizon key -> (M, n)

    def members(self, key: str) -> np.ndarray:
        return self.detect_members if key == DETECTION else self.future_members[key]


def select_state_thresholds(y: np.ndarray, p: np.ndarray, recall_target: float) -> dict[str, float]:
    """Choose (suspicious, malicious) thresholds on calibration data.

    ``malicious`` maximises F1; ``suspicious`` is the largest threshold whose
    recall still meets ``recall_target`` (capped at the malicious threshold).
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError("threshold selection needs both classes in the calibration split")
    precision, recall, thr = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    best = int(np.argmax(f1))
    tau_m = float(thr[best])
    eligible = thr[recall >= recall_target]
    tau_s = float(eligible.max()) if len(eligible) else float(thr.min())
    tau_s = min(tau_s, tau_m)
    return {"suspicious": tau_s, "malicious": tau_m, "f1_at_malicious": float(f1[best]),
            "recall_at_suspicious": float(recall[np.searchsorted(thr, tau_s, side="left")])}


class TrustRiskLayer:
    """Calibration, uncertainty, evidence, asset impact, risk and confidence."""

    def __init__(
        self,
        trust_cfg: TrustConfig,
        calibration_cfg: CalibrationConfig,
        horizon_minutes: dict[str, float],
        registry: AssetRegistry,
        scorer: RobustAnomalyScorer,
        base_cols: list[str],
    ) -> None:
        self.cfg = trust_cfg
        self.calibration_cfg = calibration_cfg
        self.horizon_minutes = dict(sorted(horizon_minutes.items(), key=lambda kv: kv[1]))
        self.horizon_keys = list(self.horizon_minutes)
        self.registry = registry
        self.scorer = scorer
        self.base_cols = list(base_cols)
        self.primary = calibration_cfg.method
        self.methods = sorted(set(calibration_cfg.compare_methods) | {self.primary})
        self.calibrators: dict[str, dict[str, Calibrator]] = {}
        self.epistemic_reference_: float | None = None
        self.thresholds_: dict[str, float] | None = None
        self.horizon_weights = {
            key: float(trust_cfg.horizon_weights.get(
                key, np.exp(-trust_cfg.default_horizon_decay * minutes)))
            for key, minutes in self.horizon_minutes.items()
        }

    @property
    def model_keys(self) -> list[str]:
        return [DETECTION, *self.horizon_keys]

    # ------------------------------------------------------------ fitting
    def fit_calibrators(self, outputs: ModelOutputs, labels: dict[str, np.ndarray],
                        ledger: FitLedger) -> TrustRiskLayer:
        """Fit every calibration method for every model on the calibration split."""
        for key in self.model_keys:
            p = outputs.members(key).mean(axis=0)
            y = np.asarray(labels[key], dtype=np.float64)
            mask = ~np.isnan(y)
            if len(np.unique(y[mask])) != 2:
                raise ValueError(f"calibration labels for {key} contain a single class")
            self.calibrators[key] = {
                m: make_calibrator(m).fit(p[mask], y[mask], "calibration", ledger, key)
                for m in self.methods
            }
        return self

    def calibrate(self, key: str, members: np.ndarray,
                  method: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(calibrated ensemble mean, calibrated members)``."""
        cal = self.calibrators[key][method or self.primary]
        return cal.transform(members.mean(axis=0)), cal.transform(members)

    def fit_reference(self, frame_cal: pd.DataFrame, y_now_cal: np.ndarray) -> TrustRiskLayer:
        """Calibration-split references: epistemic scale and state thresholds."""
        q = self.cfg.epistemic_reference_quantile
        self.epistemic_reference_ = max(float(np.quantile(frame_cal["epistemic"], q)), 1e-6)
        self.thresholds_ = select_state_thresholds(
            y_now_cal, frame_cal["p_detect"].to_numpy(), self.cfg.suspicious_recall_target)
        return self

    # --------------------------------------------------------- assessment
    def assess(self, outputs: ModelOutputs, current: pd.DataFrame) -> pd.DataFrame:
        """Per-row trust components that do not need calibration-split references."""
        cols: dict[str, Any] = {}
        det_raw = outputs.detect_members.mean(axis=0)
        det_cal, det_members = self.calibrate(DETECTION, outputs.detect_members)
        det_unc = decompose_uncertainty(det_members)
        cols.update(p_detect_raw=det_raw, p_detect=det_cal,
                    epi_detection=det_unc["epistemic"], alea_detection=det_unc["aleatoric"],
                    std_detection=det_unc["std"])
        future_cal = []
        epistemic = det_unc["epistemic"].copy()
        aleatoric_by_key = {DETECTION: det_unc["aleatoric"]}
        for key in self.horizon_keys:
            members = outputs.future_members[key]
            p_cal, m_cal = self.calibrate(key, members)
            unc = decompose_uncertainty(m_cal)
            cols[f"p_{key}_raw"] = members.mean(axis=0)
            cols[f"p_{key}"] = p_cal
            cols[f"epi_{key}"] = unc["epistemic"]
            cols[f"alea_{key}"] = unc["aleatoric"]
            future_cal.append(p_cal)
            epistemic = np.maximum(epistemic, unc["epistemic"])
            aleatoric_by_key[key] = unc["aleatoric"]
        future = np.vstack(future_cal) if future_cal else None

        z = self.scorer.z(current)
        a_score, a_evidence = self.scorer.evidence(z)
        attribution = self.registry.attribute(z, self.base_cols, self.cfg.anomaly_z_threshold)
        consistency = evidence_consistency(det_cal, a_evidence, future,
                                           self.cfg.consistency_weights)
        crit = attribution.criticality
        risk_now = det_cal * crit
        if future is not None:
            weights = np.array([self.horizon_weights[k] for k in self.horizon_keys])[:, None]
            weighted = weights * future
            peak = weighted.argmax(axis=0)
            risk_future = weighted.max(axis=0) * crit
            p_future_peak = future[peak, np.arange(future.shape[1])]
        else:
            peak = np.zeros(len(det_cal), dtype=int)
            risk_future = np.zeros_like(det_cal)
            p_future_peak = np.zeros_like(det_cal)
        driven_by_now = risk_now >= risk_future
        driver = np.where(driven_by_now, DETECTION,
                          np.array(self.horizon_keys or [DETECTION], dtype=object)[peak])
        p_driver = np.where(driven_by_now, det_cal, p_future_peak)
        if self.horizon_keys:
            future_alea = np.vstack([aleatoric_by_key[k] for k in self.horizon_keys])
            aleatoric_peak = future_alea[peak, np.arange(len(peak))]
        else:
            aleatoric_peak = det_unc["aleatoric"]
        aleatoric = np.where(driven_by_now, det_unc["aleatoric"], aleatoric_peak)
        cols.update(
            anomaly_score=a_score, anomaly_evidence=a_evidence,
            cross_source=consistency["cross_source"],
            temporal_coherence=consistency["temporal_coherence"],
            consistency=consistency["consistency"],
            asset=attribution.asset, criticality=crit, n_implicated=attribution.n_implicated,
            risk_now=risk_now, risk_future=risk_future, risk=np.maximum(risk_now, risk_future),
            risk_driver=driver, p_driver=p_driver, epistemic=epistemic, aleatoric=aleatoric,
        )
        return pd.DataFrame(cols, index=current.index)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add reference-dependent fields: relative epistemic, confidence, state."""
        if self.epistemic_reference_ is None or self.thresholds_ is None:
            raise RuntimeError("call fit_reference on the calibration split first")
        out = frame.copy()
        epi_rel = np.clip(out["epistemic"].to_numpy() / self.epistemic_reference_, 0.0, 1.0)
        p = out["p_driver"].to_numpy()
        calibrated_conf = np.maximum(p, 1.0 - p)
        out["epistemic_rel"] = epi_rel
        out["calibrated_confidence"] = calibrated_conf
        out["confidence"] = calibrated_conf * (1.0 - epi_rel) * out["consistency"].to_numpy()
        pd_ = out["p_detect"].to_numpy()
        tau_s, tau_m = self.thresholds_["suspicious"], self.thresholds_["malicious"]
        out["state"] = np.where(pd_ >= tau_m, "malicious",
                                np.where(pd_ >= tau_s, "suspicious", "benign"))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "primary_method": self.primary,
            "calibrators": {k: {m: c.state_dict() for m, c in v.items()}
                            for k, v in self.calibrators.items()},
            "epistemic_reference": self.epistemic_reference_,
            "state_thresholds": self.thresholds_,
            "horizon_weights": self.horizon_weights,
            "anomaly_scorer": self.scorer.state_dict(),
        }
