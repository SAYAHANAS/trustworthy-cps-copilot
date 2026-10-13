"""Uncertainty, evidence consistency, asset impact, thresholds and the assistance gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpscopilot.config import AnalystProfile, AssetRule, GateConfig
from cpscopilot.data.leakage import FitLedger, LeakageError
from cpscopilot.gate.policy import AssistanceGate, AssistanceMode
from cpscopilot.trust.asset import AssetRegistry
from cpscopilot.trust.evidence import RobustAnomalyScorer, evidence_consistency
from cpscopilot.trust.risk import select_state_thresholds
from cpscopilot.trust.uncertainty import decompose_uncertainty

BASELINE = AnalystProfile()


def test_uncertainty_decomposition_extremes() -> None:
    agree = decompose_uncertainty(np.array([[0.5], [0.5], [0.5]]))
    assert agree["epistemic"][0] == pytest.approx(0.0, abs=1e-9)
    assert agree["aleatoric"][0] == pytest.approx(1.0)
    disagree = decompose_uncertainty(np.array([[1e-9], [1 - 1e-9]]))
    assert disagree["epistemic"][0] == pytest.approx(1.0, abs=1e-6)
    assert disagree["aleatoric"][0] == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    probs = rng.random((5, 100))
    parts = decompose_uncertainty(probs)
    np.testing.assert_allclose(parts["epistemic"] + parts["aleatoric"], parts["total"])


def test_evidence_consistency() -> None:
    out = evidence_consistency(np.array([0.9, 0.9]), np.array([0.9, 0.1]),
                               np.array([[0.2, 0.8], [0.6, 0.3]]),
                               {"cross_source": 1.0, "temporal_coherence": 1.0})
    assert out["cross_source"] == pytest.approx([1.0, 0.2])
    assert out["temporal_coherence"] == pytest.approx([1.0, 0.5])  # 0.8 -> 0.3 violates nesting
    assert out["consistency"] == pytest.approx([1.0, 0.35])


def test_anomaly_scorer_is_train_only_and_tail_focused() -> None:
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(0, 1, (2000, 4)), columns=list("abcd"))
    y = np.zeros(2000, dtype=int)
    with pytest.raises(LeakageError):
        RobustAnomalyScorer().fit(train, y, "test", FitLedger())
    scorer = RobustAnomalyScorer(0.95).fit(train, y, "train", FitLedger())
    typical = pd.DataFrame(rng.normal(0, 1, (500, 4)), columns=list("abcd"))
    extreme = typical + 8.0
    assert np.mean(scorer.evidence(scorer.z(typical))[1] == 0) > 0.9
    assert np.all(scorer.evidence(scorer.z(extreme))[1] == 1.0)


def test_asset_attribution_picks_most_critical_implicated_asset() -> None:
    reg = AssetRegistry([AssetRule(pattern="^P1", asset="pump", criticality=0.4),
                         AssetRule(pattern="^D2", asset="dosing", criticality=0.9)], 0.5)
    z = np.array([[5.0, 4.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    att = reg.attribute(z, ["P101", "D201", "X9"], 3.0)
    assert list(att.asset) == ["dosing", "pump", "unattributed"]
    assert list(att.criticality) == [0.9, 0.4, 0.5]


def test_state_thresholds_from_calibration_data() -> None:
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.1).astype(int)
    p = np.clip(0.7 * y + rng.normal(0.15, 0.12, 5000), 0, 1)
    t = select_state_thresholds(y, p, recall_target=0.95)
    assert 0 < t["suspicious"] <= t["malicious"] < 1
    assert np.mean(p[y == 1] >= t["suspicious"]) >= 0.95


def event(**kw) -> dict:
    base = {"epistemic": 0.001, "consistency": 0.9, "risk": 0.05, "criticality": 0.5,
            "confidence": 0.9, "state": "benign"}
    return {**base, **kw}


@pytest.fixture()
def gate() -> AssistanceGate:
    cal = pd.DataFrame({"epistemic": np.linspace(0, 0.05, 1000), "risk": np.linspace(0, 1, 1000)})
    return AssistanceGate(GateConfig()).fit(cal)


@pytest.mark.parametrize("kw, mode, rule", [
    ({"epistemic": 0.2, "risk": 0.9}, AssistanceMode.ABSTAIN, "R1_abstain_epistemic"),
    ({"consistency": 0.1, "risk": 0.5}, AssistanceMode.ABSTAIN, "R2_abstain_inconsistent"),
    ({"consistency": 0.1, "risk": 0.01}, AssistanceMode.MONITOR, "R7_monitor"),
    ({"risk": 0.7}, AssistanceMode.RECOMMEND_WITH_APPROVAL, "R3_approval_high_risk"),
    ({"risk": 0.4, "criticality": 0.9}, AssistanceMode.RECOMMEND_WITH_APPROVAL,
     "R4_approval_critical_asset"),
    ({"risk": 0.4}, AssistanceMode.RECOMMEND, "R5_recommend"),
    ({"risk": 0.4, "confidence": 0.3}, AssistanceMode.EXPLAIN, "R6_explain"),
    ({"risk": 0.15}, AssistanceMode.EXPLAIN, "R6_explain"),
    ({"risk": 0.05, "state": "malicious"}, AssistanceMode.EXPLAIN, "R6_explain"),
    ({"risk": 0.05, "state": "suspicious"}, AssistanceMode.MONITOR, "R7_monitor"),
])
def test_gate_rules(gate, kw, mode, rule) -> None:
    decision = gate.decide(event(**kw), BASELINE)
    assert decision.mode == mode and decision.rule_id == rule
    assert decision.reasons


def test_junior_analyst_gets_more_explanations_and_approval(gate) -> None:
    junior = AnalystProfile(expertise="junior")
    assert gate.decide(event(risk=0.4), junior).mode == AssistanceMode.RECOMMEND_WITH_APPROVAL
    assert gate.decide(event(risk=0.08), BASELINE).mode == AssistanceMode.MONITOR
    assert gate.decide(event(risk=0.08), junior).mode == AssistanceMode.EXPLAIN


def test_overloaded_analyst_gets_fewer_low_risk_explanations(gate) -> None:
    busy = AnalystProfile(workload=0.95)
    assert gate.decide(event(risk=0.12), BASELINE).mode == AssistanceMode.EXPLAIN
    assert gate.decide(event(risk=0.12), busy).mode == AssistanceMode.MONITOR
    assert gate.decide(event(risk=0.7), busy).mode == AssistanceMode.RECOMMEND_WITH_APPROVAL


def test_batch_and_single_decisions_agree(gate) -> None:
    rng = np.random.default_rng(3)
    n = 500
    frame = pd.DataFrame({
        "epistemic": rng.exponential(0.02, n), "consistency": rng.random(n),
        "risk": rng.random(n), "criticality": rng.random(n), "confidence": rng.random(n),
        "state": rng.choice(["benign", "suspicious", "malicious"], n)})
    for profile in (BASELINE, AnalystProfile(expertise="junior", shift="night"),
                    AnalystProfile(expertise="senior", workload=0.9, override_rate=0.5)):
        batch = gate.decide_batch(frame, profile)
        single = [gate.decide(frame.iloc[i], profile).mode.value for i in range(n)]
        assert batch["mode"].tolist() == single


def test_alert_budget_raises_thresholds_on_calibration_benign_rows() -> None:
    rng = np.random.default_rng(4)
    cal = pd.DataFrame({"epistemic": rng.exponential(0.01, 4000),
                        "risk": rng.beta(2, 5, 4000)})
    benign = np.ones(4000, dtype=bool)
    gate = AssistanceGate(GateConfig(explain_benign_budget=0.1,
                                     recommend_benign_budget=0.02)).fit(cal, benign)
    t = gate.base_
    assert np.mean(cal["risk"] >= t.explain_min_risk) == pytest.approx(0.1, abs=0.005)
    assert np.mean(cal["risk"] >= t.recommend_min_risk) == pytest.approx(0.02, abs=0.005)
    assert t.explain_min_risk <= t.recommend_min_risk <= t.approval_min_risk


def test_deterministic_backends_still_get_diverse_members() -> None:
    from cpscopilot.config import ModelConfig
    from cpscopilot.models.ensemble import ProbabilisticEnsemble

    rng = np.random.default_rng(5)
    X = rng.normal(size=(1500, 6))
    y = (X[:, 0] + 0.5 * rng.normal(size=1500) > 1.0).astype(int)
    for cfg in (ModelConfig(backend="hist_gb", ensemble_size=3, block_size=50,
                            params={"n_estimators": 20, "max_depth": 3}),
                ModelConfig(backend="xgboost", ensemble_size=3, block_size=50,
                            params={"n_estimators": 20, "max_depth": 3, "subsample": 1.0,
                                    "colsample_bytree": 1.0})):
        ens = ProbabilisticEnsemble("m", cfg, 0).fit(X, y, "train", FitLedger())
        assert ens.train_info["effective_bootstrap"] == "block"
        assert np.ptp(ens.predict_member_proba(X), axis=0).max() > 1e-6
    # a single block covering every row is a permutation: members stay identical -> refused
    same = ModelConfig(backend="logistic", ensemble_size=3, bootstrap="block", block_size=1500)
    with pytest.raises(ValueError, match="identically"):
        ProbabilisticEnsemble("m", same, 0).fit(X, y, "train", FitLedger())


def test_identical_members_have_exactly_zero_epistemic() -> None:
    p = np.array([0.1, 0.37, 0.5, 0.83, 0.99])
    parts = decompose_uncertainty(np.vstack([p, p, p]))
    assert np.all(parts["epistemic"] == 0.0)


def test_step_function_calibrators_are_not_applied_member_wise() -> None:
    from cpscopilot.config import CalibrationConfig, TrustConfig
    from cpscopilot.trust.calibration import make_calibrator
    from cpscopilot.trust.risk import TrustRiskLayer

    rng = np.random.default_rng(6)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    layer = TrustRiskLayer(TrustConfig(), CalibrationConfig(method="isotonic"), {},
                           AssetRegistry([], 0.5), RobustAnomalyScorer(), [])
    iso = make_calibrator("isotonic").fit(p, y, "calibration", FitLedger(), "detection")
    temp = make_calibrator("temperature").fit(p, y, "calibration", FitLedger(), "detection")
    layer.calibrators = {"detection": {"isotonic": iso, "temperature": temp}}
    members = np.vstack([p, np.clip(p + 0.05, 0, 1), np.clip(p - 0.05, 0, 1)])
    mean_iso, for_unc = layer.calibrate("detection", members, "isotonic")
    np.testing.assert_array_equal(for_unc, members)  # raw members keep their disagreement
    np.testing.assert_allclose(mean_iso, iso.transform(members.mean(axis=0)))
    _, mapped = layer.calibrate("detection", members, "temperature")
    np.testing.assert_allclose(mapped, temp.transform(members))  # smooth maps stay member-wise


def test_alert_budget_holds_with_tied_risks() -> None:
    risk = np.r_[np.zeros(800), np.full(90, 0.12063), np.linspace(0.13, 1.0, 110)]
    cal = pd.DataFrame({"epistemic": np.zeros(1000), "risk": risk})
    gate = AssistanceGate(GateConfig(explain_benign_budget=0.10,
                                     recommend_benign_budget=0.02)).fit(cal, np.ones(1000, bool))
    assert np.mean(risk >= gate.base_.explain_min_risk) <= 0.10
    assert np.mean(risk >= gate.base_.recommend_min_risk) <= 0.02
