"""RAG copilot: retrieval, structured output, guardrails, audit and LLM backends."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from cpscopilot.config import AnalystProfile, AssetRule, CopilotConfig, GateConfig, LLMConfig
from cpscopilot.copilot.audit import audit_response
from cpscopilot.copilot.copilot import CognitiveCopilot, enforce_policy
from cpscopilot.copilot.knowledge_base import load_knowledge_base
from cpscopilot.copilot.llm import (
    AnthropicLLM,
    CachedLLM,
    FaultInjectingLLM,
    LLMError,
    LLMRefusalError,
    LLMRequest,
    LLMResult,
    TemplateLLM,
)
from cpscopilot.copilot.retriever import BM25Retriever
from cpscopilot.copilot.schemas import CopilotResponse, strict_json_schema
from cpscopilot.gate.policy import AssistanceGate
from cpscopilot.trust.asset import AssetRegistry

HORIZONS = {"h5": 5.0, "h15": 15.0, "h30": 30.0}


KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"
BASELINE = AnalystProfile()


@pytest.fixture(scope="module")
def chunks():
    return load_knowledge_base(KB_DIR)


def trust_row(**kw: Any) -> pd.Series:
    base = {"state": "malicious", "p_detect": 0.91234, "p_h5": 0.95, "p_h15": 0.97,
            "p_h30": 0.98, "risk": 0.72, "risk_driver": "detection", "confidence": 0.81,
            "calibrated_confidence": 0.91, "epistemic": 0.004, "epistemic_rel": 0.2,
            "aleatoric": 0.3, "consistency": 0.88, "anomaly_evidence": 0.95,
            "asset": "control_network", "criticality": 0.8}
    return pd.Series({**base, **kw})


class SpyLLM:
    """LLM double returning a fixed payload (or failing) and counting calls."""

    name, model = "spy", "spy-1"

    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        self.payload, self.error, self.calls = payload, error, 0

    def generate(self, request: LLMRequest) -> LLMResult:
        self.calls += 1
        if self.error:
            raise self.error
        text = self.payload(request.packet) if callable(self.payload) else self.payload
        return LLMResult(text=text, backend=self.name, model=self.model)


def make_copilot(chunks, llm) -> CognitiveCopilot:
    registry = AssetRegistry([AssetRule(pattern="^NET_", asset="control_network",
                                        criticality=0.8, description="Control network")], 0.5)
    return CognitiveCopilot(CopilotConfig(), BM25Retriever(chunks), llm, registry, HORIZONS)


def packet_for(copilot: CognitiveCopilot, row: pd.Series, profile=BASELINE):
    gate = AssistanceGate(GateConfig()).fit(pd.DataFrame({"epistemic": [0.0, 0.01, 0.02]}))
    decision = gate.decide(row, profile)
    devs = [{"feature": "NET_PKTS", "z": 9.5, "value": 4321.0, "asset": "control_network"}]
    return copilot.build_packet("test-1", "2026-01-05 10:00:00", row, decision, profile, devs)


def test_knowledge_base_chunks_are_unique_and_stable(chunks) -> None:
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) and len(ids) > 20
    again = load_knowledge_base(KB_DIR)
    assert ids == [c.chunk_id for c in again]
    assert any(c.requires_approval for c in chunks)


def test_bm25_retrieval_is_relevant_and_deterministic(chunks) -> None:
    retriever = BM25Retriever(chunks)
    hits = retriever.search("network traffic flooding denial of service packets stale", 3)
    assert hits[0].chunk.doc_id == "PB-DOS"
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in
                                               retriever.search("network traffic flooding "
                                                                "denial of service packets "
                                                                "stale", 3)]


def test_packet_never_contains_ground_truth(chunks) -> None:
    copilot = make_copilot(chunks, TemplateLLM())
    packet = packet_for(copilot, trust_row())
    text = json.dumps(packet)
    for forbidden in ('"label"', "attack_type", "campaign_id", '"phase"', "y_now"):
        assert forbidden not in text


def test_template_output_is_valid_grounded_and_mode_compliant(chunks) -> None:
    copilot = make_copilot(chunks, TemplateLLM())
    result = copilot.respond(packet_for(copilot, trust_row()))
    assert result.llm_called and result.schema_valid
    CopilotResponse.model_validate(result.response)
    assert result.mode == "recommend_with_approval"
    steps = result.response["recommended_steps"]
    assert steps and all(s["requires_human_approval"] for s in steps)
    assert result.audit_raw["groundedness"] == pytest.approx(1.0)
    assert result.audit_raw["fully_grounded"]
    assert result.response["confidence"]["score"] == 0.81


def test_abstain_mode_never_calls_the_llm(chunks) -> None:
    spy = SpyLLM(error=AssertionError("must not be called"))
    copilot = make_copilot(chunks, spy)
    result = copilot.respond(packet_for(copilot, trust_row(epistemic=0.5)))
    assert result.mode == "abstain" and not result.llm_called and spy.calls == 0
    assert result.response["abstained"] and not result.response["recommended_steps"]


def test_llm_errors_and_invalid_json_fall_back_to_abstention(chunks) -> None:
    for llm in (SpyLLM(error=LLMError("boom")), SpyLLM(payload="not json")):
        copilot = make_copilot(chunks, llm)
        result = copilot.respond(packet_for(copilot, trust_row()))
        assert result.response["abstained"]
        assert result.violations and result.error


def test_guardrails_enforce_the_gate(chunks) -> None:
    copilot = make_copilot(chunks, TemplateLLM())
    packet = packet_for(copilot, trust_row(risk=0.15, state="benign"))  # explain mode
    assert packet["gate"]["mode"] == "explain"
    good = TemplateLLM().build(copy.deepcopy(packet))
    bad = copy.deepcopy(good)
    chunk_id = packet["knowledge"][0]["chunk_id"]
    bad["recommended_steps"] = [{"order": 1, "action": "Automatically block the PLC.",
                                 "rationale": "fast", "requires_human_approval": False,
                                 "citations": [chunk_id]}]
    bad["evidence"][0]["citations"] = ["PB-FAKE#x"]
    bad["evidence"][0]["signal_refs"] = ["not_a_signal"]
    bad["confidence"]["score"] = 0.99
    bad["mode"] = "recommend"
    guarded, violations = enforce_policy(bad, packet, max_steps=5)
    assert guarded["recommended_steps"] == []
    assert guarded["mode"] == "explain"
    assert guarded["confidence"]["score"] == packet["signals"]["confidence"]
    assert "PB-FAKE#x" not in guarded["evidence"][0]["citations"]
    for v in ("steps_not_permitted", "fabricated_citation", "invalid_signal_ref",
              "confidence_mismatch", "mode_mismatch"):
        assert v in violations


def test_guardrails_force_approval_and_drop_uncited_steps(chunks) -> None:
    copilot = make_copilot(chunks, TemplateLLM())
    packet = packet_for(copilot, trust_row(risk=0.4, criticality=0.5))  # recommend
    assert packet["gate"]["mode"] == "recommend"
    cid = packet["knowledge"][0]["chunk_id"]
    response = TemplateLLM().build(copy.deepcopy(packet))
    response["recommended_steps"] = [
        {"order": 1, "action": "Immediately disconnect the controller.", "rationale": "x",
         "requires_human_approval": False, "citations": [cid]},
        {"order": 2, "action": "Check automatic control loops for anomalies.",
         "rationale": "x", "requires_human_approval": False, "citations": [cid]},
        {"order": 3, "action": "Do something uncited.", "rationale": "x",
         "requires_human_approval": True, "citations": []},
    ]
    guarded, violations = enforce_policy(response, packet, max_steps=5)
    steps = guarded["recommended_steps"]
    assert len(steps) == 2 and [s["order"] for s in steps] == [1, 2]
    assert steps[0]["requires_human_approval"] is True  # autonomous phrasing
    assert steps[1]["requires_human_approval"] is False  # "automatic control loops" is fine
    assert violations.count("autonomous_action_language") == 1
    assert "uncited_step" in violations


def test_audit_detects_hallucinations(chunks) -> None:
    copilot = make_copilot(chunks, TemplateLLM())
    packet = packet_for(copilot, trust_row())
    clean = TemplateLLM().build(copy.deepcopy(packet))
    assert audit_response(clean, packet)["fully_grounded"]
    dirty = copy.deepcopy(clean)
    dirty["situation_summary"] += " Pressure on PLC_ZX9 reached 0.7319 bar per T0999."
    dirty["recommended_steps"][0]["citations"].append("PB-NOPE#none")
    audit = audit_response(dirty, packet)
    assert set(audit["hallucinated_entities"]) == {"PLC_ZX9", "T0999"}
    assert audit["numeric_hallucination_rate"] > 0
    assert audit["fabricated_citations"] == 1
    assert not audit["fully_grounded"]
    # rounding and percentages of packet numbers are grounded
    ok = copy.deepcopy(clean)
    ok["situation_summary"] = "Attack probability 91.2% (0.91), risk 0.72 within 30 minutes."
    assert audit_response(ok, packet)["numeric_hallucination_rate"] == 0


def test_fault_injection_is_seeded_and_detected(chunks) -> None:
    rates = {"fabricated_citation": 1.0, "invented_entity": 1.0, "numeric_perturbation": 1.0}
    llm = FaultInjectingLLM(TemplateLLM(), rates, seed=0)
    copilot = make_copilot(chunks, llm)
    result = copilot.respond(packet_for(copilot, trust_row()))
    assert result.audit_raw["fabricated_citations"] >= 1
    assert "PLC_ZX9" in result.audit_raw["hallucinated_entities"]
    assert result.audit_guarded["fabricated_citations"] == 0  # removed by guardrails
    assert {"ungrounded_entity", "ungrounded_number"} <= set(result.violations)


def test_strict_schema_is_closed_and_fully_required() -> None:
    schema = strict_json_schema(CopilotResponse)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert "$ref" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    assert "recommended_steps" in schema["properties"]


def test_cached_llm_reproduces_and_never_overwrites(chunks, tmp_path: Path) -> None:
    inner = SpyLLM(payload=lambda pkt: json.dumps(TemplateLLM().build(pkt)))
    cached = CachedLLM(inner, tmp_path, {"p": 1})
    copilot = make_copilot(chunks, cached)
    packet = packet_for(copilot, trust_row())
    first = copilot.respond(packet)
    second = copilot.respond(copy.deepcopy(packet))
    assert inner.calls == 1 and cached.hits == 1 and second.cached
    assert first.response == second.response
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1


def fake_response(stop_reason: str = "end_turn", text: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=SimpleNamespace(category="cyber"),
        content=[SimpleNamespace(type="thinking", thinking=""),
                 SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20), model="claude-opus-5-5",
        _request_id="req_test")


class FakeClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.kwargs: dict[str, Any] = {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.response = response

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return self.response


def test_anthropic_backend_request_and_response_handling() -> None:
    pytest.importorskip("anthropic")
    request = LLMRequest(system="sys", user="usr", schema={"type": "object"}, packet={})
    client = FakeClient(fake_response(text='{"ok": true}'))
    llm = AnthropicLLM(LLMConfig(backend="anthropic"), client=client)
    result = llm.generate(request)
    assert result.text == '{"ok": true}' and result.request_id == "req_test"
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert kw["output_config"]["effort"] == "medium"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kw and "thinking" not in kw
    with pytest.raises(LLMRefusalError):
        AnthropicLLM(LLMConfig(backend="anthropic"),
                     client=FakeClient(fake_response("refusal"))).generate(request)
    with pytest.raises(LLMError):
        AnthropicLLM(LLMConfig(backend="anthropic"),
                     client=FakeClient(fake_response("max_tokens"))).generate(request)
