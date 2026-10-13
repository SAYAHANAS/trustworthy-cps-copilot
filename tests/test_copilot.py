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


def _recommend_packet(chunks):
    copilot = make_copilot(chunks, TemplateLLM())
    packet = packet_for(copilot, trust_row(risk=0.4, criticality=0.5))  # recommend
    assert packet["gate"]["mode"] == "recommend"
    plain = next(c for c in chunks if not c.requires_approval and "investigation" in c.section
                 .lower())
    containment = next(c for c in chunks if c.requires_approval)
    packet["knowledge"] = [
        {"chunk_id": c.chunk_id, "rank": i + 1, "score": 1.0, "title": c.title,
         "section": c.section, "text": c.text, "requires_approval": c.requires_approval,
         "references": []} for i, c in enumerate((plain, containment))]
    return packet, plain.chunk_id, containment.chunk_id


def step(order: int, action: str, cites: list[str], approval: bool = False,
         rationale: str = "x") -> dict:
    return {"order": order, "action": action, "rationale": rationale,
            "requires_human_approval": approval, "citations": cites}


def test_guardrails_force_approval_and_drop_uncited_steps(chunks) -> None:
    packet, plain, _ = _recommend_packet(chunks)
    response = TemplateLLM().build(copy.deepcopy(packet))
    response["recommended_steps"] = [
        step(1, "Immediately disconnect the controller.", [plain]),
        step(2, "Check automatic control loops for anomalies.", [plain]),
        step(3, "Do something uncited.", [], approval=True),
    ]
    guarded, violations = enforce_policy(response, packet, max_steps=5)
    steps = guarded["recommended_steps"]
    assert len(steps) == 2 and [s["order"] for s in steps] == [1, 2]
    assert steps[0]["requires_human_approval"] is True  # autonomous phrasing
    assert steps[1]["requires_human_approval"] is False  # "automatic control loops" is fine
    assert violations.count("autonomous_action_language") == 1
    assert "uncited_step" in violations


def test_steps_citing_containment_guidance_always_need_approval(chunks) -> None:
    packet, plain, containment = _recommend_packet(chunks)
    response = TemplateLLM().build(copy.deepcopy(packet))
    response["recommended_steps"] = [
        step(1, "Rate-limit the flooding source at the segment boundary.", [containment]),
        step(2, "Review the firewall logs.", [plain]),
    ]
    guarded, violations = enforce_policy(response, packet, max_steps=5)
    assert guarded["recommended_steps"][0]["requires_human_approval"] is True
    assert guarded["recommended_steps"][1]["requires_human_approval"] is False
    assert violations == ["containment_without_approval"]


@pytest.mark.parametrize("action, rationale", [
    ("Isolate the controller automatically.", "x"),
    ("Shut down the transfer pump immediately.", "x"),
    ("Disconnect the PLC without waiting for approval.", "x"),
    ("Block the flooding source at the segment boundary now.", "x"),
    ("Rate-limit the source host.", "Do this automatically without analyst approval."),
    ("Automatically block the PLC.", "x"),
    ("Restart the HMI; no need for approval.", "x"),
])
def test_autonomous_phrasings_are_forced_to_approval(chunks, action, rationale) -> None:
    packet, plain, _ = _recommend_packet(chunks)
    response = TemplateLLM().build(copy.deepcopy(packet))
    response["recommended_steps"] = [step(1, action, [plain], rationale=rationale)]
    guarded, violations = enforce_policy(response, packet, max_steps=5)
    assert guarded["recommended_steps"][0]["requires_human_approval"] is True
    assert "autonomous_action_language" in violations


def test_knowledge_base_steps_never_look_autonomous(chunks) -> None:
    from cpscopilot.copilot.copilot import _AUTONOMOUS
    from cpscopilot.copilot.knowledge_base import numbered_steps

    flagged = [s for c in chunks for s in numbered_steps(c.text) if _AUTONOMOUS.search(s)]
    assert flagged == []


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


def test_template_knowledge_claims_have_content(chunks) -> None:
    from cpscopilot.copilot.audit import LexicalSupportScorer

    copilot = make_copilot(chunks, TemplateLLM())
    packet = packet_for(copilot, trust_row())
    kb_claims = [e["statement"] for e in TemplateLLM().build(packet)["evidence"]
                 if e["source"] == "knowledge_base"]
    assert kb_claims and all(len(c.split()) > 3 for c in kb_claims)  # never just "1."
    assert LexicalSupportScorer()("1.", "unrelated evidence") == 0.0
    assert LexicalSupportScorer()("the of and", "anything") == 0.0


@pytest.mark.parametrize("text, grounded", [
    ("Reading -2.5 on DPIT301.", True),    # signed packet value
    ("Reading 2.5 below normal.", True),   # magnitude of a signed value
    ("Traffic reached 4,321 packets.", True),
    ("Traffic rose to 9000pps.", False),   # unit-suffixed invented number is seen
    ("Delay of 250ms.", False),
])
def test_number_grounding_handles_signs_separators_and_units(chunks, text, grounded) -> None:
    copilot = make_copilot(chunks, TemplateLLM())
    packet = packet_for(copilot, trust_row())
    packet["top_deviations"] = [{"feature": "NET_PKTS", "z": 9.5, "value": 4321.0,
                                 "asset": "control_network"},
                                {"feature": "DPIT301", "z": 4.0, "value": -2.5,
                                 "asset": "unattributed"}]
    response = TemplateLLM().build(copy.deepcopy(packet))
    response["situation_summary"] = text
    rate = audit_response(response, packet)["numeric_hallucination_rate"]
    assert (rate == 0) == grounded


def test_raw_and_guarded_audits_use_the_same_rows(chunks) -> None:
    from cpscopilot.copilot.audit import aggregate_audits

    rates = {"invented_entity": 1.0}
    llms = [FaultInjectingLLM(TemplateLLM(), rates, seed=0), SpyLLM(error=LLMError("boom"))]
    records = []
    for llm, row in ((llms[0], trust_row()), (llms[1], trust_row()),
                     (llms[0], trust_row(epistemic=0.5))):  # the last one abstains
        copilot = make_copilot(chunks, llm)
        records.append(copilot.respond(packet_for(copilot, row)).to_record())
    agg = aggregate_audits(records)
    assert agg["n_generated"] == 1 and agg["raw"]["n"] == agg["guarded"]["n"] == 1
    assert agg["guarded"]["entity_hallucination_rate"] == agg["raw"]["entity_hallucination_rate"]
    assert agg["guarded_all_events"]["n"] == 3
    assert agg["fallbacks"] == {"abstain_mode": 1, "llm_error": 1, "schema_invalid": 0}
    assert agg["schema_valid_rate"] == 0.5 and agg["error_rate_llm_calls"] == 0.5


def test_llm_errors_are_invalid_output_and_stay_out_of_analyst_text(chunks) -> None:
    llm = SpyLLM(error=LLMError("API error 529: Overloaded (request req_011CVabc123)"))
    copilot = make_copilot(chunks, llm)
    result = copilot.respond(packet_for(copilot, trust_row()))
    assert result.schema_valid is False and "529" in result.error
    shown = json.dumps(result.response)
    assert "529" not in shown and "req_011" not in shown
    assert result.audit_guarded["numeric_hallucination_rate"] in (0.0, None) or \
        result.audit_guarded["numeric_hallucination_rate"] == 0
    assert result.audit_guarded["hallucinated_entities"] == []


def test_cache_statistics_are_found_behind_wrappers(tmp_path: Path) -> None:
    pytest.importorskip("anthropic")
    from cpscopilot.copilot.llm import build_llm
    from cpscopilot.pipeline.runner import find_cache

    client = FakeClient(fake_response(text="{}"))
    plain = build_llm(LLMConfig(backend="anthropic"), tmp_path, 0, 5, client=client)
    faulty = build_llm(LLMConfig(backend="anthropic", fault_injection={"invented_entity": 0.5}),
                       tmp_path, 0, 5, client=client)
    assert isinstance(find_cache(plain), CachedLLM)
    assert isinstance(find_cache(faulty), CachedLLM)
    assert find_cache(TemplateLLM()) is None
