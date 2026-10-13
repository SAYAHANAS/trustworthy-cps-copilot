"""Copilot orchestration: evidence packet -> retrieval -> LLM -> guardrails -> audit.

The copilot receives *only* model outputs, trust-layer signals, the gate
decision, analyst context and current sensor deviations - never ground-truth
labels or attack types. Its output is validated against
:class:`~cpscopilot.copilot.schemas.CopilotResponse` and then sanitised by
:func:`enforce_policy`, which makes the assistance gate binding regardless of
what the LLM produced:

* no recommendations outside the recommend modes, no explanation in monitor,
* every step must cite a retrieved chunk (uncited steps are dropped),
* fabricated citations and unknown signal references are removed,
* approval flags are forced where the mode/analyst requires them, and any
  step phrased as an autonomous action is forced to require approval,
* the numeric confidence is always the trust layer's calibrated value,
* numbers or identifiers in free text that the evidence does not contain are
  flagged (``ungrounded_number`` / ``ungrounded_entity``) for the analyst - free
  text is never silently rewritten.

Both the raw and the guarded outputs are audited, so experiments report the
LLM's intrinsic hallucination rate *and* the residual rate after guardrails.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import ValidationError

from cpscopilot.config import AnalystProfile, CopilotConfig
from cpscopilot.copilot.audit import audit_response, ungrounded_spans
from cpscopilot.copilot.llm import LLMClient, LLMError, LLMRequest
from cpscopilot.copilot.prompts import SYSTEM_PROMPT, build_user_prompt
from cpscopilot.copilot.retriever import BM25Retriever
from cpscopilot.copilot.schemas import CopilotResponse, strict_json_schema
from cpscopilot.gate.policy import AssistanceMode, GateDecision
from cpscopilot.trust.asset import AssetRegistry

__all__ = ["DEFAULT_GLOSSARY", "CognitiveCopilot", "CopilotResult", "autonomous_language",
           "enforce_policy"]

DEFAULT_GLOSSARY: dict[str, str] = {
    r"^LIT": "level transmitter tank level sensor reading",
    r"^FIT": "flow transmitter flow rate sensor reading",
    r"^(MV|P)\d": "actuator valve pump command state",
    r"^AIT": "analyzer chemical dosing water quality ph conductivity setpoint",
    r"^(DPIT|PIT)": "pressure transmitter membrane filtration",
    r"^NET_PKTS": "network traffic volume packets flooding denial of service",
    r"^NET_UNIQ_DST": "network scanning reconnaissance discovery hosts",
    r"^NET_WRITES": "write commands unauthorized command message",
    r"^NET_AUTH": "authentication failures brute force credentials",
    r"(?i)bytes|pkts|event_count": "network traffic volume flows",
    r"(?i)dns": "dns queries",
    r"(?i)http": "http web requests injection",
}

# Autonomous-action language: a step that tells someone to act without a human in
# the loop. Each clause (split at . ; ! ? ,) is checked on its own so that patterns
# never span sentences, and negated clauses ("do not restart the pump without
# approval") are ignored, except for explicit approval skipping ("don't wait for
# approval"). Base-form verbs only, so descriptions ("blocked connections", "the
# pump stopped", "automatic shutdown interlock") are not matched. Imperatives with
# an urgency adverb ("Isolate the PLC immediately") are checked in the action only;
# a rationale explains and often says "now" descriptively. A match only forces
# approval, so the check errs on the conservative side.
_VERB = (r"(?:block|isolate|shut\s*(?:down|off)|disconnect|quarantine|stop|kill|disable|"
         r"terminate|cut\s+off|null-?route|reset|reboot|restart|power\s*(?:off|down))")
_APPROVAL = r"(?:approval|review|confirmation|authori[sz]ation|consent|sign-?off)"
_SKIPS_APPROVAL = re.compile(
    rf"\bno\s+need\s+(?:for|to\s+(?:wait\s+for|get|obtain|request|seek))\s+(?:\w+\s+)?"
    rf"{_APPROVAL}\b"
    rf"|\b(?:do\s+not|don't|never)\s+wait\s+for\s+(?:\w+\s+)?{_APPROVAL}\b"
    rf"|\bskip\s+(?:the\s+)?(?:\w+\s+)?{_APPROVAL}\b",
    re.IGNORECASE,
)
_AUTONOMOUS_ANYWHERE = re.compile(
    rf"\bautomatically\s+(?:\w+\s+){{0,2}}?{_VERB}\b"
    rf"|\bauto-?{_VERB}\b"
    rf"|\bimmediately\s+(?:\w+\s+)?{_VERB}\b"
    rf"|\bwithout\b.{{0,30}}?\b{_APPROVAL}\b",
    re.IGNORECASE,
)
_IMPERATIVE_NOW = re.compile(
    rf"\W*(?:(?:then|and|so|please)\s+)*{_VERB}\b.{{0,60}}?\b(?:automatically|immediately|"
    r"right\s+away|at\s+once|now|without\s+delay)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:do\s+not|don't|never|must\s+not|mustn't|should\s+not|"
                       r"shouldn't|cannot|can't|avoid)\b", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[.;!?,\n]+")


def autonomous_language(action: str, rationale: str = "") -> bool:
    """Whether a recommended step asks for action without a human in the loop."""
    for text, imperative in ((action, True), (rationale, False)):
        for clause in (c.strip() for c in _CLAUSE_BREAK.split(text or "")):
            if not clause:
                continue
            if _SKIPS_APPROVAL.search(clause):
                return True
            if _NEGATION.search(clause):
                continue
            if _AUTONOMOUS_ANYWHERE.search(clause) or (
                    imperative and _IMPERATIVE_NOW.match(clause)):
                return True
    return False


def _r(x: Any) -> float:
    return round(float(x), 3)


def _level(score: float) -> str:
    return "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"


@dataclass
class CopilotResult:
    """Everything recorded for one copilot invocation."""

    event_id: str
    mode: str
    llm_called: bool
    response: dict[str, Any]
    packet: dict[str, Any]
    raw_text: str | None = None
    raw_response: dict[str, Any] | None = None
    schema_valid: bool = True
    violations: list[str] = field(default_factory=list)
    backend: str = ""
    model: str = ""
    cached: bool = False
    latency_s: float = 0.0
    error: str | None = None
    audit_raw: dict[str, Any] | None = None
    audit_guarded: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "mode": self.mode, "llm_called": self.llm_called,
            "backend": self.backend, "model": self.model, "cached": self.cached,
            "latency_s": self.latency_s, "error": self.error, "schema_valid": self.schema_valid,
            "violations": self.violations, "retrieved": [
                {"chunk_id": k["chunk_id"], "rank": k["rank"], "score": k["score"]}
                for k in self.packet.get("knowledge", [])],
            "query": self.packet.get("retrieval_query"),
            "response": self.response, "raw_response": self.raw_response,
            "raw": self.audit_raw, "guarded": self.audit_guarded,
        }


def enforce_policy(response: dict[str, Any], packet: dict[str, Any],
                   max_steps: int) -> tuple[dict[str, Any], list[str]]:
    """Make the gate decision binding; returns ``(sanitised response, violations)``."""
    gate = packet["gate"]
    mode = AssistanceMode(gate["mode"])
    approval_all = mode == AssistanceMode.RECOMMEND_WITH_APPROVAL or \
        gate.get("recommendations_require_approval", False)
    retrieved = {k["chunk_id"] for k in packet.get("knowledge", [])}
    # Containment / response / recovery guidance always needs analyst approval.
    approval_chunks = {k["chunk_id"] for k in packet.get("knowledge", [])
                       if k.get("requires_approval")}
    allowed_refs = set(packet.get("allowed_signal_refs", []))
    out = copy.deepcopy(response)
    violations: list[str] = []

    if out.get("event_id") != packet["event_id"]:
        violations.append("event_id_mismatch")
        out["event_id"] = packet["event_id"]
    if out.get("mode") != mode.value:
        violations.append("mode_mismatch")
        out["mode"] = mode.value
    if out.get("abstained"):
        violations.append("self_abstention")

    evidence = []
    for item in out.get("evidence", []):
        kept = [c for c in item["citations"] if c in retrieved]
        if len(kept) != len(item["citations"]):
            violations.append("fabricated_citation")
        if set(kept) & approval_chunks and not mode.allows_recommendations:
            # quoting containment guidance would smuggle an action into a mode without them
            violations.append("action_in_evidence")
            continue
        item["citations"] = kept
        refs = [r for r in item["signal_refs"] if r in allowed_refs]
        if len(refs) != len(item["signal_refs"]):
            violations.append("invalid_signal_ref")
        item["signal_refs"] = refs
        evidence.append(item)
    if "evidence" in out:
        out["evidence"] = evidence

    if not mode.allows_explanation and out.get("threat_explanation"):
        violations.append("explanation_not_permitted")
        out["threat_explanation"] = ""

    steps = out.get("recommended_steps", [])
    if steps and not mode.allows_recommendations:
        violations.append("steps_not_permitted")
        steps = []
    guarded_steps = []
    for step in steps:
        kept = [c for c in step["citations"] if c in retrieved]
        if len(kept) != len(step["citations"]):
            violations.append("fabricated_citation")
        if not kept:
            violations.append("uncited_step")
            continue
        step["citations"] = kept
        if autonomous_language(step["action"], step.get("rationale", "")):
            violations.append("autonomous_action_language")
            step["requires_human_approval"] = True
        if set(kept) & approval_chunks and not step["requires_human_approval"]:
            violations.append("containment_without_approval")
            step["requires_human_approval"] = True
        if approval_all and not step["requires_human_approval"]:
            violations.append("missing_approval_flag")
            step["requires_human_approval"] = True
        guarded_steps.append(step)
    if len(guarded_steps) > max_steps:
        violations.append("too_many_steps")
        guarded_steps = guarded_steps[:max_steps]
    for i, step in enumerate(guarded_steps, start=1):
        step["order"] = i
    out["recommended_steps"] = guarded_steps

    trust_conf = float(packet["signals"]["confidence"])
    conf = out.get("confidence", {})
    if abs(float(conf.get("score", -1.0)) - trust_conf) > 0.01:
        violations.append("confidence_mismatch")
    conf["score"] = trust_conf
    conf["level"] = _level(trust_conf)
    out["confidence"] = conf

    # Free text is flagged, not rewritten: the analyst sees the warning next to the output.
    spans = ungrounded_spans(out, packet)
    if spans["numbers"]:
        violations.append("ungrounded_number")
    if spans["entities"]:
        violations.append("ungrounded_entity")
    return out, violations


class CognitiveCopilot:
    """Retrieval-augmented, gate-constrained decision support."""

    def __init__(self, cfg: CopilotConfig, retriever: BM25Retriever, llm: LLMClient,
                 registry: AssetRegistry, horizon_minutes: dict[str, float]) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm
        self.registry = registry
        self.horizon_minutes = dict(horizon_minutes)
        self.glossary = [(re.compile(p), kw) for p, kw in
                         (cfg.feature_glossary or DEFAULT_GLOSSARY).items()]
        self.schema = strict_json_schema(CopilotResponse)

    # ------------------------------------------------------------- evidence
    def _keywords(self, feature: str) -> str:
        return " ".join(kw for pattern, kw in self.glossary if pattern.search(feature))

    def build_query(self, signals: dict[str, Any], deviations: list[dict[str, Any]],
                    mode: str) -> str:
        terms = [f"{signals['security_state']} attack"]
        for d in deviations:
            terms.append(self._keywords(d["feature"]))
        terms.append(self.registry.describe(signals["asset"]))
        if signals["risk_driver"] != "detection":
            terms.append("reconnaissance precursor early warning predicted attack")
        if mode == AssistanceMode.ABSTAIN.value or signals["evidence_consistency"] < 0.5:
            terms.append("uncertain conflicting evidence verify sensor health")
        if mode in (AssistanceMode.RECOMMEND.value, AssistanceMode.RECOMMEND_WITH_APPROVAL.value):
            terms.append("investigation steps containment")
        return " ".join(t for t in terms if t).strip()

    def build_packet(self, event_id: str, timestamp: Any, row: pd.Series,
                     decision: GateDecision, ctx: AnalystProfile,
                     deviations: list[dict[str, Any]]) -> dict[str, Any]:
        """Assemble the structured evidence packet for one event."""
        keys = list(self.horizon_minutes)
        p_future = {k: _r(row[f"p_{k}"]) for k in keys}
        driver = str(row["risk_driver"])
        peak = driver if driver in p_future else (max(p_future, key=p_future.get) if p_future
                                                  else None)
        signals = {
            "security_state": str(row["state"]),
            "p_detect": _r(row["p_detect"]),
            "p_future": p_future,
            "horizons_minutes": {k: float(m) for k, m in self.horizon_minutes.items()},
            "peak_future_horizon": peak,
            "risk": _r(row["risk"]),
            "risk_driver": driver,
            "confidence": _r(row["confidence"]),
            "calibrated_confidence": _r(row["calibrated_confidence"]),
            "epistemic_uncertainty_bits": _r(row["epistemic"]),
            "epistemic_rel": _r(row["epistemic_rel"]),
            "aleatoric_uncertainty_bits": _r(row["aleatoric"]),
            "evidence_consistency": _r(row["consistency"]),
            "anomaly_evidence": _r(row["anomaly_evidence"]),
            "asset": str(row["asset"]),
            "asset_criticality": _r(row["criticality"]),
            "asset_description": self.registry.describe(str(row["asset"])),
        }
        allowed = [k for k in signals if k not in ("p_future", "horizons_minutes")]
        allowed += [f"p_future.{k}" for k in keys]
        allowed += [f"deviation:{d['feature']}" for d in deviations]
        mode = decision.mode.value
        query = self.build_query(signals, deviations, mode)
        hits = self.retriever.search(query, self.cfg.top_k)
        knowledge = [{"chunk_id": h.chunk.chunk_id, "rank": h.rank, "score": round(h.score, 4),
                      "title": h.chunk.title, "section": h.chunk.section, "text": h.chunk.text,
                      "requires_approval": h.chunk.requires_approval,
                      "references": list(h.chunk.references)} for h in hits]
        return {
            "event_id": event_id,
            "timestamp": str(timestamp),
            "gate": {"mode": mode, "rule_id": decision.rule_id, "reasons": list(decision.reasons),
                     "recommendations_require_approval":
                         decision.thresholds.recommendations_require_approval},
            "analyst_context": ctx.model_dump(),
            "signals": signals,
            "top_deviations": deviations,
            "allowed_signal_refs": allowed,
            "retrieval_query": query,
            "knowledge": knowledge,
        }

    # ------------------------------------------------------------- response
    @staticmethod
    def abstention(packet: dict[str, Any], reason: str) -> dict[str, Any]:
        s = packet["signals"]
        return {
            "event_id": packet["event_id"],
            "mode": packet["gate"]["mode"],
            "abstained": True,
            "situation_summary": f"Copilot abstains: {reason}. The raw evidence is routed to "
                                 "the analyst for manual review.",
            "threat_explanation": "",
            "evidence": [
                {"statement": f"Detection model: calibrated probability {s['p_detect']:.3f} "
                              f"(state {s['security_state']}).", "source": "detection",
                 "signal_refs": ["p_detect", "security_state"], "citations": []},
                {"statement": f"Epistemic uncertainty {s['epistemic_uncertainty_bits']:.3f} bits; "
                              f"evidence consistency {s['evidence_consistency']:.3f}.",
                 "source": "gate", "signal_refs": ["epistemic_uncertainty_bits",
                                                   "evidence_consistency"], "citations": []},
            ],
            "recommended_steps": [],
            "confidence": {"score": s["confidence"], "level": _level(s["confidence"]),
                           "rationale": "Trust conditions for automated assistance are not met."},
            "limitations": [reason, "No analysis or recommendation is provided under low trust."],
        }

    def respond(self, packet: dict[str, Any]) -> CopilotResult:
        """Produce, validate, guard and audit the copilot output for one event."""
        mode = packet["gate"]["mode"]
        result = CopilotResult(event_id=packet["event_id"], mode=mode, llm_called=False,
                               response={}, packet=packet, backend=self.llm.name,
                               model=self.llm.model)
        if mode == AssistanceMode.ABSTAIN.value:
            reason = packet["gate"]["reasons"][0] if packet["gate"]["reasons"] else "low trust"
            result.response = self.abstention(packet, reason)
            result.audit_guarded = audit_response(result.response, packet,
                                                  self.cfg.support_threshold)
            return result

        request = LLMRequest(system=SYSTEM_PROMPT,
                             user=build_user_prompt(packet, self.cfg.max_steps),
                             schema=self.schema, packet=packet)
        result.llm_called = True
        try:
            llm_out = self.llm.generate(request)
        except LLMError as exc:
            # No structured output was produced; the raw error stays out of the analyst text
            # (it would also be audited as ungrounded numbers/identifiers).
            result.error = str(exc)
            result.schema_valid = False
            result.violations = ["llm_error"]
            result.response = self.abstention(packet, "the language model was unavailable")
            result.audit_guarded = audit_response(result.response, packet,
                                                  self.cfg.support_threshold)
            return result
        result.raw_text, result.cached = llm_out.text, llm_out.cached
        result.latency_s, result.model = llm_out.latency_s, llm_out.model
        try:
            raw = json.loads(llm_out.text)
            result.raw_response = raw
            parsed = CopilotResponse.model_validate(raw).model_dump()
        except (json.JSONDecodeError, ValidationError) as exc:
            result.schema_valid = False
            result.violations = ["schema_invalid"]
            result.error = f"invalid structured output: {type(exc).__name__}"
            result.response = self.abstention(packet, "the copilot output failed validation")
            result.audit_guarded = audit_response(result.response, packet,
                                                  self.cfg.support_threshold)
            return result
        result.audit_raw = audit_response(parsed, packet, self.cfg.support_threshold)
        guarded, violations = enforce_policy(parsed, packet, self.cfg.max_steps)
        CopilotResponse.model_validate(guarded)
        result.response, result.violations = guarded, violations
        result.audit_guarded = audit_response(guarded, packet, self.cfg.support_threshold)
        return result
