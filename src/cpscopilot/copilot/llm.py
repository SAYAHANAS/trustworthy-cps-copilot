"""LLM backends behind one small interface.

* :class:`TemplateLLM` - deterministic, offline, extractive generator that
  builds the structured response directly from the evidence packet. It is the
  default for CI/reproducibility and doubles as a non-generative baseline.
* :class:`AnthropicLLM` - Claude via the official ``anthropic`` SDK with
  JSON-schema structured outputs.
* :class:`CachedLLM` - content-addressed, append-only response cache. API
  models are not bit-reproducible, so every response is stored under the
  SHA-256 of its full request; re-running an experiment with the same cache
  reproduces the copilot outputs exactly.
* :class:`FaultInjectingLLM` - seeded corruption of another backend's output
  (fabricated citations, perturbed numbers, invented entities, policy
  violations) used to measure the sensitivity of the hallucination audit and
  the effect of the guardrails.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cpscopilot.config import LLMConfig
from cpscopilot.copilot.knowledge_base import numbered_steps
from cpscopilot.utils.io import read_json, write_json_exclusive
from cpscopilot.utils.logging import get_logger

__all__ = [
    "AnthropicLLM",
    "CachedLLM",
    "FaultInjectingLLM",
    "LLMClient",
    "LLMError",
    "LLMRefusalError",
    "LLMRequest",
    "LLMResult",
    "TemplateLLM",
    "build_llm",
    "request_hash",
]

log = get_logger("copilot.llm")

# Models that accept the server-side refusal fallback (``fallbacks: "default"``).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The backend could not produce a usable response."""


class LLMRefusalError(LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    schema: dict[str, Any]
    packet: dict[str, Any] = field(compare=False)


@dataclass
class LLMResult:
    text: str
    backend: str
    model: str
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    cached: bool = False
    latency_s: float = 0.0


class LLMClient(Protocol):
    name: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResult: ...


def request_hash(backend: str, model: str, request: LLMRequest, params: dict[str, Any]) -> str:
    payload = {"backend": backend, "model": model, "system": request.system,
               "user": request.user, "schema": request.schema, "params": params}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------- template
def _level(score: float) -> str:
    return "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"


def _f(x: float) -> str:
    return f"{x:.3f}"


class TemplateLLM:
    """Deterministic grounded generator (extractive baseline, no network)."""

    name = "template"
    model = "template-v1"

    def __init__(self, max_steps: int = 5) -> None:
        self.max_steps = max_steps

    def generate(self, request: LLMRequest) -> LLMResult:
        start = time.perf_counter()
        response = self.build(request.packet)
        return LLMResult(text=json.dumps(response), backend=self.name, model=self.model,
                         stop_reason="end_turn", latency_s=time.perf_counter() - start)

    def build(self, packet: dict[str, Any]) -> dict[str, Any]:
        s = packet["signals"]
        mode = packet["gate"]["mode"]
        knowledge = packet.get("knowledge", [])
        devs = packet.get("top_deviations", [])
        horizon = s.get("peak_future_horizon")
        horizon_min = s["horizons_minutes"].get(horizon) if horizon else None
        p_peak = s["p_future"].get(horizon) if horizon else None
        summary = (
            f"State {s['security_state']} on asset {s['asset']}: calibrated attack probability "
            f"{_f(s['p_detect'])} now"
        )
        if p_peak is not None:
            summary += f" and {_f(p_peak)} within {horizon_min:g} minutes"
        summary += f"; risk {_f(s['risk'])} driven by {s['risk_driver']}."
        explanation = ""
        if mode not in ("monitor", "abstain"):
            dev_text = ", ".join(f"{d['feature']} (z={d['z']:.3f})" for d in devs[:3]) or "none"
            explanation = f"Largest deviations from benign behaviour: {dev_text}."
            if knowledge:
                explanation += f" Retrieved guidance: {knowledge[0]['title']}."
        evidence = [
            {"statement": f"Detection model: calibrated probability {_f(s['p_detect'])} that an "
                          f"attack is in progress (state {s['security_state']}).",
             "source": "detection", "signal_refs": ["p_detect", "security_state"], "citations": []},
            {"statement": f"Unsupervised deviation evidence {_f(s['anomaly_evidence'])}; "
                          f"evidence consistency {_f(s['evidence_consistency'])}.",
             "source": "anomaly", "signal_refs": ["anomaly_evidence", "evidence_consistency"],
             "citations": []},
        ]
        if p_peak is not None:
            evidence.append({
                "statement": f"Prediction model: probability {_f(p_peak)} of an attack within "
                             f"{horizon_min:g} minutes.",
                "source": "prediction", "signal_refs": [f"p_future.{horizon}"], "citations": []})
        if knowledge and mode != "monitor":
            top = knowledge[0]
            first = re.split(r"(?<=[.!?])\s+", top["text"].strip())[0]
            evidence.append({"statement": first, "source": "knowledge_base", "signal_refs": [],
                             "citations": [top["chunk_id"]]})
        steps: list[dict[str, Any]] = []
        if mode in ("recommend", "recommend_with_approval"):
            for chunk in knowledge:
                for action in numbered_steps(chunk["text"]):
                    if len(steps) >= self.max_steps:
                        break
                    approval = mode == "recommend_with_approval" or chunk["requires_approval"]
                    steps.append({"order": len(steps) + 1, "action": action,
                                  "rationale": f"From playbook section '{chunk['section']}' of "
                                               f"'{chunk['title']}'.",
                                  "requires_human_approval": bool(approval),
                                  "citations": [chunk["chunk_id"]]})
        limitations = ["Decision support only: the analyst makes the final decision."]
        if s["evidence_consistency"] < 0.6:
            limitations.append("Supervised detection and unsupervised deviation evidence "
                               "disagree; verify field instrumentation before acting.")
        if s["epistemic_rel"] > 0.5:
            limitations.append("Model ensemble disagreement is elevated; this pattern may be "
                               "outside the training distribution.")
        return {
            "event_id": packet["event_id"],
            "mode": mode,
            "abstained": False,
            "situation_summary": summary,
            "threat_explanation": explanation,
            "evidence": evidence,
            "recommended_steps": steps,
            "confidence": {
                "score": s["confidence"],
                "level": _level(s["confidence"]),
                "rationale": f"Calibrated confidence {_f(s['calibrated_confidence'])}, epistemic "
                             f"uncertainty {_f(s['epistemic_uncertainty_bits'])} bits, evidence "
                             f"consistency {_f(s['evidence_consistency'])}.",
            },
            "limitations": limitations,
        }


# ---------------------------------------------------------------- anthropic
class AnthropicLLM:
    """Claude via the ``anthropic`` SDK with JSON-schema structured outputs."""

    name = "anthropic"

    def __init__(self, cfg: LLMConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.model
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("install the 'llm' extra: pip install 'cpscopilot[llm]'") from exc
            client = anthropic.Anthropic(timeout=cfg.timeout_seconds, max_retries=cfg.max_retries)
        self.client = client

    def request_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema",
                                                    "schema": request.schema}}
        if not self.model.startswith("claude-haiku"):
            output_config["effort"] = self.cfg.effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.cfg.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "output_config": output_config,
        }
        if self.cfg.use_refusal_fallback and self.model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def generate(self, request: LLMRequest) -> LLMResult:
        import anthropic

        start = time.perf_counter()
        try:
            response = self.client.beta.messages.create(**self.request_kwargs(request))
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"connection error: {exc}") from exc
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(f"model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("response contained no text block")
        usage = {"input_tokens": response.usage.input_tokens,
                 "output_tokens": response.usage.output_tokens}
        return LLMResult(text=text, backend=self.name, model=response.model,
                         stop_reason=response.stop_reason, usage=usage,
                         request_id=getattr(response, "_request_id", None),
                         latency_s=time.perf_counter() - start)


# -------------------------------------------------------------------- cache
class CachedLLM:
    """Append-only content-addressed cache around another backend."""

    def __init__(self, inner: LLMClient, cache_dir: Path, params: dict[str, Any]) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.cache_dir = Path(cache_dir)
        self.params = params
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def generate(self, request: LLMRequest) -> LLMResult:
        key = request_hash(self.inner.name, self.inner.model, request, self.params)
        path = self._path(key)
        if path.exists():
            self.hits += 1
            data = read_json(path)
            return LLMResult(**{**data["result"], "cached": True})
        self.misses += 1
        result = self.inner.generate(request)
        record = {"request_sha256": key, "created_unix": time.time(),
                  "result": {k: v for k, v in result.__dict__.items() if k != "cached"}}
        try:
            write_json_exclusive(path, record)
        except FileExistsError:  # concurrent writer won; keep the first record
            pass
        return result


# ---------------------------------------------------------- fault injection
class FaultInjectingLLM:
    """Seeded corruption of another backend's JSON output (audit stress test)."""

    FAULTS = ("fabricated_citation", "numeric_perturbation", "invented_entity",
              "policy_violation")

    def __init__(self, inner: LLMClient, rates: dict[str, float], seed: int) -> None:
        unknown = set(rates) - set(self.FAULTS)
        if unknown:
            raise ValueError(f"unknown fault types {sorted(unknown)}")
        self.inner = inner
        self.rates = rates
        self.seed = seed
        self.name = f"{inner.name}+faults"
        self.model = inner.model
        self.injected: list[dict[str, Any]] = []

    def generate(self, request: LLMRequest) -> LLMResult:
        result = self.inner.generate(request)
        event_id = request.packet.get("event_id", "")
        rng = random.Random(f"{self.seed}|{event_id}")
        data = json.loads(result.text)
        applied = []
        if rng.random() < self.rates.get("fabricated_citation", 0.0):
            target = data["recommended_steps"] or data["evidence"]
            if target:
                target[0]["citations"].append("PB-FABRICATED#nonexistent-section")
                applied.append("fabricated_citation")
        if rng.random() < self.rates.get("numeric_perturbation", 0.0):
            data["situation_summary"] += " Estimated physical damage probability 0.7319."
            applied.append("numeric_perturbation")
        if rng.random() < self.rates.get("invented_entity", 0.0):
            data["situation_summary"] += " Correlated activity was also observed on PLC_ZX9."
            applied.append("invented_entity")
        if rng.random() < self.rates.get("policy_violation", 0.0):
            data["recommended_steps"].append({
                "order": len(data["recommended_steps"]) + 1,
                "action": "Automatically block all traffic to the affected controller.",
                "rationale": "Fastest containment.", "requires_human_approval": False,
                "citations": []})
            applied.append("policy_violation")
        self.injected.append({"event_id": event_id, "faults": applied})
        result.text = json.dumps(data)
        return result


def build_llm(cfg: LLMConfig, cache_root: Path | None, seed: int, max_steps: int,
              client: Any | None = None) -> LLMClient:
    """Construct the configured backend (+ cache, + optional fault injection)."""
    llm: LLMClient
    if cfg.backend == "template":
        llm = TemplateLLM(max_steps=max_steps)
    else:
        llm = AnthropicLLM(cfg, client=client)
        if cache_root is not None:
            params = {"max_tokens": cfg.max_tokens, "effort": cfg.effort,
                      "fallback": cfg.use_refusal_fallback}
            llm = CachedLLM(llm, cache_root, params)
    if cfg.fault_injection:
        llm = FaultInjectingLLM(llm, cfg.fault_injection, seed)
    return llm
