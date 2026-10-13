"""Structured copilot output schema (pydantic) and its strict JSON Schema.

The same pydantic model validates every backend's output, and
:func:`strict_json_schema` derives the JSON Schema sent to LLM APIs with
structured outputs (all objects closed with ``additionalProperties: false``,
all properties required, ``$ref`` inlined, numeric/length constraints removed
because they are enforced by the validator instead).
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ConfidenceStatement",
    "CopilotResponse",
    "EvidenceItem",
    "RecommendedStep",
    "strict_json_schema",
]

ModeLiteral = Literal["monitor", "explain", "recommend", "recommend_with_approval", "abstain"]
SourceLiteral = Literal["detection", "prediction", "anomaly", "asset", "knowledge_base", "gate"]


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(_Closed):
    """One factual claim and what supports it."""

    statement: str = Field(description="One factual sentence grounded in the packet.")
    source: SourceLiteral
    signal_refs: list[str] = Field(description="Keys from allowed_signal_refs.")
    citations: list[str] = Field(description="Knowledge chunk ids from the packet.")


class RecommendedStep(_Closed):
    """One investigation/response step for the human analyst."""

    order: int = Field(ge=1)
    action: str
    rationale: str
    requires_human_approval: bool
    citations: list[str]


class ConfidenceStatement(_Closed):
    score: float = Field(ge=0.0, le=1.0, description="Copy signals.confidence exactly.")
    level: Literal["low", "medium", "high"]
    rationale: str


class CopilotResponse(_Closed):
    """Complete structured copilot output for one event."""

    event_id: str
    mode: ModeLiteral
    abstained: bool
    situation_summary: str
    threat_explanation: str
    evidence: list[EvidenceItem]
    recommended_steps: list[RecommendedStep]
    confidence: ConfidenceStatement
    limitations: list[str]


_DROP_KEYS = {"title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
              "minLength", "maxLength", "minItems", "maxItems", "pattern", "format"}


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Closed, fully-required, ``$ref``-free JSON Schema for structured outputs."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            out = {}
            for key, value in node.items():
                if key == "properties":  # keys here are field names, never drop them
                    out[key] = {name: resolve(sub) for name, sub in value.items()}
                elif key not in _DROP_KEYS:
                    out[key] = resolve(value)
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"])
            return out
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)
