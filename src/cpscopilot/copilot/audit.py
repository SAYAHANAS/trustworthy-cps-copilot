"""Groundedness and hallucination audit of copilot outputs.

All checks are deterministic and reference-free (no LLM judge):

``citation_precision``      cited chunk ids that were actually retrieved
``citation_coverage``       steps / knowledge claims carrying >= 1 valid citation
``signal_ref_validity``     signal references that exist in the packet
``supported_claim_rate``    cited claims whose content words are covered (>=
                            ``support_threshold``) by the cited chunks - a
                            lexical-entailment proxy; plug an NLI model in via
                            :class:`SupportScorer` for stronger checks
``numeric_hallucination``   numbers in free text not derivable from the packet
                            (rounding and percentages allowed)
``entity_hallucination``    identifier-like tokens (sensor tags, technique ids,
                            asset ids) absent from the packet
``groundedness``            mean of the available component scores
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

from cpscopilot.copilot.retriever import tokenize

__all__ = ["LexicalSupportScorer", "SupportScorer", "aggregate_audits", "audit_response",
           "extract_entities", "extract_numbers", "ungrounded_spans"]

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)(%?)(?![A-Za-z0-9_])(?!\.\d)")
_ENTITY = re.compile(r"\b(?:[A-Za-z][A-Za-z0-9_]*\d[A-Za-z0-9_]*|[A-Z][A-Z0-9]*_[A-Za-z0-9_]+)\b")
_TRIVIAL_INTEGERS = set(range(0, 11))


class SupportScorer(Protocol):
    def __call__(self, claim: str, evidence: str) -> float: ...


class LexicalSupportScorer:
    """Fraction of the claim's content words that occur in the evidence."""

    def __call__(self, claim: str, evidence: str) -> float:
        claim_tokens = set(tokenize(claim))
        if not claim_tokens:
            return 1.0
        return len(claim_tokens & set(tokenize(evidence))) / len(claim_tokens)


def extract_numbers(text: str) -> list[tuple[float, int, bool]]:
    """``(value, decimals, is_percent)`` for each standalone number."""
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(1)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((float(raw), decimals, m.group(2) == "%"))
    return out


def extract_entities(text: str) -> set[str]:
    return set(_ENTITY.findall(text))


def _packet_numbers(obj: Any, acc: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            acc.append(float(obj))
    elif isinstance(obj, str):
        acc.extend(v for v, _, _ in extract_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _packet_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _packet_numbers(v, acc)


def _number_grounded(value: float, decimals: int, allowed: np.ndarray) -> bool:
    if decimals == 0 and value in _TRIVIAL_INTEGERS:
        return True
    plain = np.abs(np.round(allowed, decimals) - value) < 1e-9
    pct = np.abs(np.round(100 * allowed, decimals) - value) < 1e-9
    return bool(plain.any() or pct.any())


def _free_text(response: dict[str, Any]) -> list[str]:
    texts = [response.get("situation_summary", ""), response.get("threat_explanation", "")]
    texts += [e.get("statement", "") for e in response.get("evidence", [])]
    for s in response.get("recommended_steps", []):
        texts += [s.get("action", ""), s.get("rationale", "")]
    texts.append(response.get("confidence", {}).get("rationale", ""))
    texts += list(response.get("limitations", []))
    return [t for t in texts if t]


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def ungrounded_spans(response: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    """Numbers and identifiers in the response's free text that the packet does not contain."""
    allowed_numbers: list[float] = []
    _packet_numbers(packet, allowed_numbers)
    allowed = np.array(sorted(set(allowed_numbers)) or [np.nan])
    packet_entities = extract_entities(json.dumps(packet, ensure_ascii=False))
    n_numbers = 0
    bad_numbers: list[str] = []
    entities: set[str] = set()
    for text in _free_text(response):
        for value, decimals, pct in extract_numbers(text):
            n_numbers += 1
            if not _number_grounded(value, decimals, allowed):
                bad_numbers.append(f"{value:.{decimals}f}{'%' if pct else ''}")
        entities |= extract_entities(text)
    return {"n_numbers": n_numbers, "numbers": bad_numbers, "n_entities": len(entities),
            "entities": sorted(e for e in entities if e not in packet_entities)}


def audit_response(
    response: dict[str, Any],
    packet: dict[str, Any],
    support_threshold: float = 0.5,
    scorer: SupportScorer | None = None,
) -> dict[str, Any]:
    """Audit one structured response against the evidence packet it was given."""
    scorer = scorer or LexicalSupportScorer()
    retrieved = {k["chunk_id"]: k["text"] for k in packet.get("knowledge", [])}
    allowed_refs = set(packet.get("allowed_signal_refs", []))

    cited = valid_cited = 0
    claims_needing = claims_covered = 0
    supported = support_checked = 0
    refs = valid_refs = 0

    def check_claim(text: str, citations: Iterable[str], needs_citation: bool) -> None:
        nonlocal cited, valid_cited, claims_needing, claims_covered, supported, support_checked
        cites = list(citations)
        valid = [c for c in cites if c in retrieved]
        cited += len(cites)
        valid_cited += len(valid)
        if needs_citation:
            claims_needing += 1
            claims_covered += bool(valid)
        if valid:
            support_checked += 1
            evidence = " ".join(retrieved[c] for c in valid)
            supported += scorer(text, evidence) >= support_threshold

    for item in response.get("evidence", []):
        check_claim(item.get("statement", ""), item.get("citations", []),
                    item.get("source") == "knowledge_base")
        for ref in item.get("signal_refs", []):
            refs += 1
            valid_refs += ref in allowed_refs
    for step in response.get("recommended_steps", []):
        check_claim(f"{step.get('action', '')}", step.get("citations", []), True)

    spans = ungrounded_spans(response, packet)
    numbers, bad_numbers = spans["n_numbers"], len(spans["numbers"])
    bad_entities = spans["entities"]
    n_entities = spans["n_entities"]

    metrics = {
        "citation_precision": _ratio(valid_cited, cited),
        "fabricated_citations": cited - valid_cited,
        "citation_coverage": _ratio(claims_covered, claims_needing),
        "signal_ref_validity": _ratio(valid_refs, refs),
        "supported_claim_rate": _ratio(supported, support_checked),
        "numeric_hallucination_rate": _ratio(bad_numbers, numbers),
        "entity_hallucination_rate": _ratio(len(bad_entities), n_entities),
        "hallucinated_entities": bad_entities,
        "n_numbers": numbers,
        "n_entities": n_entities,
        "n_steps": len(response.get("recommended_steps", [])),
        "n_citations": cited,
    }
    components = [
        metrics["citation_precision"], metrics["citation_coverage"],
        metrics["signal_ref_validity"], metrics["supported_claim_rate"],
        1 - metrics["numeric_hallucination_rate"]
        if not math.isnan(metrics["numeric_hallucination_rate"]) else float("nan"),
        1 - metrics["entity_hallucination_rate"]
        if not math.isnan(metrics["entity_hallucination_rate"]) else float("nan"),
    ]
    finite = [c for c in components if not math.isnan(c)]
    metrics["groundedness"] = float(np.mean(finite)) if finite else float("nan")
    metrics["fully_grounded"] = bool(
        metrics["fabricated_citations"] == 0 and not bad_entities and bad_numbers == 0
        and (math.isnan(metrics["citation_coverage"]) or metrics["citation_coverage"] == 1.0)
    )
    return metrics


_MEAN_KEYS = ("citation_precision", "citation_coverage", "signal_ref_validity",
              "supported_claim_rate", "numeric_hallucination_rate", "entity_hallucination_rate",
              "groundedness")


def aggregate_audits(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-response audits (``records`` hold ``mode``, ``raw``, ``guarded``,
    ``violations`` and ``schema_valid``)."""

    def summarise(rows: list[dict[str, Any]], which: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        audits = [r[which] for r in rows if r.get(which)]
        for key in _MEAN_KEYS:
            vals = [a[key] for a in audits if not math.isnan(a[key])]
            out[key] = float(np.mean(vals)) if vals else None
        out["fabricated_citations_total"] = int(sum(a["fabricated_citations"] for a in audits))
        out["fully_grounded_rate"] = float(np.mean([a["fully_grounded"] for a in audits])) \
            if audits else None
        out["n"] = len(audits)
        return out

    llm_rows = [r for r in records if r.get("llm_called")]
    violations: dict[str, int] = {}
    for r in records:
        for v in r.get("violations", []):
            violations[v] = violations.get(v, 0) + 1
    by_mode: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in records}):
        rows = [r for r in records if r["mode"] == mode]
        by_mode[mode] = {"n": len(rows), "raw": summarise(rows, "raw"),
                         "guarded": summarise(rows, "guarded")}
    return {
        "n_events": len(records),
        "n_llm_calls": len(llm_rows),
        "schema_valid_rate": float(np.mean([r["schema_valid"] for r in llm_rows]))
        if llm_rows else None,
        "error_rate": float(np.mean([bool(r.get("error")) for r in records])) if records else None,
        "raw": summarise(records, "raw"),
        "guarded": summarise(records, "guarded"),
        "policy_violations": dict(sorted(violations.items())),
        "events_with_violations": int(sum(bool(r.get("violations")) for r in records)),
        "by_mode": by_mode,
    }
