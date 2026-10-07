"""Trustworthy proactive security operations for cyber-physical systems.

A six-stage research pipeline:

1. CPS data layer (ingestion, causal temporal preprocessing)
2. Detection module (current security state)
3. Future threat prediction (t+5 / t+15 / t+30 minutes)
4. Trust & risk layer (calibration, uncertainty, evidence consistency, asset impact)
5. Assistance gate (Monitor / Explain / Recommend / Recommend+Approval / Abstain)
6. RAG + LLM copilot (grounded, policy-constrained decision support)

The LLM is never the detector: stages 2-4 produce structured evidence and the
copilot only turns that evidence into controlled decision support for a human
SOC analyst, who makes the final decision.
"""

__version__ = "0.1.0"
