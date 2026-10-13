"""Ensemble-based uncertainty decomposition (binary, in bits).

For member probabilities ``p_1..p_M`` and their mean ``p̄``:

* total (predictive) entropy   ``H[p̄]``
* aleatoric (expected entropy) ``E_m H[p_m]``  - irreducible data noise
* epistemic (mutual information / BALD) ``H[p̄] - E_m H[p_m] >= 0`` -
  disagreement between members, i.e. model ignorance (large off-distribution)

All three lie in ``[0, 1]`` bits for binary outcomes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["binary_entropy", "decompose_uncertainty"]

_EPS = 1e-12
_MIN_EPISTEMIC_BITS = 1e-9


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy in bits of a Bernoulli(p) variable (elementwise)."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def decompose_uncertainty(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose uncertainty from an ``(M, n)`` matrix of member probabilities."""
    probs = np.atleast_2d(np.asarray(member_probs, dtype=np.float64))
    mean = probs.mean(axis=0)
    total = binary_entropy(mean)
    aleatoric = binary_entropy(probs).mean(axis=0)
    epistemic = np.clip(total - aleatoric, 0.0, None)
    # Identical members leave ~1e-16 bits of rounding noise; it must never become a
    # data-driven abstention threshold, so anything below 1e-9 bits is exactly zero.
    epistemic = np.where(epistemic < _MIN_EPISTEMIC_BITS, 0.0, epistemic)
    return {
        "mean": mean,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "std": probs.std(axis=0),
    }
