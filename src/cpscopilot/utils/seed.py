"""Global seed management.

``seed_everything`` seeds every RNG the framework can touch (Python ``random``,
NumPy's legacy global RNG and, when installed, PyTorch) and returns a record
that is saved with each run. Scikit-learn and XGBoost have no global RNG: every
estimator in this code base receives an explicit ``random_state`` obtained from
:func:`derive_seed`, so component seeds are a pure function of the experiment
seed and a component name.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

__all__ = ["derive_seed", "make_rng", "seed_everything"]


def derive_seed(base_seed: int, *keys: object) -> int:
    """Deterministically derive a 31-bit child seed from ``base_seed`` and ``keys``.

    The derivation is a SHA-256 hash, so it is stable across Python processes
    (unlike ``hash()``, which is salted by ``PYTHONHASHSEED``).
    """
    material = "|".join([str(int(base_seed)), *map(str, keys)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def make_rng(base_seed: int, *keys: object) -> np.random.Generator:
    """Return an independent ``numpy.random.Generator`` for a named component."""
    return np.random.default_rng(derive_seed(base_seed, *keys))


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed all global RNGs and return a JSON-serialisable record of what was done.

    ``PYTHONHASHSEED`` is exported for child processes; it cannot change the
    hash salt of the already-running interpreter, which is why nothing in this
    code base relies on ``hash()`` ordering.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    record: dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy_global": True,
        "PYTHONHASHSEED": str(seed),
        "torch": None,
    }
    try:  # optional dependency
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return record

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    record["torch"] = {"version": torch.__version__, "deterministic": deterministic_torch}
    return record
