#!/usr/bin/env python3
"""Run one experiment configuration (optionally for several seeds).

Thin wrapper around ``cpscopilot run`` so the pipeline can be used without
installing the package::

    python scripts/run_experiment.py --config configs/experiments/synthetic_smoke.yaml
    python scripts/run_experiment.py --config configs/experiments/swat.yaml --seeds 0 1 2 3 4
    python scripts/run_experiment.py --config configs/experiments/synthetic_full.yaml \
        --set copilot.llm.backend=anthropic --set copilot.max_events=100
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpscopilot.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
