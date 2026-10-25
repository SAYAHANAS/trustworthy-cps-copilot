#!/usr/bin/env python3
"""Generate publication-grade vector figures from saved experiment runs.

The figures are rebuilt purely from run artefacts (no retraining), written to a
new directory with a provenance manifest and draft captions.

Usage::

    python scripts/generate_paper_figures.py runs/<run_id> [runs/<run_id> ...] \
        [--out paper/figures/v1] [--formats pdf svg pgf] [--latex auto|on|off] \
        [--only fig_reliability_diagrams fig_roc_pr_curves]

Pass several runs of the same configuration (e.g. five seeds) to plot means with
+/-1 s.d. bands. PGF export requires a TeX installation; it is skipped otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpscopilot.plotting.figures import FIGURES, generate_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--out", type=Path, help="new output directory")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg"],
                        choices=["pdf", "svg", "pgf", "eps", "png"])
    parser.add_argument("--latex", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="subset of figures")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="accept repeated seeds within one configuration (replicas "
                             "are not independent samples)")
    args = parser.parse_args(argv)
    out = generate_all(args.runs, args.out, formats=args.formats, latex=args.latex,
                       only=args.only, allow_duplicates=args.allow_duplicates)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
