"""Matplotlib style for journal figures.

* Vector output (PDF/SVG/PGF/EPS) with embedded TrueType fonts (``pdf.fonttype
  42``) - many publishers reject Type-3 fonts.
* Optional true LaTeX text rendering (``usetex``) when a TeX installation is
  found; otherwise Computer-Modern mathtext with the closest available serif.
* Deterministic files: creation dates are stripped and the SVG hash salt is
  fixed, so re-generating a figure from the same run yields identical bytes.
* Colours: categorical slots in a fixed, colour-vision-deficiency-validated
  order; an ordinal single-hue ramp for the ordered gate modes. Identity is
  never carried by colour alone (legends, markers, direct labels).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# fontTools warns about the zeroed font timestamps used for reproducible PDFs.
logging.getLogger("fontTools").setLevel(logging.ERROR)

__all__ = [
    "CATEGORICAL",
    "COLUMN_WIDTH",
    "GATE_COLORS",
    "INK",
    "METHOD_COLORS",
    "MODEL_COLORS",
    "MODEL_MARKERS",
    "TEXT_WIDTH",
    "apply_style",
    "latex_available",
    "save_figure",
]

COLUMN_WIDTH = 3.5  # inches, single column (IEEE/Elsevier two-column layouts)
TEXT_WIDTH = 7.16  # inches, full text width

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
               "#e34948"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984",
       "grid": "#e6e5e1", "axis": "#b9b8b2", "surface": "#ffffff", "neutral": "#9a9994"}

# Entities keep their colour in every figure.
MODEL_COLORS = {"detection": CATEGORICAL[0], "h5": CATEGORICAL[1], "h15": CATEGORICAL[2],
                "h30": CATEGORICAL[3]}
MODEL_MARKERS = {"detection": "o", "h5": "s", "h15": "^", "h30": "D"}
METHOD_COLORS = {"none": INK["neutral"], "temperature": CATEGORICAL[0],
                 "platt": CATEGORICAL[1], "isotonic": CATEGORICAL[2]}
# Ordered assistance levels on one ordinal ramp; abstain is a different kind of outcome.
GATE_COLORS = {"monitor": "#86b6ef", "explain": "#3987e5", "recommend": "#1c5cab",
               "recommend_with_approval": "#0d366b", "abstain": CATEGORICAL[1]}

_SERIF_CANDIDATES = ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman",
                     "Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]


def latex_available() -> bool:
    """True when a TeX engine usable by matplotlib is on PATH."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def _first_available_serif() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in _SERIF_CANDIDATES if n in names), "serif")


def apply_style(latex: Literal["auto", "on", "off"] = "auto", base_size: float = 8.0) -> bool:
    """Apply the journal style; returns whether LaTeX text rendering is active."""
    use_tex = latex == "on" or (latex == "auto" and latex_available())
    if latex == "on" and not latex_available():
        raise RuntimeError("latex='on' requested but no TeX installation was found")
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "cpscopilot",
        "font.family": "serif",
        "font.serif": [_first_available_serif(), "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.labelsize": base_size,
        "legend.fontsize": base_size - 1,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["primary"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "text.color": INK["primary"],
        "text.usetex": use_tex,
    })
    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    return use_tex


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    """Save a figure in every requested format with reproducible metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "pgf" and not latex_available():
            continue  # PGF export needs a TeX engine; skipped (recorded in the manifest)
        path = out_dir / f"{stem}.{fmt}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        metadata: dict[str, object] | None
        if fmt == "pdf":
            metadata = {"CreationDate": None, "ModDate": None, "Creator": "cpscopilot"}
        elif fmt == "svg":
            metadata = {"Date": None, "Creator": "cpscopilot"}
        else:
            metadata = None
        fig.savefig(path, format=fmt, metadata=metadata)
        written.append(path)
    plt.close(fig)
    return written
