"""Paper figures generated exclusively from saved run artefacts.

Figures never re-train or re-evaluate anything: they read ``metrics.json``,
``reliability.json``, ``curves.json``, ``predictions_test.csv.gz`` and
``copilot_audit.json`` from one or more run directories. With several runs
(typically seeds of one configuration) curves are averaged on their shared
grids and bars show the mean +/- one standard deviation across runs.

Every invocation writes into a *new* directory together with
``figures_manifest.json`` (source runs and their manifest hashes, output-file
hashes, rendering settings) and ``captions.md`` (draft captions with numbers).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt  # backend is fixed to Agg by plotting.style
import numpy as np
import pandas as pd

from cpscopilot.pipeline.aggregate import check_independent_runs
from cpscopilot.plotting.style import (
    COLUMN_WIDTH,
    GATE_COLORS,
    INK,
    METHOD_COLORS,
    MODEL_COLORS,
    MODEL_MARKERS,
    TEXT_WIDTH,
    apply_style,
    latex_available,
    save_figure,
)
from cpscopilot.utils.io import read_json, sha256_file
from cpscopilot.utils.run_manager import utc_stamp

__all__ = ["FIGURES", "RunArtifacts", "generate_all"]

MODE_ORDER = ["monitor", "explain", "recommend", "recommend_with_approval", "abstain"]
MODE_LABELS = {"monitor": "Monitor", "explain": "Explain", "recommend": "Recommend",
               "recommend_with_approval": "Recommend + approval", "abstain": "Abstain"}
METHOD_LABELS = {"none": "Uncalibrated", "temperature": "Temperature scaling",
                 "platt": "Platt scaling", "isotonic": "Isotonic regression"}
CATEGORY_LABELS = {"benign": "Benign", "imminent": "Attack imminent", "ongoing": "Attack ongoing"}

_USETEX = False


def _pct(x: float, digits: int = 0, plain: bool = False) -> str:
    sign = r"\%" if (_USETEX and not plain) else "%"
    return f"{100 * x:.{digits}f}{sign}"


def model_label(key: str) -> str:
    if key == "detection":
        return "Detection (now)"
    return f"+{key[1:]} min"


@dataclass
class RunArtifacts:
    """Read-only view of one run directory."""

    run_dir: Path
    metrics: dict[str, Any]
    reliability: dict[str, Any]
    curves: dict[str, Any]
    copilot_audit: dict[str, Any] | None
    _predictions: pd.DataFrame | None = None

    @classmethod
    def load(cls, run_dir: Path) -> RunArtifacts:
        run_dir = Path(run_dir)
        audit_path = run_dir / "copilot_audit.json"
        return cls(
            run_dir=run_dir,
            metrics=read_json(run_dir / "metrics.json"),
            reliability=read_json(run_dir / "reliability.json"),
            curves=read_json(run_dir / "curves.json"),
            copilot_audit=read_json(audit_path) if audit_path.exists() else None,
        )

    @property
    def predictions(self) -> pd.DataFrame:
        if self._predictions is None:
            self._predictions = pd.read_csv(self.run_dir / "predictions_test.csv.gz")
        return self._predictions

    @property
    def model_keys(self) -> list[str]:
        keys = list(self.metrics["models"])
        horizons = sorted((k for k in keys if k != "detection"), key=lambda k: float(k[1:]))
        return (["detection"] if "detection" in keys else []) + horizons

    @property
    def primary_method(self) -> str:
        return read_json(self.run_dir / "trust_state.json")["primary_method"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _fmt(mean: float, std: float, digits: int = 3, plain: bool = False) -> str:
    """``mean±std`` for figure text (math mode) or captions (``plain=True``)."""
    if not np.isfinite(mean):
        return "n/a"
    if std and std > 0:
        pm = "±" if plain else r"$\pm$"
        return f"{mean:.{digits}f}{pm}{std:.{digits}f}"
    return f"{mean:.{digits}f}"


def _cap(mean: float, std: float, digits: int = 3) -> str:
    return _fmt(mean, std, digits, plain=True)


def short_label(key: str) -> str:
    return "now" if key == "detection" else f"+{key[1:]}"


# ------------------------------------------------------------------ figures
def _pooled_reliability(runs: list[RunArtifacts], key: str, method: str) -> dict[str, np.ndarray]:
    """Pool reliability bins across runs (bins share edges): summed counts and
    count-weighted mean confidence / observed frequency per bin."""
    first = runs[0].reliability[key][method]
    lower, upper = np.array(first["lower"], dtype=float), np.array(first["upper"], dtype=float)
    count = np.zeros(len(lower))
    conf_sum = np.zeros(len(lower))
    freq_sum = np.zeros(len(lower))
    for r in runs:
        rel = r.reliability[key][method]
        c = np.array(rel["count"], dtype=float)
        count += c
        conf_sum += np.nan_to_num(np.array(rel["confidence"], dtype=float)) * c
        freq_sum += np.nan_to_num(np.array(rel["frequency"], dtype=float)) * c
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(count > 0, conf_sum / count, np.nan)
        freq = np.where(count > 0, freq_sum / count, np.nan)
    ece = float(np.nansum(count * np.abs(freq - conf)) / count.sum()) if count.sum() else np.nan
    return {"lower": lower, "upper": upper, "count": count, "confidence": conf,
            "frequency": freq, "ece": ece}


def _shared_methods(runs: list[RunArtifacts], key: str, primary: str) -> list[str]:
    return [m for m in dict.fromkeys(("none", primary))
            if all(m in r.reliability.get(key, {}) for r in runs)]


def fig_reliability(runs: list[RunArtifacts]) -> tuple[plt.Figure, str]:
    """Reliability diagrams before/after calibration, one column per model.

    Curves and histograms pool the bins of all runs; the in-panel ECE is the ECE
    of the plotted (pooled) curve, and the caption gives the mean +/- s.d. of the
    per-run ECEs.
    """
    run = runs[0]
    keys = run.model_keys
    primary = run.primary_method
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.6))
    grid = fig.add_gridspec(2, len(keys), height_ratios=[3.2, 1], hspace=0.08, wspace=0.28)
    caption_parts = []
    for j, key in enumerate(keys):
        ax = fig.add_subplot(grid[0, j])
        hist = fig.add_subplot(grid[1, j], sharex=ax)
        ax.plot([0, 1], [0, 1], color=INK["muted"], linewidth=0.7, zorder=1)
        methods = _shared_methods(runs, key, primary)
        notes, caps = [], []
        for method in methods:
            pooled = _pooled_reliability(runs, key, method)
            ok = pooled["count"] > 0
            ax.plot(pooled["confidence"][ok], pooled["frequency"][ok],
                    color=METHOD_COLORS.get(method, INK["secondary"]), marker="o",
                    label=METHOD_LABELS.get(method, method),
                    zorder=3 if method == primary else 2)
            short = "uncal." if method == "none" else "calib."
            notes.append(f"ECE {short} {pooled['ece']:.3f}")
            per_run = _mean_std([r.metrics["models"][key][method]["calibration"]["ece"]
                                 for r in runs])
            caps.append(f"{METHOD_LABELS.get(method, method).lower()} {_cap(*per_run)}")
        caption_parts.append(f"{model_label(key)}: per-run ECE " + ", ".join(caps))
        pooled = _pooled_reliability(runs, key, methods[-1])
        hist.bar(pooled["lower"], pooled["count"], width=pooled["upper"] - pooled["lower"],
                 align="edge", color=INK["neutral"], edgecolor=INK["surface"], linewidth=0.6)
        hist.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(model_label(key))
        ax.text(0.97, 0.04, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.8, color=INK["secondary"], linespacing=1.3, zorder=5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": INK["surface"],
                      "edgecolor": INK["grid"], "linewidth": 0.5, "alpha": 0.9})
        ax.tick_params(labelbottom=False)
        hist.set_xlabel("Predicted probability")
        if j == 0:
            ax.set_ylabel("Observed attack frequency")
            hist.set_ylabel("Count")
        hist.grid(False)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    caption = (f"Reliability diagrams on the test split before and after "
               f"{METHOD_LABELS[primary].lower()} fitted on the calibration split. Curves and "
               f"histograms pool the bins of {len(runs)} run(s) and the in-panel ECE is that of "
               "the pooled curve; bottom: samples per bin (log scale). "
               + "; ".join(caption_parts) + " (mean ± s.d. over runs).")
    return fig, caption


def _mean_curve(runs: list[RunArtifacts], key: str, method: str, kind: str) -> tuple:
    xs, ys = [], []
    for r in runs:
        c = r.curves.get(key, {}).get(method)
        if not c or not c.get("available"):
            continue
        if kind == "roc":
            xs.append(np.array(c["roc"]["fpr"]))
            ys.append(np.array(c["roc"]["tpr"]))
        else:
            xs.append(np.array(c["pr"]["recall"]))
            ys.append(np.array(c["pr"]["precision"]))
    if not ys:
        return None, None, None
    y = np.vstack(ys)
    return xs[0], y.mean(axis=0), y.std(axis=0, ddof=1) if len(ys) > 1 else None


def fig_roc_pr(runs: list[RunArtifacts]) -> tuple[plt.Figure, str]:
    """ROC and precision-recall curves for detection and every horizon."""
    run = runs[0]
    primary = run.primary_method
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.62, 2.3))
    caption_parts = []
    for key in run.model_keys:
        color, marker = MODEL_COLORS.get(key, INK["secondary"]), MODEL_MARKERS.get(key, "o")
        roc_auc = _mean_std([r.metrics["models"][key][primary]["classification"]["roc_auc"]
                             for r in runs])
        pr_auc = _mean_std([r.metrics["models"][key][primary]["classification"]["pr_auc"]
                            for r in runs])
        prev = _mean_std([r.metrics["models"][key][primary]["classification"]["prevalence"]
                          for r in runs])[0]
        for ax, kind, auc in ((axes[0], "roc", roc_auc), (axes[1], "pr", pr_auc)):
            x, y, s = _mean_curve(runs, key, primary, kind)
            if x is None:
                continue
            label = f"{short_label(key)} ({_fmt(*auc, digits=2)})"
            ax.plot(x, y, color=color, marker=marker, markevery=max(1, len(x) // 8), label=label)
            if s is not None:
                ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), color=color,
                                alpha=0.15, linewidth=0)
        caption_parts.append(f"{model_label(key)} ROC-AUC {_cap(*roc_auc)}, "
                             f"PR-AUC {_cap(*pr_auc)} (prevalence {prev:.3f})")
    axes[0].plot([0, 1], [0, 1], color=INK["muted"], linewidth=0.7)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC (AUC)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall (AP)")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=6,
                  title="Horizon (AUC)" if ax is axes[0] else "Horizon (AP)", title_fontsize=6)
    fig.tight_layout()
    caption = ("Discrimination on the chronologically held-out test split (calibrated scores; "
               "shaded: ±1 s.d. across runs). " + "; ".join(caption_parts) + ".")
    return fig, caption


def _stacked_rows(ax: plt.Axes, rows: list[str], fractions: np.ndarray) -> None:
    left = np.zeros(len(rows))
    y = np.arange(len(rows))
    for j, mode in enumerate(MODE_ORDER):
        widths = fractions[:, j]
        ax.barh(y, widths, left=left, height=0.62, color=GATE_COLORS[mode],
                edgecolor=INK["surface"], linewidth=1.0, label=MODE_LABELS[mode])
        for i, (w, l0) in enumerate(zip(widths, left)):
            if w >= 0.09:
                dark = mode in ("explain", "recommend", "recommend_with_approval", "abstain")
                ax.text(l0 + w / 2, i, _pct(w), ha="center", va="center", fontsize=6,
                        color=INK["surface"] if dark else INK["primary"])
        left += widths
    ax.set_yticks(y, rows)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _pct(v)))


def fig_gate_distribution(runs: list[RunArtifacts]) -> tuple[plt.Figure, str]:
    """Assistance-mode distribution by ground truth and by analyst profile."""
    run = runs[0]
    profiles = list(run.metrics["operational"]["gate"])
    baseline = profiles[0]
    cats = ["benign", "imminent", "ongoing"]
    by_cat = np.zeros((len(cats), len(MODE_ORDER)))
    by_profile = np.zeros((len(profiles), len(MODE_ORDER)))
    for r in runs:
        gate = r.metrics["operational"]["gate"]
        table = gate[baseline]["crosstab"]
        for i, c in enumerate(cats):
            counts = np.array([table[c][m] for m in MODE_ORDER], dtype=float)
            by_cat[i] += counts / max(counts.sum(), 1)
        for i, p in enumerate(profiles):
            by_profile[i] += np.array([gate[p]["mode_fractions"][m] for m in MODE_ORDER])
    by_cat /= len(runs)
    by_profile /= len(runs)
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH * 1.35, 2.9),
                             gridspec_kw={"height_ratios": [len(cats), len(profiles)]})
    _stacked_rows(axes[0], [CATEGORY_LABELS[c] for c in cats], by_cat)
    _stacked_rows(axes[1], [p.replace("_", " ") for p in profiles], by_profile)
    axes[0].set_title(f"(a) By ground truth ({baseline.replace('_', ' ')} analyst)", loc="left")
    axes[1].set_title("(b) By analyst profile (all test rows)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    def mean_of(field: str) -> str:
        m = _mean_std([r.metrics["operational"]["gate"][baseline][field] for r in runs])[0]
        return _pct(m, 1, plain=True) if np.isfinite(m) else "n/a"

    caption = (f"Assistance-gate decisions on the test split (mean over {len(runs)} run(s)). "
               f"Attack coverage (attack rows routed to any mode other than Monitor) "
               f"{mean_of('attack_coverage')}; benign rows routed to Explain or above "
               f"{_pct(1 - by_cat[0, 0], 1, plain=True)}; benign rows needing analyst attention "
               f"(approval or abstain) {mean_of('benign_attention_rate')}; abstentions "
               f"{mean_of('abstain_rate')}, with a detection error rate of "
               f"{mean_of('decision_error_rate_abstained')} inside abstentions vs "
               f"{mean_of('decision_error_rate_not_abstained')} outside.")
    return fig, caption


def fig_calibration_methods(runs: list[RunArtifacts]) -> tuple[plt.Figure, str]:
    """ECE and Brier score per model and calibration method."""
    run = runs[0]
    keys = run.model_keys
    methods = [m for m in ("none", "temperature", "platt", "isotonic")
               if m in run.metrics["models"][keys[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.62, 2.2))
    width = 0.8 / len(methods)
    x = np.arange(len(keys))
    for ax, metric, title in ((axes[0], "ece", "Expected calibration error"),
                              (axes[1], "brier", "Brier score")):
        for j, method in enumerate(methods):
            stats_ = [_mean_std([r.metrics["models"][k][method]["calibration"][metric]
                                 for r in runs]) for k in keys]
            means = np.array([s[0] for s in stats_])
            stds = np.array([s[1] for s in stats_])
            ax.bar(x - 0.4 + width * (j + 0.5), means, width=width, color=METHOD_COLORS[method],
                   edgecolor=INK["surface"], linewidth=0.8, label=METHOD_LABELS[method],
                   yerr=stds if len(runs) > 1 else None, ecolor=INK["secondary"],
                   error_kw={"elinewidth": 0.6, "capsize": 1.5})
        ax.set_xticks(x, [short_label(k) for k in keys])
        ax.set_xlabel("Horizon (minutes)")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), fontsize=6,
               bbox_to_anchor=(0.5, 1.08))
    caption = ("Calibration error on the test split for each post-hoc calibration method "
               f"(all fitted on the calibration split; {len(runs)} run(s)"
               + ("; error bars: ±1 s.d. across runs" if len(runs) > 1 else "") + ").")
    return fig, caption


def fig_risk_coverage(runs: list[RunArtifacts]) -> tuple[plt.Figure, str]:
    """Selective risk of the detection decision as coverage decreases."""
    names = {"raw_probability": ("Raw probability", INK["neutral"], "o"),
             "calibrated_probability": ("Calibrated probability", METHOD_COLORS["temperature"],
                                        "s"),
             "trust_adjusted": ("Trust-adjusted confidence", MODEL_COLORS["h5"], "^")}
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    parts = []
    for key, (label, color, marker) in names.items():
        curves = [r.metrics["operational"]["selective_prediction"][key] for r in runs]
        cov = np.array(curves[0]["coverage"])
        risk = np.mean([c["selective_risk"] for c in curves], axis=0)
        aurc = _mean_std([c["aurc"] for c in curves])
        # The raw curve is drawn wide underneath: monotone calibration often leaves the
        # ranking (and hence the curve) unchanged, and it must stay visible.
        wide = key == "raw_probability"
        ax.plot(cov, risk, color=color, marker=marker, markevery=(5, 10),
                linewidth=3.2 if wide else 1.4, zorder=1 if wide else 2,
                label=f"{label} (AURC {_fmt(*aurc, digits=4)})")
        parts.append(f"{label.lower()} AURC {_cap(*aurc, digits=4)}")
    oracle = _mean_std([r.metrics["operational"]["selective_prediction"]["trust_adjusted"]
                        ["optimal_aurc"] for r in runs])
    ax.set_xlabel("Coverage (fraction of decisions kept)")
    ax.set_ylabel("Selective error rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    caption = ("Risk-coverage curves of the thresholded detection decision ranked by three "
               "confidence scores (identical curves mean the score induces the same ranking); "
               + "; ".join(parts) + f"; oracle AURC {_cap(*oracle, digits=4)}.")
    return fig, caption


def _decision_rows(run: RunArtifacts) -> pd.DataFrame:
    """Test rows used for operational metrics (same mask as the runner) with the
    detection decision error of each row."""
    df = run.predictions
    ycols = [c for c in df.columns if c.startswith("y_h")]
    keep = df[ycols].notna().all(axis=1) | (df["y_now"] == 1)
    df = df.loc[keep]
    tau = run.metrics["trust"]["state_thresholds"]["malicious"]
    return df.assign(error=((df["p_detect"] >= tau).astype(int) != df["y_now"]))


def fig_uncertainty(runs: list[RunArtifacts]) -> tuple[plt.Figure, str]:
    """Detection-ensemble epistemic uncertainty of correct vs. erroneous detection
    decisions (ECDF over the rows of all runs)."""
    df = pd.concat([_decision_rows(r) for r in runs], ignore_index=True)
    errors = df["error"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))
    floor = 1e-6
    for mask, label, color, marker in ((~errors, "Correct decisions", METHOD_COLORS["temperature"],
                                        "o"), (errors, "Erroneous decisions", MODEL_COLORS["h5"],
                                               "s")):
        values = np.sort(np.clip(df.loc[mask, "epi_detection"].to_numpy(), floor, None))
        if len(values) == 0:
            continue
        ecdf = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, ecdf, where="post", color=color, label=f"{label} (n={len(values)})")
        ax.plot(values[:: max(1, len(values) // 8)], ecdf[:: max(1, len(values) // 8)],
                linestyle="none", marker=marker, color=color)
    auroc = _mean_std([r.metrics["operational"]["uncertainty"]["error_detection_auroc"]
                       ["epistemic"] for r in runs])
    ax.set_xscale("log")
    ax.set_xlabel("Detection-ensemble epistemic uncertainty (bits)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    caption = ("Empirical CDF of the detection ensemble's epistemic uncertainty (mutual "
               "information) for correct and erroneous detection decisions on the test split, "
               f"pooled over {len(runs)} run(s); values below {floor:g} bits are drawn at "
               f"{floor:g}. Error-detection AUROC of this uncertainty {_cap(*auroc)} "
               "(mean ± s.d. over runs).")
    return fig, caption


def fig_horizon(runs: list[RunArtifacts]) -> tuple[plt.Figure, str]:
    """PR-AUC and ECE as the prediction horizon grows (no dual axes: two panels)."""
    run = runs[0]
    keys = run.model_keys
    primary = run.primary_method
    x = np.arange(len(keys))
    labels = ["now" if k == "detection" else f"+{k[1:]}" for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.62, 2.6))

    def series(fn: Callable[[RunArtifacts, str], float | None]) -> tuple[np.ndarray, np.ndarray]:
        stats_ = [_mean_std([fn(r, k) for r in runs]) for k in keys]
        return np.array([s[0] for s in stats_]), np.array([s[1] for s in stats_])

    def cls(r: RunArtifacts, k: str, field: str) -> float:
        return r.metrics["models"][k][primary]["classification"][field]

    def proactive(field: str) -> Callable[[RunArtifacts, str], float | None]:
        def get(r: RunArtifacts, k: str) -> float | None:
            sub = r.metrics["models"][k][primary].get("proactive_subset")
            return sub["classification"][field] if sub else None
        return get

    # Each PR-AUC series is drawn with its own no-skill level (its prevalence).
    blue, orange = METHOD_COLORS["temperature"], MODEL_COLORS["h5"]
    for (m, s), label, color, marker, hollow in (
        (series(lambda r, k: cls(r, k, "pr_auc")), "All test rows", blue, "o", False),
        (series(lambda r, k: cls(r, k, "prevalence")), "No-skill, all test rows", blue, "o", True),
        (series(proactive("pr_auc")), "No attack in progress", orange, "s", False),
        (series(proactive("prevalence")), "No-skill, no attack in progress", orange, "s", True),
    ):
        ok = np.isfinite(m)
        axes[0].errorbar(x[ok], m[ok], yerr=s[ok] if len(runs) > 1 else None, color=color,
                         marker=marker, label=label, capsize=1.5, elinewidth=0.6,
                         linewidth=0.8 if hollow else 1.4,
                         markerfacecolor=INK["surface"] if hollow else color)
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("(a) Discrimination", loc="left")
    methods = [mm for mm in dict.fromkeys(("none", primary))
               if all(mm in r.metrics["models"][k] for r in runs for k in keys)]
    for method in methods:
        m, s = series(lambda r, k, mm=method: r.metrics["models"][k][mm]["calibration"]["ece"])
        axes[1].errorbar(x, m, yerr=s if len(runs) > 1 else None, color=METHOD_COLORS[method],
                         marker="o", label=METHOD_LABELS[method], capsize=1.5, elinewidth=0.6)
    axes[1].set_ylabel("ECE")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("(b) Calibration", loc="left")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Horizon (minutes)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=1, fontsize=6)
    fig.tight_layout(w_pad=2.5)
    caption = ("Performance versus prediction horizon. Left: PR-AUC on all test rows and on rows "
               "with no attack in progress (truly proactive prediction), each with its own "
               "no-skill level (its prevalence, hollow markers). Right: ECE before and after "
               "calibration.")
    return fig, caption


def fig_groundedness(runs: list[RunArtifacts]) -> tuple[plt.Figure, str] | None:
    """Copilot groundedness/hallucination audit: raw LLM output vs. guarded output."""
    audits = [r.copilot_audit for r in runs if r.copilot_audit]
    if not audits:
        return None
    items = [("citation_precision", "Citation precision", False),
             ("citation_coverage", "Citation coverage", False),
             ("signal_ref_validity", "Signal-reference validity", False),
             ("supported_claim_rate", "Supported claims", False),
             ("numeric_hallucination_rate", "Numbers grounded", True),
             ("entity_hallucination_rate", "Entities grounded", True),
             ("groundedness", "Groundedness (mean)", False)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.7, 2.3),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(items))
    h = 0.36
    for j, (which, label, color) in enumerate((("raw", "Raw LLM output", MODEL_COLORS["h5"]),
                                               ("guarded", "After guardrails",
                                                METHOD_COLORS["temperature"]))):
        vals = []
        for key, _, invert in items:
            v = _mean_std([a[which].get(key) for a in audits])[0]
            vals.append(1 - v if invert and np.isfinite(v) else v)
        vals_arr = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)
        axes[0].barh(y + (j - 0.5) * h, vals_arr, height=h, color=color, label=label,
                     edgecolor=INK["surface"], linewidth=0.8)
    axes[0].set_yticks(y, [label for _, label, _ in items])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(axis="y", visible=False)
    axes[0].set_title("(a) Audit scores (higher is better)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    violations: dict[str, float] = {}
    for a in audits:
        for k, v in a.get("policy_violations", {}).items():
            violations[k] = violations.get(k, 0) + v / len(audits)
    if violations:
        names = sorted(violations, key=violations.get, reverse=True)
        axes[1].barh(np.arange(len(names)), [violations[n] for n in names],
                     color=METHOD_COLORS["temperature"], height=0.6)
        axes[1].set_yticks(np.arange(len(names)), [n.replace("_", " ") for n in names])
        axes[1].invert_yaxis()
    else:
        axes[1].text(0.5, 0.5, "no interventions", ha="center", va="center",
                     transform=axes[1].transAxes, color=INK["secondary"])
        axes[1].set_yticks([])
        axes[1].set_xticks([])
        axes[1].spines["bottom"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
    axes[1].grid(axis="y", visible=False)
    axes[1].set_title("(b) Guardrail interventions per run", loc="left")
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=6,
               bbox_to_anchor=(0.5, 1.08))
    a0 = audits[0]
    backend = a0.get("backend", "?")
    fallbacks = a0.get("fallbacks", {})
    caption = (f"Copilot audit ({backend} backend, {a0['n_events']} events per run). Raw and "
               f"guarded scores are computed on the same {a0.get('n_generated', '?')} generated "
               "responses per run, so the difference is the effect of the guardrails alone; "
               f"fallback abstentions (abstain mode {fallbacks.get('abstain_mode', 0)}, LLM "
               f"errors {fallbacks.get('llm_error', 0)}, invalid output "
               f"{fallbacks.get('schema_invalid', 0)}) are excluded. Raw groundedness "
               f"{_cap(*_mean_std([a['raw']['groundedness'] for a in audits]))}, after "
               f"guardrails {_cap(*_mean_std([a['guarded']['groundedness'] for a in audits]))}; "
               f"fabricated citations in raw output "
               f"{sum(a['raw']['fabricated_citations_total'] for a in audits)}.")
    return fig, caption


FIGURES: dict[str, Callable[[list[RunArtifacts]], Any]] = {
    "fig_reliability_diagrams": fig_reliability,
    "fig_roc_pr_curves": fig_roc_pr,
    "fig_gate_distribution": fig_gate_distribution,
    "fig_calibration_methods": fig_calibration_methods,
    "fig_risk_coverage": fig_risk_coverage,
    "fig_uncertainty_ecdf": fig_uncertainty,
    "fig_horizon_performance": fig_horizon,
    "fig_copilot_groundedness": fig_groundedness,
}


def generate_all(run_dirs: list[Path], out_dir: Path | None = None,
                 formats: list[str] | None = None, latex: str = "auto",
                 only: list[str] | None = None, allow_duplicates: bool = False) -> Path:
    """Render every figure for the given runs into a new directory.

    Figures are rendered into a temporary sibling directory that is renamed to
    ``out_dir`` only when every figure succeeded, so a failure never leaves a
    partial output behind.
    """
    global _USETEX
    formats = formats or ["pdf", "svg"]
    runs = [RunArtifacts.load(Path(d)) for d in run_dirs]
    hashes = sorted({r.metrics["config_hash"] for r in runs})
    if len(hashes) > 1:
        raise ValueError(f"figures mix runs from different configurations: {hashes}")
    check_independent_runs([r.metrics for r in runs], allow_duplicates)
    ids = "|".join(sorted(r.metrics["run_id"] for r in runs))
    if out_dir is None:
        tag = hashlib.sha256(ids.encode()).hexdigest()[:8]
        root = runs[0].run_dir.parent.parent / "figures"
        out_dir = root / f"{utc_stamp()}_{runs[0].metrics['experiment']}_{tag}"
    final_dir = Path(out_dir)
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FileExistsError(f"{final_dir} is not empty; figures are never overwritten")
    out_dir = final_dir.with_name(final_dir.name + ".partial")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists (an interrupted earlier call?); remove it")
    out_dir.mkdir(parents=True)
    try:
        _render(runs, out_dir, formats, latex, only, hashes[0])
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    if final_dir.exists():
        final_dir.rmdir()  # empty (checked above)
    out_dir.rename(final_dir)
    return final_dir


def _render(runs: list[RunArtifacts], out_dir: Path, formats: list[str], latex: str,
            only: list[str] | None, config_hash: str) -> None:
    global _USETEX
    _USETEX = apply_style(latex)  # type: ignore[arg-type]
    written: dict[str, list[str]] = {}
    captions: list[str] = []
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        result = fn(runs)
        if result is None:
            continue
        fig, caption = result
        paths = save_figure(fig, out_dir, name, formats)
        written[name] = [p.name for p in paths]
        captions.append(f"## {name}\n\n{caption}\n")
    (out_dir / "captions.md").write_text("# Draft figure captions\n\n" + "\n".join(captions),
                                         encoding="utf-8")
    manifest = {
        "runs": [{"run_id": r.metrics["run_id"], "path": str(r.run_dir),
                  "manifest_sha256": sha256_file(r.run_dir / "manifest.json")
                  if (r.run_dir / "manifest.json").exists() else None} for r in runs],
        "config_hash": config_hash,
        "formats": formats,
        "pgf_skipped": "pgf" in formats and not latex_available(),
        "usetex": _USETEX,
        "matplotlib": matplotlib.__version__,
        "files": {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()},
    }
    (out_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                   encoding="utf-8")
