"""End-to-end experiment runner for the six-stage pipeline.

One call to :meth:`ExperimentRunner.run` produces one immutable run directory::

    runs/<UTC>_<name>_s<seed>_<hash8>/
        config.resolved.yaml     resolved configuration (after extends/overrides)
        environment.json         interpreter, packages, git commit
        seed_state.json          seeds applied
        data_fingerprint.json    dataset summary + raw-file SHA-256
        split_audit.json         chronological boundaries and embargo checks
        leakage_audit.json       causality / label / fit-ledger checks
        fit_ledger.json          every fit() call and the split it used
        trust_state.json         calibrators, thresholds, references
        gate_thresholds.json     resolved and analyst-adapted thresholds
        metrics.json             all metrics (with block-bootstrap CIs)
        reliability.json         reliability-diagram bins per model and method
        curves.json              ROC / PR curves
        predictions_test.csv.gz  per-row test outputs (trust signals, gate modes)
        copilot_outputs.jsonl    evidence packets' results, raw + guarded outputs
        copilot_audit.json       groundedness / hallucination audit
        knowledge_base.json      knowledge-base file hashes
        raw_data_integrity.json  raw files re-hashed after the run
        run.log                  full log
        manifest.json            SHA-256 of every file above (written last)
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cpscopilot.config import ExperimentConfig
from cpscopilot.copilot.audit import aggregate_audits
from cpscopilot.copilot.copilot import CognitiveCopilot
from cpscopilot.copilot.knowledge_base import kb_fingerprint, load_knowledge_base
from cpscopilot.copilot.llm import CachedLLM, FaultInjectingLLM, build_llm
from cpscopilot.copilot.retriever import BM25Retriever
from cpscopilot.data.features import CausalWindowFeaturizer, segment_ids
from cpscopilot.data.labels import horizon_to_steps, segmentwise_future_labels
from cpscopilot.data.leakage import (
    FitLedger,
    LeakageError,
    LeakageReport,
    audit_feature_causality,
    audit_feature_columns,
    audit_future_labels,
    index_fingerprint,
)
from cpscopilot.data.loaders import load_dataset
from cpscopilot.data.preprocessing import LeakageSafePreprocessor, TrainOnlyOneHotEncoder
from cpscopilot.data.schema import ATTACK_TYPE, LABEL, TIMESTAMP, CPSDataset
from cpscopilot.data.splitting import SPLIT_ORDER, TemporalSplitter
from cpscopilot.evaluation.bootstrap import block_bootstrap_ci, default_block_size
from cpscopilot.evaluation.calibration_metrics import (
    brier_score,
    calibration_report,
    expected_calibration_error,
)
from cpscopilot.evaluation.gate_metrics import gate_metrics, risk_coverage
from cpscopilot.evaluation.metrics import classification_metrics, compact_curves, pr_auc, roc_auc
from cpscopilot.gate.policy import AssistanceGate
from cpscopilot.models.detection import DetectionModule
from cpscopilot.models.prediction import FutureThreatPredictor, horizon_key
from cpscopilot.trust.asset import AssetRegistry
from cpscopilot.trust.evidence import RobustAnomalyScorer
from cpscopilot.trust.risk import DETECTION, ModelOutputs, TrustRiskLayer, select_state_thresholds
from cpscopilot.utils.io import environment_info, sha256_file
from cpscopilot.utils.logging import close_log_files, get_logger, setup_logging
from cpscopilot.utils.run_manager import RunContext, create_run
from cpscopilot.utils.seed import make_rng, seed_everything

__all__ = ["ExperimentRunner", "PreparedSplit", "RunResult", "find_cache", "prepare_split",
           "select_copilot_events"]

log = get_logger("pipeline.runner")


@dataclass
class PreparedSplit:
    """Features, labels and metadata of one split (warm-up rows removed)."""

    name: str
    row_ids: np.ndarray
    timestamps: pd.Series
    X: pd.DataFrame
    current: pd.DataFrame
    y_now: np.ndarray
    y_future: dict[str, np.ndarray]
    segments: np.ndarray
    meta: pd.DataFrame
    dropped_warmup: int

    @property
    def n(self) -> int:
        return len(self.row_ids)

    def labels(self) -> dict[str, np.ndarray]:
        return {DETECTION: self.y_now.astype(np.float64), **self.y_future}

    def complete_mask(self) -> np.ndarray:
        """Rows whose every future label is defined."""
        mask = np.ones(self.n, dtype=bool)
        for y in self.y_future.values():
            mask &= ~np.isnan(y)
        return mask

    def evaluation_mask(self) -> np.ndarray:
        """Rows usable for operational metrics: every future label defined, or an attack
        in progress (in ``onset`` mode those rows have undefined future labels by design,
        but they must still count as ongoing attacks)."""
        return self.complete_mask() | (self.y_now == 1)

    def future_any(self, keys: list[str], mask: np.ndarray) -> np.ndarray:
        """Attack within the longest horizon (undefined labels count as 0)."""
        return np.max(np.vstack([np.nan_to_num(self.y_future[k][mask], nan=0.0)
                                 for k in keys]), axis=0)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]


def prepare_split(
    name: str,
    frame: pd.DataFrame,
    base_cols: list[str],
    featurizer: CausalWindowFeaturizer,
    horizon_steps: dict[str, int],
    future_mode: str,
    sampling_seconds: float,
    max_gap_factor: float,
    meta_cols: list[str],
) -> PreparedSplit:
    """Features and labels computed strictly inside one split."""
    segs = segment_ids(frame[TIMESTAMP], sampling_seconds, max_gap_factor)
    X, valid = featurizer.transform(frame, base_cols, segs)
    current = frame[base_cols].astype(np.float64).groupby(segs).ffill(
        limit=featurizer.cfg.ffill_limit)
    y = frame[LABEL].to_numpy(dtype=np.int8)
    y_future = {k: segmentwise_future_labels(y, segs, h, future_mode)  # type: ignore[arg-type]
                for k, h in horizon_steps.items()}
    return PreparedSplit(
        name=name,
        row_ids=frame.index.to_numpy()[valid],
        timestamps=frame[TIMESTAMP][valid],
        X=X[valid],
        current=current[valid],
        y_now=y[valid],
        y_future={k: v[valid] for k, v in y_future.items()},
        segments=segs[valid],
        meta=frame.loc[valid, [c for c in meta_cols if c in frame]],
        dropped_warmup=int((~valid).sum()),
    )


def find_cache(llm: Any) -> CachedLLM | None:
    """The response cache inside an LLM wrapper chain (fault injection -> cache -> backend)."""
    layer = llm
    while not isinstance(layer, CachedLLM) and hasattr(layer, "inner"):
        layer = layer.inner
    return layer if isinstance(layer, CachedLLM) else None


def select_copilot_events(trust: pd.DataFrame, modes: np.ndarray, max_events: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Peak-risk row of each alert episode, stratified across gate modes."""
    modes = np.asarray(modes).astype(str)
    run_id = np.cumsum(np.r_[True, modes[1:] != modes[:-1]])
    risk = trust["risk"].to_numpy()
    candidates: dict[str, list[int]] = {}
    for rid in np.unique(run_id):
        pos = np.flatnonzero(run_id == rid)
        best = int(pos[np.argmax(risk[pos])])
        candidates.setdefault(modes[best], []).append(best)
    order = sorted(candidates)
    quota = {m: 0 for m in order}
    remaining = max_events
    pools = {m: list(candidates[m]) for m in order}
    while remaining > 0 and any(len(pools[m]) > quota[m] for m in order):
        for m in order:
            if remaining and len(pools[m]) > quota[m]:
                quota[m] += 1
                remaining -= 1
    chosen: list[int] = []
    for m in order:
        pool = np.array(pools[m])
        if quota[m]:
            chosen.extend(rng.choice(pool, size=quota[m], replace=False).tolist())
    return np.sort(np.array(chosen, dtype=int))


class ExperimentRunner:
    """Run one configured experiment and persist every artefact."""

    def __init__(self, config: ExperimentConfig, output_root: Path | None = None,
                 llm_client: Any | None = None) -> None:
        self.cfg = config
        self.output_root = output_root
        self.llm_client = llm_client

    # ------------------------------------------------------------------ data
    def _embargo(self, featurizer: CausalWindowFeaturizer, horizon_steps: dict[str, int]) -> int:
        if self.cfg.split.embargo_steps is not None:
            return self.cfg.split.embargo_steps
        return max(featurizer.warmup_steps + 1, max(horizon_steps.values()))

    def _prepare(self, dataset: CPSDataset, ledger: FitLedger, ctx: RunContext,
                 report: LeakageReport) -> tuple[dict[str, PreparedSplit], list[str],
                                                 dict[str, int], dict[str, Any]]:
        cfg = self.cfg
        featurizer = CausalWindowFeaturizer(cfg.features)
        horizon_steps = {horizon_key(h): horizon_to_steps(h, dataset.sampling_seconds)
                         for h in cfg.labels.horizons_minutes}
        embargo = self._embargo(featurizer, horizon_steps)
        split = TemporalSplitter(cfg.split, embargo).split(dataset.frame)
        split_audit = split.audit(dataset.sampling_seconds)
        ctx.save_json("split_audit.json", split_audit)
        report.add("split_boundaries", split_audit["passed"],
                   embargo_steps=embargo, gaps_seconds=split_audit["boundary_gaps_seconds"])

        frames = dict(split.frames)
        # Column usability is decided on the training split only.
        coverage = frames["train"][dataset.feature_cols].notna().mean()
        low_coverage = [c for c in dataset.feature_cols
                        if coverage[c] < cfg.features.min_train_coverage]
        if low_coverage:
            log.warning("dropping %d columns with train coverage < %.2f: %s", len(low_coverage),
                        cfg.features.min_train_coverage, low_coverage[:10])
        base_cols = [c for c in dataset.feature_cols if c not in set(low_coverage)]
        if not base_cols:
            raise ValueError("no input column has enough non-missing values in the train split")
        if dataset.categorical_cols:
            encoder = TrainOnlyOneHotEncoder(dataset.categorical_cols).fit(
                frames["train"], "train", ledger)
            for name in SPLIT_ORDER:
                frames[name] = pd.concat([frames[name], encoder.transform(frames[name])], axis=1)
            base_cols += encoder.output_columns()

        prepared = {
            name: prepare_split(name, frames[name], base_cols, featurizer, horizon_steps,
                                cfg.labels.future_mode, dataset.sampling_seconds,
                                cfg.dataset.max_gap_factor, dataset.meta_cols)
            for name in SPLIT_ORDER
        }
        for name, part in prepared.items():
            if part.y_now.sum() < max(1, cfg.split.min_positive_rows):
                raise LeakageError(f"{name} split has no attack rows after warm-up removal")

        rng = make_rng(cfg.seed, "leakage-audit")
        for name in SPLIT_ORDER:
            causality = audit_feature_causality(featurizer, frames[name], base_cols,
                                                cfg.runtime.leakage_probes, rng)
            report.add_result(f"feature_causality[{name}]", causality)
            part = prepared[name]
            for key, steps in horizon_steps.items():
                frame_y = frames[name][LABEL].to_numpy(dtype=np.int8)
                segs = segment_ids(frames[name][TIMESTAMP], dataset.sampling_seconds,
                                   cfg.dataset.max_gap_factor)
                full = segmentwise_future_labels(frame_y, segs, steps, cfg.labels.future_mode)
                check = audit_future_labels(frame_y, full, steps, cfg.labels.future_mode,
                                            cfg.runtime.leakage_probes, rng, segs)
                report.add_result(f"future_labels[{name}][{key}]", check)
            report.add(f"split_rows[{name}]", part.n > 0, n_rows=part.n,
                       dropped_warmup=part.dropped_warmup, n_attack_rows=int(part.y_now.sum()))
        info = {"embargo_steps": embargo, "horizon_steps": horizon_steps,
                "warmup_steps": featurizer.warmup_steps, "base_columns": base_cols,
                "dropped_low_train_coverage": low_coverage,
                "n_window_features": prepared["train"].X.shape[1]}
        return prepared, base_cols, horizon_steps, info

    # ------------------------------------------------------------ evaluation
    def _evaluate_models(self, trust: TrustRiskLayer, outputs: dict[str, ModelOutputs],
                         prepared: dict[str, PreparedSplit]) -> tuple[dict, dict, dict]:
        cfg = self.cfg
        ev = cfg.evaluation
        rng = make_rng(cfg.seed, "bootstrap")
        metrics: dict[str, Any] = {}
        reliability: dict[str, Any] = {}
        curves: dict[str, Any] = {}
        cal, te = prepared["calibration"], prepared["test"]
        for key in trust.model_keys:
            y_cal_all, y_te_all = cal.labels()[key], te.labels()[key]
            m_cal, m_te = outputs["calibration"].members(key), outputs["test"].members(key)
            mask_cal, mask_te = ~np.isnan(y_cal_all), ~np.isnan(y_te_all)
            y_cal = y_cal_all[mask_cal].astype(int)
            y_te = y_te_all[mask_te].astype(int)
            per_method: dict[str, Any] = {}
            reliability[key] = {}
            for method in trust.methods:
                p_cal = trust.calibrate(key, m_cal, method)[0][mask_cal]
                p_te = trust.calibrate(key, m_te, method)[0][mask_te]
                thr = select_state_thresholds(y_cal, p_cal, cfg.trust.suspicious_recall_target)
                report = calibration_report(y_te, p_te, ev.n_bins)
                reliability[key][method] = report.pop("reliability")
                per_method[method] = {
                    "classification": classification_metrics(y_te, p_te, thr["malicious"]),
                    "calibration": report,
                }
                if method in (trust.primary, "none"):
                    curves.setdefault(key, {})[method] = compact_curves(y_te, p_te, ev.curve_points)
                if method == trust.primary:
                    threshold = thr["malicious"]
                    block = ev.bootstrap_block_size or default_block_size(len(y_te))
                    per_method[method]["bootstrap"] = block_bootstrap_ci(
                        y_te, p_te,
                        {
                            "pr_auc": pr_auc,
                            "roc_auc": roc_auc,
                            "ece": lambda y, p: expected_calibration_error(y, p, ev.n_bins),
                            "brier": brier_score,
                            "f1_attack": lambda y, p, t=threshold: classification_metrics(
                                y, p, t)["f1_attack"],
                        },
                        ev.bootstrap_resamples, block, rng, ev.confidence_level)
                    if key != DETECTION:
                        benign_now = te.y_now[mask_te] == 0
                        if benign_now.any() and len(np.unique(y_te[benign_now])) == 2:
                            per_method[method]["proactive_subset"] = {
                                "description": "test rows with no attack in progress",
                                "classification": classification_metrics(
                                    y_te[benign_now], p_te[benign_now], threshold),
                                "calibration": {k: v for k, v in calibration_report(
                                    y_te[benign_now], p_te[benign_now], ev.n_bins).items()
                                    if k != "reliability"},
                            }
            metrics[key] = per_method
        return metrics, reliability, curves

    def _evaluate_trust_and_gate(self, trust: TrustRiskLayer, trust_test: pd.DataFrame,
                                 decisions: dict[str, pd.DataFrame], te: PreparedSplit,
                                 sampling_seconds: float) -> dict[str, Any]:
        mask = te.evaluation_mask()
        y_now = te.y_now[mask]
        y_any = te.future_any(trust.horizon_keys, mask)
        frame = trust_test[mask]
        assert trust.thresholds_ is not None
        decision = (frame["p_detect"].to_numpy() >= trust.thresholds_["malicious"]).astype(int)
        errors = decision != y_now
        hours = float(mask.sum() * sampling_seconds / 3600.0)
        gate = {pid: gate_metrics(d["mode"].to_numpy()[mask], y_now, y_any, errors, hours)
                for pid, d in decisions.items()}
        # Selective prediction of the thresholded detection decision. Every score ranks the
        # same decisions; "confidence" is the probability that the decision is correct.
        p_raw = frame["p_detect_raw"].to_numpy()
        p_cal = frame["p_detect"].to_numpy()
        conf_raw = np.where(decision == 1, p_raw, 1 - p_raw)
        conf_cal = np.where(decision == 1, p_cal, 1 - p_cal)
        assert trust.epistemic_reference_detection_ is not None
        epi_det = frame["epi_detection"].to_numpy()
        epi_rel = np.clip(epi_det / trust.epistemic_reference_detection_, 0, 1)
        conf_trust = conf_cal * (1 - epi_rel) * frame["cross_source"].to_numpy()
        selective = {
            "raw_probability": risk_coverage(errors, conf_raw),
            "calibrated_probability": risk_coverage(errors, conf_cal),
            "trust_adjusted": risk_coverage(errors, conf_trust),
        }
        err = errors.astype(int)
        uncertainty = {
            # Uncertainty of the detection ensemble itself vs. its own decision errors.
            "error_detection_auroc": {
                "epistemic": roc_auc(err, epi_det),
                "aleatoric": roc_auc(err, frame["alea_detection"].to_numpy()),
                "one_minus_trust_adjusted_confidence": roc_auc(err, 1 - conf_trust),
                "inconsistency": roc_auc(err, 1 - frame["cross_source"].to_numpy()),
            },
            # Gate-level signals (max over all models / risk driver), reported separately.
            "error_detection_auroc_gate_signals": {
                "epistemic_max_over_models": roc_auc(err, frame["epistemic"].to_numpy()),
                "aleatoric_risk_driver": roc_auc(err, frame["aleatoric"].to_numpy()),
                "inconsistency_combined": roc_auc(err, 1 - frame["consistency"].to_numpy()),
            },
            "mean_epistemic_detection_correct": float(epi_det[~errors].mean()),
            "mean_epistemic_detection_error": float(epi_det[errors].mean())
            if errors.any() else None,
            "decision_error_rate": float(errors.mean()),
        }
        states = {s: int((frame["state"] == s).sum()) for s in ("benign", "suspicious",
                                                                 "malicious")}
        return {"gate": gate, "selective_prediction": selective, "uncertainty": uncertainty,
                "state_counts": states, "n_rows": int(mask.sum()), "hours": hours}

    # ---------------------------------------------------------------- copilot
    def _run_copilot(self, ctx: RunContext, trust: TrustRiskLayer, trust_test: pd.DataFrame,
                     gate: AssistanceGate, decisions: pd.DataFrame, te: PreparedSplit,
                     scorer: RobustAnomalyScorer, registry: AssetRegistry,
                     base_cols: list[str]) -> dict[str, Any]:
        cfg = self.cfg
        kb_dir = cfg.resolve_path(cfg.copilot.knowledge_base_dir)
        chunks = load_knowledge_base(kb_dir, cfg.copilot.chunk_words,
                                     cfg.copilot.chunk_overlap_words)
        ctx.save_json("knowledge_base.json", {"directory": str(kb_dir), "n_chunks": len(chunks),
                                              "files": kb_fingerprint(kb_dir)})
        cache = cfg.resolve_path(cfg.copilot.llm.cache_dir) if cfg.copilot.llm.cache_dir else None
        llm = build_llm(cfg.copilot.llm, cache, cfg.seed, cfg.copilot.max_steps, self.llm_client)
        copilot = CognitiveCopilot(cfg.copilot, BM25Retriever(chunks), llm, registry,
                                   trust.horizon_minutes)
        profile = cfg.gate.analyst_profiles[0]
        modes = decisions["mode"].to_numpy()
        positions = select_copilot_events(trust_test, modes, cfg.copilot.max_events,
                                          make_rng(cfg.seed, "copilot-events"))
        z_all = scorer.z(te.current.iloc[positions])
        records = []
        for j, pos in enumerate(positions):
            row = trust_test.iloc[pos]
            decision = gate.decide(row, profile)
            if decision.mode.value != modes[pos]:
                raise RuntimeError("per-event and batch gate decisions diverged")
            order = np.argsort(-z_all[j], kind="mergesort")[: cfg.trust.top_k_features]
            deviations = [{"feature": base_cols[i], "z": round(float(z_all[j, i]), 3),
                           "value": round(float(te.current.iloc[pos, i]), 3),
                           "asset": registry.asset_of(base_cols[i])} for i in order]
            packet = copilot.build_packet(f"test-{int(te.row_ids[pos])}",
                                          te.timestamps.iloc[pos], row, decision, profile,
                                          deviations)
            records.append(copilot.respond(packet).to_record())
        ctx.save_jsonl("copilot_outputs.jsonl", records)
        audit = aggregate_audits(records)
        audit["backend"] = llm.name
        audit["analyst_profile"] = profile.analyst_id
        if isinstance(llm, FaultInjectingLLM):
            audit["fault_injection"] = {"rates": cfg.copilot.llm.fault_injection,
                                        "injected": llm.injected}
        cache_layer = find_cache(llm)
        if cache_layer is not None:
            audit["llm_cache"] = {"hits": cache_layer.hits, "misses": cache_layer.misses}
        ctx.save_json("copilot_audit.json", audit)
        return audit

    # -------------------------------------------------------------------- run
    def run(self) -> RunResult:
        """Execute the experiment; a failed run keeps its directory plus ``status.json``."""
        ctx = create_run(self.cfg, self.output_root)
        setup_logging(ctx.path("run.log"), self.cfg.runtime.log_level)
        log.info("run %s started", ctx.run_id)
        try:
            result = self._run(ctx)
        except Exception as exc:
            log.exception("run %s failed", ctx.run_id)
            close_log_files()
            ctx.save_json("status.json", {"status": "failed", "error_type": type(exc).__name__,
                                          "error": str(exc)})
            raise
        return result

    def _run(self, ctx: RunContext) -> RunResult:
        cfg = self.cfg
        timings: dict[str, float] = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round(now - clock, 3)
            clock = now

        project_root = Path(cfg.project_root) if cfg.project_root else None
        ctx.save_yaml("config.resolved.yaml", cfg.model_dump(mode="json"))
        ctx.save_json("environment.json", environment_info(project_root))
        ctx.save_json("seed_state.json", seed_everything(cfg.seed))
        ledger = FitLedger()
        report = LeakageReport()

        # Stage 1 - data
        dataset = load_dataset(cfg.dataset, cfg.resolve_path, cfg.seed)
        ctx.save_json("data_fingerprint.json", {"summary": dataset.summary(),
                                                "fingerprint": dataset.fingerprint})
        log.info("dataset %s: %s", dataset.source, dataset.summary())
        prepared, base_cols, horizon_steps, prep_info = self._prepare(dataset, ledger, ctx, report)
        tr, cal, te = prepared["train"], prepared["calibration"], prepared["test"]
        pre = LeakageSafePreprocessor("standard").fit(tr.X, "train", ledger)
        X = {name: pre.transform(p.X) for name, p in prepared.items()}
        report.add("preprocessor_fitted_on_train_rows",
                   pre.fit_fingerprint_ == index_fingerprint(tr.X.index))
        columns_check = audit_feature_columns(
            list(tr.X.columns), [LABEL, ATTACK_TYPE, TIMESTAMP, *dataset.meta_cols],
            tr.X, tr.y_now)
        report.add("no_label_or_meta_features", columns_check["passed"],
                   forbidden_columns_found=columns_check["forbidden_columns_found"])
        report.add("label_proxy_screen", not columns_check["label_proxies"], critical=False,
                   label_proxies=columns_check["label_proxies"][:20])
        lap("data")

        # Stages 2-3 - detection and prediction
        n_jobs = cfg.runtime.n_jobs
        detector = DetectionModule(cfg.detection, cfg.seed, n_jobs).fit(
            X["train"], tr.y_now, "train", ledger, tr.row_ids)
        predictor = FutureThreatPredictor(cfg.labels.horizons_minutes, cfg.prediction, cfg.seed,
                                          n_jobs).fit(X["train"], tr.y_future, "train", ledger,
                                                      tr.row_ids)
        outputs = {name: ModelOutputs(detector.predict_members(X[name]),
                                      predictor.predict_members(X[name]))
                   for name in ("calibration", "test")}
        lap("models")

        # Stage 4 - trust & risk
        scorer = RobustAnomalyScorer(cfg.trust.anomaly_tail_quantile).fit(
            tr.current, tr.y_now, "train", ledger)
        registry = AssetRegistry(cfg.trust.asset_rules, cfg.trust.default_criticality)
        horizon_minutes = {horizon_key(h): h for h in cfg.labels.horizons_minutes}
        trust = TrustRiskLayer(cfg.trust, cfg.calibration, horizon_minutes, registry, scorer,
                               base_cols)
        trust.fit_calibrators(outputs["calibration"], cal.labels(), ledger)
        trust_cal = trust.assess(outputs["calibration"], cal.current)
        trust.fit_reference(trust_cal, cal.y_now)
        trust_cal = trust.score(trust_cal)
        trust_test = trust.score(trust.assess(outputs["test"], te.current))
        ctx.save_json("trust_state.json", trust.state_dict())
        lap("trust")

        # Stage 5 - assistance gate (alert budget resolved on calibration benign rows)
        # Benign = complete labels, no attack now and none within the longest horizon.
        cal_future_any = cal.future_any(list(cal.y_future), np.ones(cal.n, dtype=bool))
        benign_cal = cal.complete_mask() & (cal.y_now == 0) & (cal_future_any == 0)
        gate = AssistanceGate(cfg.gate).fit(trust_cal, benign_cal)
        decisions = {p.analyst_id: gate.decide_batch(trust_test, p)
                     for p in cfg.gate.analyst_profiles}
        ctx.save_json("gate_thresholds.json", {
            "base": gate.base_.as_dict() if gate.base_ else None,
            "resolution": gate.resolution_,
            "profiles": {p.analyst_id: {"thresholds": gate.adapt(p)[0].as_dict(),
                                        "adjustments": gate.adapt(p)[1]}
                         for p in cfg.gate.analyst_profiles}})
        lap("gate")

        # Leakage gate before any metric is computed.
        report.add("nothing_fitted_on_test", not ledger.components_on("test"),
                   components=ledger.components_on("test"))
        train_only = [e for e in ledger.entries
                      if e["component"] in ("preprocessor", "anomaly_scorer",
                                            "categorical_encoder")
                      or e["component"].startswith("model:")]
        report.add("train_only_components_on_train", all(e["split"] == "train" for e in train_only))
        calibrators = [e for e in ledger.entries if e["component"].startswith("calibrator:")]
        report.add("calibrators_on_calibration_split",
                   all(e["split"] == "calibration" for e in calibrators))
        ctx.save_json("fit_ledger.json", ledger.to_dict())
        ctx.save_json("leakage_audit.json", report.to_dict())
        if cfg.runtime.strict_leakage_checks:
            report.raise_if_failed()

        # Evaluation
        model_metrics, reliability, curves = self._evaluate_models(trust, outputs, prepared)
        operational = self._evaluate_trust_and_gate(trust, trust_test, decisions, te,
                                                    dataset.sampling_seconds)
        lap("evaluation")

        # Stage 6 - copilot
        copilot_audit = None
        if cfg.copilot.enabled:
            primary = cfg.gate.analyst_profiles[0].analyst_id
            copilot_audit = self._run_copilot(ctx, trust, trust_test, gate, decisions[primary],
                                              te, scorer, registry, base_cols)
        lap("copilot")

        # Persist per-row test outputs
        table = pd.DataFrame({"row_id": te.row_ids, TIMESTAMP: te.timestamps.to_numpy(),
                              "y_now": te.y_now})
        for key, y in te.y_future.items():
            table[f"y_{key}"] = y
        for col in te.meta.columns:
            table[col] = te.meta[col].to_numpy()
        trust_cols = trust_test.reset_index(drop=True)
        table = pd.concat([table, trust_cols], axis=1)
        for pid, d in decisions.items():
            table[f"mode__{pid}"] = d["mode"].to_numpy()
            table[f"rule__{pid}"] = d["rule_id"].to_numpy()
        ctx.save_frame("predictions_test.csv.gz", table)
        ctx.save_json("reliability.json", reliability)
        ctx.save_json("curves.json", curves)
        if cfg.runtime.save_models:
            ctx.save_bytes("models.pkl", pickle.dumps(
                {"preprocessor": pre, "detector": detector, "predictor": predictor,
                 "trust": trust}, protocol=5))

        # Raw data must be unchanged by the run.
        integrity = []
        for f in dataset.fingerprint.get("files", []):
            now = sha256_file(Path(f["path"]))
            integrity.append({"path": f["path"], "sha256_before": f["sha256"],
                              "sha256_after": now, "unchanged": now == f["sha256"]})
        ctx.save_json("raw_data_integrity.json", {"files": integrity,
                                                  "all_unchanged": all(i["unchanged"]
                                                                       for i in integrity)})
        if not all(i["unchanged"] for i in integrity):
            raise RuntimeError("raw data files changed during the run")

        metrics = {
            "run_id": ctx.run_id,
            "config_hash": ctx.config_hash,
            "full_hash": ctx.full_hash,
            "experiment": cfg.name,
            "seed": cfg.seed,
            "dataset": dataset.summary(),
            "preparation": {**prep_info, "base_columns": len(prep_info["base_columns"]),
                            "rows": {n: p.n for n, p in prepared.items()},
                            "attack_rows": {n: int(p.y_now.sum()) for n, p in prepared.items()},
                            "preprocessor": pre.summary()},
            "leakage_audit_passed": report.passed,
            "models": model_metrics,
            "trust": {"state_thresholds": trust.thresholds_,
                      "epistemic_reference": trust.epistemic_reference_,
                      "epistemic_reference_detection": trust.epistemic_reference_detection_,
                      "temperatures": {k: v["temperature"].state_dict().get("temperature")
                                       for k, v in trust.calibrators.items()
                                       if "temperature" in v}},
            "operational": operational,
            "feature_importance": {
                "detection": detector.ensemble.feature_importance(pre.kept_),
                **{k: m.feature_importance(pre.kept_) for k, m in predictor.models.items()}},
            "copilot": copilot_audit,
            "timings_seconds": timings,
        }
        ctx.save_json("metrics.json", metrics)
        log.info("run %s finished: %s", ctx.run_id, ctx.run_dir)
        close_log_files()  # run.log must be complete before it is hashed and sealed
        ctx.finalize()
        return RunResult(run_id=ctx.run_id, run_dir=ctx.run_dir, metrics=metrics)
