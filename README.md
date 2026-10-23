# Trustworthy Proactive Security Operations — experimental framework

Reproducible research code for **"Toward Trustworthy Proactive Security Operations:
A Risk- and Confidence-Calibrated Cognitive LLM Copilot for Cyber-Physical Systems."**

```
CPS data ─► Detection ─► Future threat ─► Trust & risk ─► Assistance ─► RAG + LLM ─► Human SOC
 (stage 1)   (stage 2)    prediction (3)    layer (4)      gate (5)     copilot (6)    analyst
```

**The LLM is never the detector.** Gradient-boosted ensembles estimate the current
state and the probability of an attack within 5/15/30 minutes; the trust layer
calibrates those probabilities and quantifies uncertainty, evidence consistency and
asset impact; a policy gate decides how much assistance is allowed; the copilot only
turns that structured evidence into grounded, cited, policy-constrained decision
support. The analyst makes every decision.

| Stage | Module | What it does |
|---|---|---|
| 1 CPS data | `cpscopilot.data` | SWaT / WADI / TON_IoT / generic CSV / synthetic loaders, causal resampling, chronological split with embargo, trailing-window features, future labels, train-only preprocessing, leakage auditor |
| 2 Detection | `cpscopilot.models.detection` | seeded XGBoost ensemble for *attack in progress now* (backends: `xgboost`, `hist_gb`, `random_forest`, `logistic`) |
| 3 Prediction | `cpscopilot.models.prediction` | one ensemble per horizon for *attack within (t, t+h]* |
| 4 Trust & risk | `cpscopilot.trust` | temperature scaling (+ Platt / isotonic baselines), epistemic/aleatoric decomposition, robust anomaly evidence, cross-source and temporal consistency, asset criticality, risk and confidence, benign/suspicious/malicious state |
| 5 Assistance gate | `cpscopilot.gate` | ordered, auditable rules → Monitor / Explain / Recommend / Recommend + approval / Abstain; analyst-adaptive thresholds; alert budget |
| 6 Copilot | `cpscopilot.copilot` | BM25 retrieval over OT playbooks, structured output schema, Claude or deterministic backend, guardrails, groundedness / hallucination audit |
| Evaluation | `cpscopilot.evaluation` | PR-AUC, balanced accuracy, F1-macro, MCC, ECE/MCE/aECE/Brier/NLL, reliability bins, block-bootstrap CIs, gate and selective-prediction metrics |
| Orchestration | `cpscopilot.pipeline`, `cpscopilot.cli` | experiment runner, multi-seed aggregation, CLI |
| Figures | `cpscopilot.plotting`, `scripts/generate_paper_figures.py` | vector figures from saved runs only |

## Quick start

```bash
pip install -r requirements-lock.txt      # exact tested versions (or: pip install -e ".[dev,llm]")
make test                                 # 192 tests, ~2 min
make smoke                                # one end-to-end run on the synthetic plant
python scripts/generate_paper_figures.py runs/<run_id> --formats pdf svg pgf
```

Everything runs offline by default: the synthetic plant needs no downloads, and the
copilot uses the deterministic `template` backend. Real datasets go under
`data/raw/` (see [`data/README.md`](data/README.md) for the exact files and formats).

The whole study (audit, every seed, aggregation, figures) is one command, on Linux,
macOS and Windows alike; finished runs are reused when it is restarted. Windows
users: see [`docs/WINDOWS.md`](docs/WINDOWS.md).

```bash
python scripts/run_study.py --datasets swat wadi --audit-only      # check the data first
python scripts/run_study.py --datasets synthetic swat wadi --seeds 0 1 2 3 4
python scripts/run_study.py --datasets swat wadi --backend anthropic   # Claude copilot
```

```bash
# five seeds of one configuration + aggregate table (mean, s.d., 95 % t-interval)
python scripts/run_experiment.py --config configs/experiments/swat.yaml --seeds 0 1 2 3 4
# dotted overrides on the command line
python scripts/run_experiment.py --config configs/experiments/synthetic_full.yaml \
    --set calibration.method=platt --set gate.explain_benign_budget=0.05
# data + split + leakage checks only (no training)
PYTHONPATH=src python -m cpscopilot.cli audit --config configs/experiments/swat.yaml
# re-hash a run against its manifest
PYTHONPATH=src python -m cpscopilot.cli verify runs/<run_id>
```

## Anti-leakage design

| Risk | Mechanism | Verified by |
|---|---|---|
| Shuffled or overlapping splits | contiguous chronological blocks, train → calibration → test, with `embargo_steps` purged rows between them (default: max(feature warm-up + 1, longest horizon)) | `split_audit.json`; `tests/test_splitting.py` |
| Rolling windows reading the future | trailing windows only (`center=False`, full-window `min_periods`), causal forward-fill, features computed *inside* each split and contiguous segment; warm-up rows dropped | runtime truncation + future-perturbation audit on random rows of every split (`leakage_audit.json`); `tests/test_features_no_leakage.py` |
| Resampling across outages or before data exist | right-closed bins labelled at their right edge; runs of empty bins longer than `max_gap_factor` periods are dropped (never filled with fake benign rows) so a new segment starts; flow records (e.g. TON_IoT/Zeek) are binned at `ts + duration`, when they become observable (`event_duration_col`), while attack intervals are matched to start times; for event logs (`record_kind: event`) a silent bin is an attack only between attack bins | `fingerprint.resampling` in `data_fingerprint.json`; `tests/test_loaders.py` |
| Labels crossing the split boundary | future labels use the strictly future in-segment window `(t, t+h]`; rows whose horizon leaves the split are undefined and dropped | brute-force label audit; `tests/test_labels.py` |
| Statistics fitted on held-out data | imputer, scaler, one-hot vocabulary, models and anomaly scorer refuse to fit on anything but `train`; columns with too little data are dropped based on train coverage only (`features.min_train_coverage`, over bins that contain records; a sum is NaN when no record carries the field); calibrators only on `calibration`; a fit ledger records every fit | `fit_ledger.json`, audit check `nothing_fitted_on_test`; `tests/test_preprocessing.py` |
| Label/metadata used as input | metadata columns (`attack_type`, `phase`, …) cannot be features; label-proxy screen on train | `CPSDataset` validation, audit check `no_label_or_meta_features` |
| Test data tuning thresholds | state thresholds, gate thresholds, alert budget and the epistemic reference are chosen on the calibration split | `trust_state.json`, `gate_thresholds.json` |
| Ground truth reaching the copilot | evidence packets contain model outputs and sensor deviations only; the retrieval index is built from the knowledge base only | `tests/test_copilot.py::test_packet_never_contains_ground_truth` |

With `runtime.strict_leakage_checks: true` (default) any failed critical check aborts
the run *before* metrics are computed.

## Reproducibility and data integrity

* **Ensemble diversity** — backends without internal randomness (`hist_gb`, `logistic`,
  XGBoost without subsampling) automatically use the seeded block bootstrap, and an
  ensemble whose members predict identically is rejected, so epistemic uncertainty is
  never rounding noise. Step-function calibrators (isotonic) are not applied member-wise;
  the decomposition then uses the raw members.
* **Seeds** — `seed_everything` seeds Python, NumPy and (if installed) PyTorch; every
  estimator receives an explicit seed derived by SHA-256 from the experiment seed and
  its component name. Same seed ⇒ identical metrics and byte-identical prediction
  files (tested).
* **Run IDs** — `runs/<UTC>_<name>_s<seed>_<config-hash>`; the config hash excludes
  the seed so seeds of one configuration aggregate together.
* **Never overwrite** — every artifact is written with exclusive-create semantics; a
  finished run gets `manifest.json` (SHA-256 of every file) and read-only files;
  `cpscopilot verify` detects tampering. Failed runs keep their directory plus
  `status.json`.
* **Raw data** — raw files are fingerprinted before loading and re-hashed after the run
  (`raw_data_integrity.json`); the run fails if they changed.
* **Provenance** — `config.resolved.yaml`, `environment.json` (packages, git commit and
  dirty flag), `seed_state.json`, `knowledge_base.json` (playbook hashes).
* **LLM outputs** — API models are not bit-reproducible, so responses are stored in an
  append-only cache keyed by the SHA-256 of the full request (`llm_cache/`); re-running
  with the cache reproduces copilot outputs exactly. Cache hits/misses go to
  `llm_cache_stats.json`, not to `metrics.json`, so a warm cache does not change the metrics.

## Trust layer and gate in brief

* `p_detect`, `p_h5/15/30`: calibrated (temperature scaling by default) on the
  calibration split; Platt and isotonic are evaluated alongside as baselines.
* Uncertainty (bits): epistemic = ensemble mutual information, aleatoric = expected
  member entropy.
* Consistency: `0.7·(1 − |p_detect − anomaly_evidence|) + 0.3·temporal coherence`
  (nested horizons must be monotone).
* `risk = max(p_detect, max_h w_h·p_h) × criticality`;
  `confidence = max(p*, 1 − p*) × (1 − epistemic_rel) × consistency`.
* Gate rules (first match): R1 abstain on epistemic uncertainty → R2 abstain on
  inconsistent evidence → R3/R4 recommend + approval on high risk or critical asset →
  R5 recommend → R6 explain → R7 monitor. Explain/recommend thresholds are raised to
  meet an **alert budget** (default ≤ 10 % / 2 % of benign calibration rows). Analyst
  profiles (expertise, workload, shift, override rate) adapt thresholds transparently,
  e.g. every recommendation needs approval for junior analysts.

## Copilot

* Retrieval: deterministic BM25 over `knowledge_base/*.md` (OT-safety-first playbooks
  mapped to MITRE ATT&CK for ICS); chunk ids are stable citations.
* Output: `CopilotResponse` (summary, explanation, evidence with signal references and
  citations, recommended steps with approval flags, confidence, limitations), enforced
  as a strict JSON schema.
* Backends: `template` (deterministic extractive baseline, default) and `anthropic`
  (Claude via the official SDK with structured outputs; default model
  `claude-opus-5-5`, `effort: medium`, server-side refusal fallback enabled via
  `fallbacks: "default"` — disable with `copilot.llm.use_refusal_fallback=false`).
  Set `ANTHROPIC_API_KEY` (or log in with `ant auth login`) and run with
  `--set copilot.llm.backend=anthropic`.
* Guardrails (`enforce_policy`): mode-forbidden content removed (including evidence that
  quotes containment guidance in modes without recommendations), uncited steps dropped,
  fabricated citations and unknown signal references removed, approval forced for steps
  that cite containment/response/recovery guidance, for analyst profiles and modes that
  require it, and for autonomous-action phrasing in the action or its rationale
  (checked clause by clause; negated clauses such as "do not restart without approval"
  are not flagged);
  confidence overwritten with the trust layer's value; ungrounded numbers/identifiers
  flagged. Abstain mode never calls the LLM, and LLM failures fall back to an abstention
  whose text does not contain the raw error.
* Audit: citation precision/coverage, signal-reference validity, lexically supported
  claims, numeric and entity hallucination rates, groundedness. `raw` and `guarded` are
  computed on the same generated responses, so their difference is the guardrails'
  effect; abstentions and failed calls are reported separately (`fallbacks`,
  `guarded_all_events`). `configs/experiments/synthetic_fault_injection.yaml` injects
  seeded faults to measure the audit's sensitivity.

## Outputs of one run

`config.resolved.yaml`, `environment.json`, `seed_state.json`, `data_fingerprint.json`,
`split_audit.json`, `leakage_audit.json`, `fit_ledger.json`, `trust_state.json`,
`gate_thresholds.json`, `metrics.json`, `reliability.json`, `curves.json`,
`predictions_test.csv.gz`, `copilot_outputs.jsonl`, `copilot_audit.json`,
`knowledge_base.json`, `raw_data_integrity.json`, `run.log`, `manifest.json`.

## Figures

`scripts/generate_paper_figures.py` builds, from saved runs only (several seeds ⇒
mean ± s.d.): reliability diagrams before/after calibration, ROC and PR curves,
gate-decision distribution by ground truth and analyst profile, calibration-method
comparison, risk–coverage curves, epistemic-uncertainty ECDFs, performance vs.
horizon, and the copilot audit. PDF/SVG (fonts embedded as TrueType, deterministic
metadata), PGF when a TeX installation is present (`--latex on` for TeX text).
Each output directory also holds `figures_manifest.json` and draft `captions.md`.

## Layout

```
.
├── configs/default.yaml, configs/experiments/*.yaml
├── knowledge_base/*.md            RAG playbooks (illustrative, validate with site SOPs)
├── data/raw/                      raw datasets (git-ignored, read-only)
├── scripts/run_experiment.py, scripts/generate_paper_figures.py
├── src/cpscopilot/{data,models,trust,gate,copilot,evaluation,pipeline,plotting,utils}
└── tests/                         pytest suite
```

## Limitations and threats to validity

* The synthetic plant is for testing and controlled ablations, not a benchmark; report
  results on SWaT/WADI/TON_IoT. With few attack campaigns, horizon models can memorise
  campaign context instead of attack mechanics (visible on `synthetic_smoke`).
* Post-hoc calibration assumes the calibration period resembles the test period; under
  temporal drift it can *increase* test ECE — the framework reports this rather than
  hiding it (compare `none` vs. `temperature` in `metrics.json`).
* Claim support is lexical; plug an NLI model into `SupportScorer` for stronger
  groundedness checks. Number grounding compares magnitudes (a wrong sign is not
  detected) and treats percentages only as renderings of fractions in [0, 1]. The template backend is grounded by construction and serves as
  a baseline, not as evidence about LLM behaviour.
* Asset criticalities and gate thresholds in the configs are illustrative and must be
  set with plant operations.
* Dataset labels are used as distributed. The 2019 WADI labels start about 3 s late and
  place attack 13 twelve hours early (see `data/README.md`); TON_IoT network data is not
  contiguous in time, mostly attacks, and has one time window per attack type, so its
  chronological test split measures generalisation to unseen attack types.
