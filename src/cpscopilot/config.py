"""Typed experiment configuration.

Configurations are YAML files validated against the pydantic schema below.
A config may inherit from another with ``extends: <relative path>``; mappings
are deep-merged (child wins) and lists are replaced. Dotted command-line
overrides (``detection.ensemble_size=3``) are applied last.

Two hashes identify a configuration:

* ``config_hash`` covers every scientific setting *except* the seed and output
  location, so runs that differ only by seed share it and can be aggregated.
* ``full_hash`` additionally covers the seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AnalystProfile",
    "AssetRule",
    "CalibrationConfig",
    "CopilotConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureConfig",
    "GateConfig",
    "LLMConfig",
    "LabelConfig",
    "ModelConfig",
    "RuntimeConfig",
    "SplitConfig",
    "SyntheticConfig",
    "TrustConfig",
    "apply_overrides",
    "config_hash",
    "find_project_root",
    "load_config",
]


class _Strict(BaseModel):
    """Base model that rejects unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- data
class SyntheticConfig(_Strict):
    """Parameters of the physics-inspired synthetic water-treatment generator."""

    days: float = Field(2.0, gt=0)
    sampling_seconds: int = Field(10, gt=0)
    n_attacks: int = Field(20, ge=1)
    attack_duration_minutes: tuple[float, float] = (5.0, 25.0)
    precursor_lead_minutes: tuple[float, float] = (8.0, 35.0)
    precursor_probability: float = Field(0.8, ge=0, le=1)
    false_precursor_rate_per_day: float = Field(4.0, ge=0)
    missing_rate: float = Field(0.001, ge=0, lt=0.5)
    drift_strength: float = Field(0.0, ge=0)
    attack_types: list[str] = Field(
        default_factory=lambda: [
            "sensor_spoofing",
            "actuator_manipulation",
            "network_dos",
            "setpoint_tampering",
        ]
    )
    start: str = "2026-01-05T00:00:00"


class DatasetConfig(_Strict):
    """Where and how to read the CPS telemetry."""

    name: Literal["synthetic", "swat", "wadi", "ton_iot", "generic_csv"] = "synthetic"
    paths: list[str] = Field(default_factory=list)
    timestamp_col: str | None = None
    timestamp_cols: list[str] = Field(default_factory=list)
    timestamp_format: str | None = None
    timestamp_unit: Literal["s", "ms", "us", "ns"] | None = None
    dayfirst: bool = False
    label_col: str | None = None
    positive_labels: list[Any] = Field(default_factory=list)
    attack_type_col: str | None = None
    feature_include: list[str] | None = None
    feature_exclude: list[str] = Field(default_factory=list)
    categorical_cols: list[str] = Field(default_factory=list)
    count_distinct_cols: list[str] = Field(default_factory=list)
    resample_seconds: int | None = Field(None, gt=0)
    resample_aggs: list[Literal["mean", "sum", "max", "min", "std", "last"]] = Field(
        default_factory=lambda: ["mean"]
    )
    sampling_seconds: float | None = Field(None, gt=0)
    max_gap_factor: float = Field(5.0, gt=1)
    max_rows: int | None = Field(None, gt=0)
    attack_intervals_path: str | None = None
    read_csv_kwargs: dict[str, Any] = Field(default_factory=dict)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)


class SplitConfig(_Strict):
    """Chronological train / calibration / test split with embargo gaps."""

    strategy: Literal["fraction", "timestamp"] = "fraction"
    train_frac: float = Field(0.6, gt=0, lt=1)
    calibration_frac: float = Field(0.2, gt=0, lt=1)
    test_frac: float = Field(0.2, gt=0, lt=1)
    train_end: str | None = None
    calibration_end: str | None = None
    embargo_steps: int | None = Field(
        None, ge=0, description="Rows discarded between splits; None = auto (max window/horizon)."
    )
    min_positive_rows: int = Field(1, ge=0)

    @model_validator(mode="after")
    def _check(self) -> SplitConfig:
        if self.strategy == "fraction":
            total = self.train_frac + self.calibration_frac + self.test_frac
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"split fractions must sum to 1, got {total:.6f}")
        elif self.train_end is None or self.calibration_end is None:
            raise ValueError("timestamp split requires train_end and calibration_end")
        return self


class FeatureConfig(_Strict):
    """Causal (trailing) rolling-window features."""

    windows: list[int] = Field(default_factory=lambda: [6, 30])
    stats: list[Literal["mean", "std", "min", "max", "range", "slope"]] = Field(
        default_factory=lambda: ["mean", "std", "min", "max"]
    )
    include_raw: bool = True
    diff_lags: list[int] = Field(default_factory=lambda: [1])
    ffill_limit: int | None = Field(None, ge=1)

    @field_validator("windows")
    @classmethod
    def _windows(cls, v: list[int]) -> list[int]:
        if not v or any(w < 2 for w in v):
            raise ValueError("windows must be a non-empty list of integers >= 2")
        return sorted(set(v))

    @field_validator("diff_lags")
    @classmethod
    def _lags(cls, v: list[int]) -> list[int]:
        if any(lag < 1 for lag in v):
            raise ValueError("diff_lags must be >= 1")
        return sorted(set(v))

    @property
    def warmup_steps(self) -> int:
        """Rows of history required before a feature vector is complete."""
        return max(max(self.windows) - 1, max(self.diff_lags, default=0))


class LabelConfig(_Strict):
    """Current-state and future-horizon targets."""

    horizons_minutes: list[float] = Field(default_factory=lambda: [5.0, 15.0, 30.0])
    future_mode: Literal["any", "onset"] = "any"

    @field_validator("horizons_minutes")
    @classmethod
    def _horizons(cls, v: list[float]) -> list[float]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("horizons_minutes must be positive")
        return sorted(set(float(h) for h in v))


class ModelConfig(_Strict):
    """A seeded ensemble of probabilistic classifiers."""

    backend: Literal["xgboost", "hist_gb", "random_forest", "logistic"] = "xgboost"
    ensemble_size: int = Field(5, ge=1)
    bootstrap: Literal["none", "block"] = "none"
    block_size: int = Field(360, ge=1)
    class_weight: Literal["balanced", "none"] = "balanced"
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
        }
    )


class CalibrationConfig(_Strict):
    """Post-hoc calibration fitted on the calibration split only."""

    method: Literal["temperature", "platt", "isotonic", "none"] = "temperature"
    compare_methods: list[Literal["temperature", "platt", "isotonic", "none"]] = Field(
        default_factory=lambda: ["none", "temperature", "platt", "isotonic"]
    )
    n_bins: int = Field(15, ge=2)


class AssetRule(_Strict):
    """Map feature names (regex) to a physical asset and its criticality."""

    pattern: str
    asset: str
    criticality: float = Field(ge=0, le=1)
    description: str = ""


class TrustConfig(_Strict):
    """Trust & risk layer settings."""

    anomaly_tail_quantile: float = Field(0.95, gt=0, lt=1)
    anomaly_z_threshold: float = Field(3.0, gt=0)
    top_k_features: int = Field(5, ge=1)
    horizon_weights: dict[str, float] = Field(default_factory=dict)
    default_horizon_decay: float = Field(0.01, ge=0, description="Per-minute risk discount.")
    consistency_weights: dict[str, float] = Field(
        default_factory=lambda: {"cross_source": 0.7, "temporal_coherence": 0.3}
    )
    epistemic_reference_quantile: float = Field(0.99, gt=0, lt=1)
    suspicious_recall_target: float = Field(0.95, gt=0, le=1)
    default_criticality: float = Field(0.5, ge=0, le=1)
    asset_rules: list[AssetRule] = Field(default_factory=list)


class AnalystProfile(_Strict):
    """Analyst context used by the adaptive assistance gate."""

    analyst_id: str = "default"
    expertise: Literal["junior", "intermediate", "senior"] = "intermediate"
    workload: float = Field(0.5, ge=0, le=1)
    shift: Literal["day", "night"] = "day"
    override_rate: float = Field(0.0, ge=0, le=1)


class GateConfig(_Strict):
    """Policy thresholds of the assistance gate (see ``cpscopilot.gate.policy``)."""

    abstain_epistemic_quantile: float = Field(0.99, gt=0, lt=1)
    abstain_max_epistemic: float | None = Field(None, ge=0)
    abstain_min_consistency: float = Field(0.35, ge=0, le=1)
    approval_min_risk: float = Field(0.6, ge=0, le=1)
    approval_min_criticality: float = Field(0.85, ge=0, le=1)
    recommend_min_risk: float = Field(0.3, ge=0, le=1)
    recommend_min_confidence: float = Field(0.6, ge=0, le=1)
    explain_min_risk: float = Field(0.1, ge=0, le=1)
    explain_benign_budget: float | None = Field(
        0.10, gt=0, lt=1,
        description="Max fraction of benign calibration rows allowed to reach Explain or above; "
                    "raises explain_min_risk to the matching risk quantile (None = off).")
    recommend_benign_budget: float | None = Field(
        0.02, gt=0, lt=1,
        description="Same alert budget for Recommend and above (None = off).")
    junior_requires_approval: bool = True
    junior_explain_factor: float = Field(0.75, gt=0)
    senior_confidence_offset: float = Field(-0.05, le=0)
    high_workload_threshold: float = Field(0.8, ge=0, le=1)
    high_workload_explain_factor: float = Field(1.5, ge=1)
    night_approval_factor: float = Field(0.9, gt=0, le=1)
    override_rate_threshold: float = Field(0.3, ge=0, le=1)
    override_confidence_offset: float = Field(0.1, ge=0)
    analyst_profiles: list[AnalystProfile] = Field(
        default_factory=lambda: [
            AnalystProfile(analyst_id="baseline"),
            AnalystProfile(analyst_id="junior_day", expertise="junior", workload=0.4),
            AnalystProfile(analyst_id="senior_overloaded", expertise="senior", workload=0.9),
            AnalystProfile(
                analyst_id="night_shift", expertise="intermediate", workload=0.6, shift="night"
            ),
        ]
    )

    @model_validator(mode="after")
    def _ordered(self) -> GateConfig:
        if not (self.explain_min_risk <= self.recommend_min_risk <= self.approval_min_risk):
            raise ValueError("require explain_min_risk <= recommend_min_risk <= approval_min_risk")
        if not self.analyst_profiles:
            raise ValueError("at least one analyst profile is required")
        return self


class LLMConfig(_Strict):
    """LLM backend. ``template`` is deterministic and offline (default for CI)."""

    backend: Literal["template", "anthropic"] = "template"
    model: str = "claude-opus-5-5"
    max_tokens: int = Field(8000, ge=256)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    use_refusal_fallback: bool = True
    timeout_seconds: float = Field(120.0, gt=0)
    max_retries: int = Field(2, ge=0)
    cache_dir: str | None = "llm_cache"
    fault_injection: dict[str, float] = Field(default_factory=dict)


class CopilotConfig(_Strict):
    """Retrieval-augmented copilot."""

    enabled: bool = True
    knowledge_base_dir: str = "knowledge_base"
    top_k: int = Field(4, ge=1)
    chunk_words: int = Field(140, ge=20)
    chunk_overlap_words: int = Field(30, ge=0)
    max_events: int = Field(60, ge=1)
    max_steps: int = Field(5, ge=1)
    support_threshold: float = Field(0.5, gt=0, le=1)
    feature_glossary: dict[str, str] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)


class EvaluationConfig(_Strict):
    """Metric computation."""

    n_bins: int = Field(15, ge=2)
    bootstrap_resamples: int = Field(500, ge=0)
    bootstrap_block_size: int | None = Field(None, ge=1)
    confidence_level: float = Field(0.95, gt=0, lt=1)
    curve_points: int = Field(400, ge=10)


class RuntimeConfig(_Strict):
    """Execution settings."""

    n_jobs: int = Field(4, ge=1)
    strict_leakage_checks: bool = True
    leakage_probes: int = Field(12, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_models: bool = False


class ExperimentConfig(_Strict):
    """Root configuration of one experiment."""

    name: str = "experiment"
    description: str = ""
    seed: int = 42
    output_root: str = "runs"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    detection: ModelConfig = Field(default_factory=ModelConfig)
    prediction: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    trust: TrustConfig = Field(default_factory=TrustConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    copilot: CopilotConfig = Field(default_factory=CopilotConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    project_root: str | None = Field(
        None, description="Filled in by load_config; relative paths resolve against it."
    )

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a config path relative to the project root (or CWD)."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        base = Path(self.project_root) if self.project_root else Path.cwd()
        return (base / p).resolve()


# ------------------------------------------------------------------- loading
_NON_SCIENTIFIC_KEYS = {"seed", "output_root", "project_root", "description"}


def find_project_root(start: Path) -> Path:
    """Walk upwards from ``start`` to the directory containing ``pyproject.toml``."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain: {' -> '.join(map(str, (*_seen, path)))}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _read_yaml_with_extends((path.parent / parent), (*_seen, path))
    return _deep_merge(base, data)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars/lists."""
    out = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override {key!r}: {part!r} is not a mapping")
        node[parts[-1]] = value
    return out


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load, merge and validate an experiment configuration.

    Args:
        path: YAML file (may use ``extends``). ``None`` yields pure defaults.
        overrides: dotted ``key=value`` strings applied after merging.
        project_root: base directory for relative paths; defaults to the nearest
            ancestor of the config file containing ``pyproject.toml``.
    """
    data: dict[str, Any] = {}
    root: Path
    if path is not None:
        cfg_path = Path(path)
        data = _read_yaml_with_extends(cfg_path)
        root = Path(project_root) if project_root else find_project_root(cfg_path.parent)
    else:
        root = Path(project_root) if project_root else find_project_root(Path.cwd())
    data = apply_overrides(data, overrides)
    data["project_root"] = str(Path(root).resolve())
    return ExperimentConfig.model_validate(data)


def _canonical(cfg: ExperimentConfig, exclude: set[str]) -> str:
    payload = {k: v for k, v in cfg.model_dump(mode="json").items() if k not in exclude}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: ExperimentConfig, include_seed: bool = False) -> str:
    """SHA-256 of the canonical scientific configuration."""
    exclude = set(_NON_SCIENTIFIC_KEYS)
    if include_seed:
        exclude.discard("seed")
    return hashlib.sha256(_canonical(cfg, exclude).encode("utf-8")).hexdigest()
