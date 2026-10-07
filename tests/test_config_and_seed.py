"""Configuration loading/hashing and global seeding."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from cpscopilot.config import apply_overrides, config_hash, load_config
from cpscopilot.utils.seed import derive_seed, make_rng, seed_everything


def test_experiment_configs_validate(project_root: Path) -> None:
    for path in sorted((project_root / "configs" / "experiments").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.name
        assert cfg.project_root == str(project_root)


def test_extends_deep_merges(project_root: Path) -> None:
    cfg = load_config(project_root / "configs" / "experiments" / "synthetic_smoke.yaml")
    base = load_config(project_root / "configs" / "default.yaml")
    assert cfg.detection.ensemble_size == 3
    assert cfg.labels.horizons_minutes == base.labels.horizons_minutes  # inherited
    assert cfg.trust.asset_rules == base.trust.asset_rules


def test_overrides_are_typed(project_root: Path) -> None:
    cfg = load_config(project_root / "configs" / "default.yaml",
                      ["detection.ensemble_size=7", "labels.horizons_minutes=[10, 20]"])
    assert cfg.detection.ensemble_size == 7
    assert cfg.labels.horizons_minutes == [10.0, 20.0]
    with pytest.raises(ValueError):
        apply_overrides({}, ["no-equals-sign"])


def test_unknown_keys_and_bad_values_fail(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"detection": {"ensembel_size": 3}}))
    with pytest.raises(ValidationError):
        load_config(bad)
    with pytest.raises(ValidationError):
        load_config(None, ["split.train_frac=0.9"])  # fractions no longer sum to 1
    with pytest.raises(ValidationError):
        load_config(None, ["features.windows=[1]"])


def test_config_hash_ignores_seed_but_not_science() -> None:
    a = load_config(None, ["seed=1"])
    b = load_config(None, ["seed=2"])
    c = load_config(None, ["seed=1", "calibration.method=platt"])
    assert config_hash(a) == config_hash(b)
    assert config_hash(a, include_seed=True) != config_hash(b, include_seed=True)
    assert config_hash(a) != config_hash(c)


def test_seed_everything_reproduces_global_streams() -> None:
    seed_everything(123)
    first = (random.random(), np.random.rand())
    seed_everything(123)
    assert (random.random(), np.random.rand()) == first


def test_derived_seeds_are_stable_and_distinct() -> None:
    assert derive_seed(42, "detection", 0) == derive_seed(42, "detection", 0)
    seeds = {derive_seed(42, "detection", m) for m in range(50)}
    assert len(seeds) == 50
    assert derive_seed(42, "a") != derive_seed(43, "a")
    assert make_rng(1, "x").random() == make_rng(1, "x").random()
