"""End-to-end runs of the six-stage pipeline on a tiny synthetic plant."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cpscopilot.config import load_config
from cpscopilot.data.synthetic import generate_synthetic_cps
from cpscopilot.pipeline.aggregate import aggregate_runs
from cpscopilot.pipeline.runner import ExperimentRunner
from cpscopilot.plotting.figures import FIGURES, generate_all
from cpscopilot.utils.io import read_json, sha256_file
from cpscopilot.utils.run_manager import verify_run
from tests.conftest import SMOKE, TINY_OVERRIDES

pytestmark = pytest.mark.slow

EXPECTED = ["config.resolved.yaml", "environment.json", "seed_state.json",
            "data_fingerprint.json", "split_audit.json", "leakage_audit.json", "fit_ledger.json",
            "trust_state.json", "gate_thresholds.json", "metrics.json", "reliability.json",
            "curves.json", "predictions_test.csv.gz", "copilot_outputs.jsonl",
            "copilot_audit.json", "knowledge_base.json", "raw_data_integrity.json", "run.log",
            "manifest.json"]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = load_config(SMOKE, TINY_OVERRIDES)
    root = tmp_path_factory.mktemp("runs")
    return ExperimentRunner(cfg, output_root=root).run().run_dir


def test_all_artifacts_exist_and_verify(run_dir: Path) -> None:
    for name in EXPECTED:
        assert (run_dir / name).exists(), name
    assert verify_run(run_dir)["ok"]


def test_leakage_audit_and_fit_ledger(run_dir: Path) -> None:
    audit = read_json(run_dir / "leakage_audit.json")
    assert audit["passed"], audit["failures"]
    names = {c["name"] for c in audit["checks"]}
    for split in ("train", "calibration", "test"):
        assert f"feature_causality[{split}]" in names
    ledger = read_json(run_dir / "fit_ledger.json")["entries"]
    assert not [e for e in ledger if e["split"] == "test"]
    assert {e["split"] for e in ledger if e["component"].startswith("calibrator:")} == \
        {"calibration"}
    assert {e["split"] for e in ledger if e["component"].startswith("model:")} == {"train"}


def test_metrics_cover_imbalance_and_calibration(run_dir: Path) -> None:
    m = read_json(run_dir / "metrics.json")
    for key in ("detection", "h5", "h15", "h30"):
        for method in ("none", "temperature", "platt", "isotonic"):
            entry = m["models"][key][method]
            for field in ("pr_auc", "balanced_accuracy", "f1_macro", "mcc", "prevalence"):
                assert field in entry["classification"]
            for field in ("ece", "mce", "brier", "nll", "adaptive_ece", "ece_top_label"):
                assert field in entry["calibration"]
        assert "bootstrap" in m["models"][key]["temperature"]
    assert m["models"]["h30"]["temperature"]["proactive_subset"]["classification"]["n"] > 0
    gate = m["operational"]["gate"]
    assert set(gate) == {"baseline", "junior_day", "senior_overloaded", "night_shift"}
    assert sum(gate["baseline"]["mode_counts"].values()) == m["operational"]["n_rows"]
    assert m["copilot"]["n_events"] > 0 and m["copilot"]["schema_valid_rate"] == 1.0


def test_predictions_belong_to_the_test_period(run_dir: Path) -> None:
    split = read_json(run_dir / "split_audit.json")
    preds = pd.read_csv(run_dir / "predictions_test.csv.gz", parse_dates=["timestamp"])
    assert preds["timestamp"].min() > pd.Timestamp(split["splits"]["calibration"]["end_ts"])
    assert preds["row_id"].min() >= split["splits"]["test"]["start"]


def test_same_seed_reproduces_metrics_and_predictions(run_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(SMOKE, TINY_OVERRIDES)
    again = ExperimentRunner(cfg, output_root=tmp_path).run().run_dir
    a, b = read_json(run_dir / "metrics.json"), read_json(again / "metrics.json")
    for volatile in ("run_id", "timings_seconds"):
        a.pop(volatile)
        b.pop(volatile)
    assert a == b
    assert sha256_file(run_dir / "predictions_test.csv.gz") == \
        sha256_file(again / "predictions_test.csv.gz")
    summary, table = aggregate_runs([run_dir, again])
    assert summary["config_hashes"] == [a["config_hash"]]
    row = table.set_index("metric").loc["models.detection.temperature.classification.pr_auc"]
    assert row["std"] == 0.0 and row["n"] == 2


def test_raw_csv_is_never_modified(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, TINY_OVERRIDES)
    frame, _ = generate_synthetic_cps(cfg.dataset.synthetic, 7)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "plant.csv"
    frame.drop(columns=["attack_type", "phase", "campaign_id"]).to_csv(raw, index=False)
    raw.chmod(0o444)
    before = sha256_file(raw)
    cfg = load_config(SMOKE, TINY_OVERRIDES + [
        "dataset.name=generic_csv", f"dataset.paths=[{raw}]", "dataset.timestamp_col=timestamp",
        "dataset.label_col=label", "copilot.enabled=false"])
    result = ExperimentRunner(cfg, output_root=tmp_path / "runs").run()
    integrity = read_json(result.run_dir / "raw_data_integrity.json")
    assert integrity["all_unchanged"] and sha256_file(raw) == before
    assert not (result.run_dir / "copilot_outputs.jsonl").exists()


def test_figures_are_generated_without_overwriting(run_dir: Path, tmp_path: Path) -> None:
    out = generate_all([run_dir], tmp_path / "figs", formats=["pdf", "svg"], latex="off")
    for name in FIGURES:
        assert (out / f"{name}.pdf").stat().st_size > 1000
        assert (out / f"{name}.svg").exists()
    manifest = json.loads((out / "figures_manifest.json").read_text())
    assert manifest["runs"][0]["manifest_sha256"] == sha256_file(run_dir / "manifest.json")
    assert (out / "captions.md").read_text().startswith("# Draft figure captions")
    with pytest.raises(FileExistsError):
        generate_all([run_dir], out, formats=["pdf"], latex="off")
