"""End-to-end runs of the six-stage pipeline on a tiny synthetic plant."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cpscopilot.config import load_config
from cpscopilot.copilot.llm import TemplateLLM
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
    assert m["trust"]["epistemic_reference_detection"] > 0
    unc = m["operational"]["uncertainty"]
    assert set(unc["error_detection_auroc"]) == {
        "epistemic", "aleatoric", "one_minus_trust_adjusted_confidence", "inconsistency"}
    assert "epistemic_max_over_models" in unc["error_detection_auroc_gate_signals"]


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
    with pytest.raises(ValueError, match="seeds repeat"):
        aggregate_runs([run_dir, again])  # replicas are not independent samples
    with pytest.raises(ValueError, match="more than once"):
        aggregate_runs([run_dir, run_dir])
    summary, table = aggregate_runs([run_dir, again], allow_duplicates=True)
    assert summary["config_hashes"] == [a["config_hash"]]
    row = table.set_index("metric").loc["models.detection.temperature.classification.pr_auc"]
    assert row["std"] == 0.0 and row["n"] == 2
    with pytest.raises(ValueError, match="more than once"):  # never, even when allowed
        aggregate_runs([run_dir, again, again], allow_duplicates=True)


def test_different_configurations_may_share_seeds(run_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(SMOKE, TINY_OVERRIDES + ["labels.future_mode=onset",
                                               "copilot.enabled=false"])
    other = ExperimentRunner(cfg, output_root=tmp_path).run().run_dir
    with pytest.raises(ValueError, match="different configurations"):
        aggregate_runs([run_dir, other])
    summary, _ = aggregate_runs([run_dir, other], allow_mixed=True)  # same seed, other config
    assert len(summary["config_hashes"]) == 2


def test_raw_csv_is_never_modified(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, TINY_OVERRIDES)
    frame, _ = generate_synthetic_cps(cfg.dataset.synthetic, 7)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "plant.csv"
    n = len(frame)
    frame = frame.drop(columns=["attack_type", "phase", "campaign_id"])
    # available in train only (kept), and missing throughout train (dropped on train data)
    frame["LATE_FAIL"] = np.where(np.arange(n) < 0.45 * n, 1.0, np.nan)
    frame["EARLY_FAIL"] = np.where(np.arange(n) < 0.7 * n, np.nan, 1.0)
    frame.to_csv(raw, index=False)
    raw.chmod(0o444)
    before = sha256_file(raw)
    cfg = load_config(SMOKE, TINY_OVERRIDES + [
        "dataset.name=generic_csv", f"dataset.paths=[{raw}]", "dataset.timestamp_col=timestamp",
        "dataset.label_col=label", "copilot.enabled=false"])
    result = ExperimentRunner(cfg, output_root=tmp_path / "runs").run()
    integrity = read_json(result.run_dir / "raw_data_integrity.json")
    assert integrity["all_unchanged"] and sha256_file(raw) == before
    assert not (result.run_dir / "copilot_outputs.jsonl").exists()
    assert result.metrics["preparation"]["dropped_low_train_coverage"] == ["EARLY_FAIL"]


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
    with pytest.raises(ValueError, match="more than once"):
        generate_all([run_dir, run_dir], tmp_path / "dup", formats=["pdf"], latex="off")
    assert not (tmp_path / "dup").exists() and not (tmp_path / "dup.partial").exists()


@pytest.mark.parametrize("target", ["cwd", "existing_empty_dir"])
def test_figures_can_fill_an_existing_empty_directory(run_dir: Path, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      target: str) -> None:
    out = tmp_path / "out"
    out.mkdir()
    if target == "cwd":
        monkeypatch.chdir(out)
    written = generate_all([run_dir], Path(".") if target == "cwd" else out,
                           formats=["pdf"], latex="off", only=["fig_reliability_diagrams"])
    assert written == out.resolve()
    assert sorted(p.name for p in out.iterdir()) == [
        "captions.md", "fig_reliability_diagrams.pdf", "figures_manifest.json"]


class TemplateAnthropic:
    """Deterministic stand-in for the API client: answers with the template output."""

    def __init__(self) -> None:
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        user = kwargs["messages"][0]["content"]
        packet = json.loads(user.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
        text = json.dumps(TemplateLLM().build(packet))
        return SimpleNamespace(stop_reason="end_turn", model=kwargs["model"],
                               content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def test_warm_llm_cache_does_not_change_metrics(tmp_path: Path) -> None:
    pytest.importorskip("anthropic")
    cfg = load_config(SMOKE, TINY_OVERRIDES + [
        "copilot.llm.backend=anthropic", f"copilot.llm.cache_dir={tmp_path / 'cache'}"])
    runs, clients = [], []
    for _ in range(2):
        clients.append(TemplateAnthropic())
        runs.append(ExperimentRunner(cfg, output_root=tmp_path / "runs",
                                     llm_client=clients[-1]).run().run_dir)
    assert clients[0].calls > 0 and clients[1].calls == 0  # the second run is all cache hits
    stats = [read_json(d / "llm_cache_stats.json") for d in runs]
    assert stats[1] == {"hits": clients[0].calls, "misses": 0}
    a, b = (read_json(d / "metrics.json") for d in runs)
    for volatile in ("run_id", "timings_seconds"):
        a.pop(volatile)
        b.pop(volatile)
    assert a == b and "llm_cache" not in a["copilot"]


def test_onset_mode_keeps_ongoing_attacks_and_figures_need_no_baseline_method(
        tmp_path: Path) -> None:
    cfg = load_config(SMOKE, TINY_OVERRIDES + ["labels.future_mode=onset",
                                               "calibration.compare_methods=[temperature]"])
    result = ExperimentRunner(cfg, output_root=tmp_path / "runs").run()
    op = result.metrics["operational"]
    ongoing = sum(op["gate"]["baseline"]["crosstab"]["ongoing"].values())
    preds = pd.read_csv(result.run_dir / "predictions_test.csv.gz")
    assert ongoing == int(preds["y_now"].sum()) > 0  # every attack-in-progress row is counted
    assert op["gate"]["baseline"]["ongoing_coverage"] is not None
    assert "none" in result.metrics["models"]["detection"]  # uncalibrated reference kept
    out = generate_all([result.run_dir], tmp_path / "figs", formats=["pdf"], latex="off")
    assert (out / "fig_reliability_diagrams.pdf").exists()
    assert (out / "fig_horizon_performance.pdf").exists()
