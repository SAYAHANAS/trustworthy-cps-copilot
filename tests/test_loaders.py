"""Dataset-format loaders (SWaT / WADI / TON_IoT / generic CSV) on miniature files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpscopilot.config import DatasetConfig
from cpscopilot.data.loaders import load_dataset
from cpscopilot.data.schema import LABEL, TIMESTAMP, CPSDataset
from cpscopilot.utils.io import sha256_file


def resolver(base: Path):
    return lambda p: (base / p).resolve()


def test_swat_format(tmp_path: Path) -> None:
    ts = pd.date_range("2015-12-28 10:00:00", periods=6, freq="1s")
    raw = pd.DataFrame({
        " Timestamp": [f" {t:%d/%m/%Y %I:%M:%S %p}" for t in ts],
        "FIT101": np.arange(6.0), " MV101": [1, 1, 2, 2, 1, 1], "LIT101": np.arange(6.0) * 2,
        "Normal/Attack": ["Normal", "Normal", "Attack", "A ttack", "Normal", "Normal"],
    })
    path = tmp_path / "swat.csv"
    raw.to_csv(path, index=False)
    digest = sha256_file(path)
    ds = load_dataset(DatasetConfig(name="swat", paths=["swat.csv"]), resolver(tmp_path), 0)
    assert ds.feature_cols == ["FIT101", "MV101", "LIT101"]
    assert ds.labels.tolist() == [0, 0, 1, 1, 0, 0]
    assert ds.frame[TIMESTAMP].iloc[0] == pd.Timestamp("2015-12-28 10:00:00")
    assert ds.sampling_seconds == 1.0
    assert ds.fingerprint["files"][0]["sha256"] == digest
    assert sha256_file(path) == digest  # raw file untouched


def test_wadi_format_with_long_tags_and_signed_labels(tmp_path: Path) -> None:
    prefix = r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA" + "\\"
    lines = ["WADI export header line",
             f"Row,Date,Time,{prefix}1_AIT_001_PV,{prefix}2_FIT_001_PV,"
             "Attack LABLE (1:No Attack -1:Attack)"]
    for i in range(5):
        lines.append(f"{i},10/9/2017,6:00:0{i}.000 PM,{i * 1.5},{i * 2.0},{-1 if i == 3 else 1}")
    path = tmp_path / "wadi.csv"
    path.write_text("\n".join(lines) + "\n")
    cfg = DatasetConfig(name="wadi", paths=["wadi.csv"], read_csv_kwargs={"skiprows": 1})
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert ds.feature_cols == ["1_AIT_001_PV", "2_FIT_001_PV"]
    assert ds.labels.tolist() == [0, 0, 0, 1, 0]


def test_ton_iot_events_are_binned_causally(tmp_path: Path) -> None:
    raw = pd.DataFrame({
        "ts": [1556000001, 1556000004, 1556000010, 1556000012, 1556000025],
        "src_ip": ["a", "b", "a", "c", "a"], "dst_port": [80, 80, 22, 23, 80],
        "src_bytes": [100, 50, 10, 999, 1], "duration": [0.1, 0.2, 0.3, 0.4, 0.5],
        "label": [0, 0, 0, 1, 0], "type": ["normal", "normal", "normal", "scanning", "normal"],
    })
    raw.to_csv(tmp_path / "net.csv", index=False)
    cfg = DatasetConfig(name="ton_iot", paths=["net.csv"], timestamp_col="ts",
                        timestamp_unit="s", label_col="label", positive_labels=[1],
                        attack_type_col="type", feature_include=["src_bytes", "duration"],
                        count_distinct_cols=["src_ip", "dst_port"], resample_seconds=10,
                        resample_aggs=["sum", "max"])
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    f = ds.frame.set_index(TIMESTAMP)
    first = pd.Timestamp(1556000010, unit="s")
    assert f.loc[first, "src_bytes__sum"] == 160  # events at 1, 4 and 10 s -> bin (0, 10]
    assert f.loc[first, "src_ip__nunique"] == 2
    assert f.loc[first, "event_count"] == 3
    second = pd.Timestamp(1556000020, unit="s")
    assert f.loc[second, LABEL] == 1 and f.loc[second, "attack_type"] == "scanning"
    assert "src_ip" not in ds.feature_cols  # raw identifiers are never features


def test_generic_csv_and_attack_intervals(tmp_path: Path) -> None:
    ts = pd.date_range("2026-01-01", periods=10, freq="10s")
    pd.DataFrame({"time": ts.astype(str), "x": np.arange(10.0), "y": np.ones(10)}).to_csv(
        tmp_path / "plant.csv", index=False)
    pd.DataFrame({"start": [str(ts[3])], "end": [str(ts[5])], "attack_type": ["spoof"]}).to_csv(
        tmp_path / "attacks.csv", index=False)
    cfg = DatasetConfig(name="generic_csv", paths=["plant.csv"], timestamp_col="time",
                        attack_intervals_path="attacks.csv")
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert ds.labels.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert set(ds.frame.loc[ds.frame[LABEL] == 1, "attack_type"]) == {"spoof"}


def test_synthetic_is_seed_deterministic() -> None:
    cfg = DatasetConfig(name="synthetic")
    a = load_dataset(cfg, Path, 3)
    b = load_dataset(cfg, Path, 3)
    c = load_dataset(cfg, Path, 4)
    assert a.fingerprint["frame_sha256"] == b.fingerprint["frame_sha256"]
    assert a.fingerprint["frame_sha256"] != c.fingerprint["frame_sha256"]
    assert {"label", "attack_type", "phase", "campaign_id"}.isdisjoint(a.feature_cols)


def test_label_columns_can_never_be_features() -> None:
    frame = pd.DataFrame({TIMESTAMP: pd.date_range("2026-01-01", periods=3, freq="1s"),
                          "x": [1.0, 2.0, 3.0], LABEL: [0, 1, 0]})
    with pytest.raises(ValueError):
        CPSDataset(frame=frame, feature_cols=["x", LABEL], sampling_seconds=1.0, source="t")
