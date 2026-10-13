"""Dataset-format loaders (SWaT / WADI / TON_IoT / generic CSV) on miniature files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpscopilot.config import DatasetConfig
from cpscopilot.data.features import segment_ids
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


def test_ton_iot_flows_are_binned_at_their_observable_end_time(tmp_path: Path) -> None:
    t0 = 1556000000
    raw = pd.DataFrame({
        "ts": [t0 + 1, t0 + 4, t0 + 10, t0 + 2, t0 + 12, t0 + 25],
        "duration": [0.0, 2.0, 0.0, 25.0, 0.4, 0.5],
        "src_ip": ["a", "b", "a", "d", "c", "a"], "dst_port": [80, 80, 22, 443, 23, 80],
        "src_bytes": [100, 50, 10, 5000, 999, 1],
        "label": [0, 0, 0, 0, 1, 0],
        "type": ["normal", "normal", "normal", "normal", "scanning", "normal"],
    })
    raw.to_csv(tmp_path / "net.csv", index=False)
    cfg = DatasetConfig(name="ton_iot", paths=["net.csv"], timestamp_col="ts",
                        timestamp_unit="s", label_col="label", positive_labels=[1],
                        attack_type_col="type", feature_include=["src_bytes", "duration"],
                        count_distinct_cols=["src_ip", "dst_port"], resample_seconds=10,
                        resample_aggs=["sum", "max"])
    ds = load_dataset(cfg, resolver(tmp_path), 0)  # ton_iot defaults to duration shifting
    f = ds.frame.set_index(TIMESTAMP)
    first = pd.Timestamp(t0 + 10, unit="s")
    # the 25 s flow that *started* at t0+2 is only observable at t0+27: not in bin (0, 10]
    assert f.loc[first, "src_bytes__sum"] == 160
    assert f.loc[first, "src_ip__nunique"] == 2
    assert f.loc[first, "event_count"] == 3
    second = pd.Timestamp(t0 + 20, unit="s")
    assert f.loc[second, LABEL] == 1 and f.loc[second, "attack_type"] == "scanning"
    third = pd.Timestamp(t0 + 30, unit="s")
    assert f.loc[third, "src_bytes__sum"] == 5001 and f.loc[third, "duration__max"] == 25.0
    assert "src_ip" not in ds.feature_cols  # raw identifiers are never features


def test_resampling_drops_outages_and_starts_a_new_segment(tmp_path: Path) -> None:
    t0 = pd.Timestamp("2026-01-01")
    before = t0 + pd.to_timedelta(np.arange(0, 600, 3), "s")
    short_gap_end = before[-1] + pd.Timedelta(seconds=35)  # 3 empty 10 s bins: a dropout
    middle = short_gap_end + pd.to_timedelta(np.arange(0, 300, 3), "s")
    after = middle[-1] + pd.Timedelta(hours=2) + pd.to_timedelta(np.arange(0, 600, 3), "s")
    ts = before.append(middle).append(after)
    label = np.zeros(len(ts), dtype=int)
    label[len(before) - 20: len(before)] = 1  # attack running into the short dropout
    raw = pd.DataFrame({"ts": ts.astype(str), "x": np.arange(len(ts), dtype=float),
                        "label": label})
    raw.to_csv(tmp_path / "plant.csv", index=False)
    cfg = DatasetConfig(name="generic_csv", paths=["plant.csv"], timestamp_col="ts",
                        label_col="label", resample_seconds=10, max_gap_factor=60)
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    stats = ds.fingerprint["resampling"]
    assert stats["bins_dropped_outage"] > 700  # the 2-hour outage is not filled
    assert stats["bins_empty"] - stats["bins_dropped_outage"] == 3  # the dropout is kept
    segs = segment_ids(ds.frame[TIMESTAMP], ds.sampling_seconds, cfg.max_gap_factor)
    assert len(np.unique(segs)) == 2
    kept_empty = ds.frame[ds.frame["event_count"] == 0]
    assert len(kept_empty) == 3 and (kept_empty[LABEL] == 1).all()  # inherits attack state


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
    assert ds.fingerprint["attack_intervals"][0]["rows_matched"] == 3


def test_attack_intervals_respect_day_first_and_must_match(tmp_path: Path) -> None:
    ts = pd.date_range("2017-10-09 19:00", "2017-10-11 12:00", freq="10min")
    pd.DataFrame({"time": ts.strftime("%d/%m/%Y %H:%M:%S"), "x": np.arange(len(ts), dtype=float)}
                 ).to_csv(tmp_path / "plant.csv", index=False)
    pd.DataFrame({"start": ["9/10/2017 19:25:00", "10/10/2017 10:20:00", "11/10/2017 11:10:00"],
                  "end": ["9/10/2017 19:50:00", "10/10/2017 10:40:00", "11/10/2017 11:40:00"]}
                 ).to_csv(tmp_path / "attacks.csv", index=False)
    cfg = DatasetConfig(name="generic_csv", paths=["plant.csv"], timestamp_col="time",
                        timestamp_format="%d/%m/%Y %H:%M:%S", dayfirst=True,
                        attack_intervals_path="attacks.csv")
    ds = load_dataset(cfg, resolver(tmp_path), 0)  # all three day-first intervals match
    assert [r["rows_matched"] for r in ds.fingerprint["attack_intervals"]] == [3, 3, 4]
    with pytest.raises(ValueError, match="matches no rows"):
        load_dataset(cfg.model_copy(update={"dayfirst": False}), resolver(tmp_path), 0)
    pd.DataFrame({"start": ["10/10/2017 10:40:00"], "end": ["10/10/2017 10:20:00"]}).to_csv(
        tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="ends before it starts"):
        load_dataset(cfg.model_copy(update={"attack_intervals_path": "bad.csv"}),
                     resolver(tmp_path), 0)


def test_partially_missing_columns_are_not_screened_on_the_whole_file(tmp_path: Path) -> None:
    n = 100
    late_fail = np.r_[np.arange(45.0), np.full(n - 45, np.nan)]  # fails in calibration/test
    pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="10s").astype(str),
                  "x": np.arange(float(n)), "late": late_fail, "text": ["-"] * n,
                  "label": [0] * n}).to_csv(tmp_path / "plant.csv", index=False)
    cfg = DatasetConfig(name="generic_csv", paths=["plant.csv"], timestamp_col="time",
                        label_col="label")
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert "late" in ds.feature_cols  # usability is decided on the train split later
    assert "text" not in ds.feature_cols  # no numeric value at all


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


def test_iso_interval_tables_are_never_read_day_first(tmp_path: Path) -> None:
    ts = pd.date_range("2026-01-02 08:00", "2026-01-02 11:00", freq="10min").append(
        pd.date_range("2026-02-01 08:00", "2026-02-01 11:00", freq="10min"))
    pd.DataFrame({"time": ts.strftime("%d/%m/%Y %H:%M:%S"), "x": np.arange(len(ts), dtype=float)}
                 ).to_csv(tmp_path / "plant.csv", index=False)
    pd.DataFrame({"start": ["2026-02-01 08:00:00", "2026-01-02 10:00:00"],
                  "end": ["2026-02-01 08:20:00", "2026-01-02 10:30:00"],
                  "attack_type": ["a2", "a1"]}).to_csv(tmp_path / "attacks.csv", index=False)
    cfg = DatasetConfig(name="generic_csv", paths=["plant.csv"], timestamp_col="time",
                        dayfirst=True, attack_intervals_path="attacks.csv")
    f = load_dataset(cfg, resolver(tmp_path), 0).frame
    attacks = f[f[LABEL] == 1].groupby("attack_type")[TIMESTAMP]
    assert attacks.min().to_dict() == {"a1": pd.Timestamp("2026-01-02 10:00"),
                                       "a2": pd.Timestamp("2026-02-01 08:00")}


def _flows(tmp_path: Path, raw: pd.DataFrame, **extra) -> CPSDataset:
    raw.to_csv(tmp_path / "net.csv", index=False)
    cfg = DatasetConfig(name="ton_iot", paths=["net.csv"], timestamp_col="ts",
                        timestamp_unit="s", label_col="label", positive_labels=[1],
                        attack_type_col="type", resample_seconds=10,
                        resample_aggs=["sum", "mean"], max_gap_factor=60, **extra)
    return load_dataset(cfg, resolver(tmp_path), 0)


def test_flow_bins_without_traffic_are_not_attacks_unless_bracketed(tmp_path: Path) -> None:
    t0 = 1556000000
    benign = list(range(0, 100, 2))
    times = benign + [101, 103, 105] + [t + 600 for t in benign] + [701, 731] + [
        t + 800 for t in benign]
    label = [0] * 50 + [1, 1, 1] + [0] * 50 + [1, 1] + [0] * 50
    raw = pd.DataFrame({"ts": [t0 + t for t in times], "duration": 0.0,
                        "src_bytes": 1.0, "label": label,
                        "type": ["scanning" if v else "normal" for v in label]})
    ds = _flows(tmp_path, raw)
    assert ds.fingerprint["resampling"]["record_kind"] == "event"
    f = ds.frame.set_index(TIMESTAMP)
    silent = f["event_count"] == 0
    # a burst followed by silence: only the bin holding the burst is an attack
    assert f.loc[pd.Timestamp(t0 + 110, unit="s"), LABEL] == 1  # holds 101, 103, 105
    after_burst = f.loc[pd.Timestamp(t0 + 120, unit="s"): pd.Timestamp(t0 + 590, unit="s")]
    assert after_burst["event_count"].eq(0).all() and after_burst[LABEL].eq(0).all()
    assert set(after_burst["attack_type"]) == {"normal"}
    # silence *between* two attack bins (701 -> bin 710, 731 -> bin 740) stays an attack
    between = f.loc[pd.Timestamp(t0 + 710, unit="s"): pd.Timestamp(t0 + 740, unit="s")]
    assert between[LABEL].tolist() == [1, 1, 1, 1]
    assert set(between["attack_type"]) == {"scanning"}
    assert int(f.loc[silent, LABEL].sum()) == 2  # only the two bracketed silent bins
    assert (f.loc[silent, "src_bytes__sum"] == 0).all()  # no traffic: the sum is 0


def test_sums_are_missing_when_records_lack_the_field(tmp_path: Path) -> None:
    from cpscopilot.pipeline.runner import train_coverage

    t0 = 1556000000
    n = 400
    times = np.r_[np.arange(0, 2 * n, 4), np.arange(2 * n + 400, 4 * n + 400, 4)]
    body = np.where(times < 0.7 * times.max(), np.nan, 5.0)  # field absent early on
    raw = pd.DataFrame({"ts": t0 + times, "duration": 0.5, "src_bytes": 10.0,
                        "http_len": body, "label": 0, "type": "normal"})
    raw.loc[::50, "label"] = 1
    f = _flows(tmp_path, raw).frame
    rows = f[f["event_count"] > 0]
    early = rows[TIMESTAMP] < pd.Timestamp(t0 + 0.6 * times.max(), unit="s")
    assert rows.loc[early, "http_len__sum"].isna().all()  # not a fake 0
    assert (f.loc[f["event_count"] == 0, "http_len__sum"] == 0).all()
    train = f.iloc[: int(0.6 * len(f))]
    coverage = train_coverage(train, ["src_bytes__sum", "src_bytes__mean", "http_len__sum",
                                      "http_len__mean"], resampled=True)
    # silent bins do not count as missing; all aggregates of a field share its coverage
    assert coverage["src_bytes__mean"] == 1.0 == coverage["src_bytes__sum"]
    assert coverage["http_len__sum"] == coverage["http_len__mean"] == 0.0


def test_attack_intervals_match_flow_start_times(tmp_path: Path) -> None:
    t0 = pd.Timestamp("2026-01-01 10:00:00")
    starts = [t0 + pd.Timedelta(seconds=s) for s in range(0, 1800, 5)] + [
        t0 + pd.Timedelta(seconds=1203)]
    raw = pd.DataFrame({"ts": [str(t) for t in starts], "duration": [0.5] * 360 + [900.0],
                        "bytes": [10.0] * 360 + [1e7]})
    raw.to_csv(tmp_path / "net.csv", index=False)
    pd.DataFrame({"start": ["2026-01-01 10:20:01"], "end": ["2026-01-01 10:20:04"],
                  "attack_type": ["exfiltration"]}).to_csv(tmp_path / "attacks.csv", index=False)
    cfg = DatasetConfig(name="generic_csv", paths=["net.csv"], timestamp_col="ts",
                        event_duration_col="duration", attack_intervals_path="attacks.csv",
                        resample_seconds=10, resample_aggs=["sum"])
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert ds.fingerprint["attack_intervals"][0]["rows_matched"] == 1  # the long flow only
    f = ds.frame.set_index(TIMESTAMP)
    end_bin = pd.Timestamp("2026-01-01 10:35:10")  # 10:20:03 + 900 s, right-labelled
    assert f.loc[end_bin, "bytes"] >= 1e7 and f.loc[end_bin, LABEL] == 1
    assert f.loc[end_bin, "attack_type"] == "exfiltration"
    assert int(f[LABEL].sum()) == 1
