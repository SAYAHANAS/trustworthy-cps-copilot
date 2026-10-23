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


# ------------------------------------------------- formats of the real distributions
def _wadi_2019_file(path: Path, rows: list[int], labels: dict[int, int] | None = None,
                    origin: str = "2017-10-09 18:00:00") -> None:
    """Replica of WADI.A2 WADI_attackdataLABLE.csv: index row, padded header, M/D/YY
    dates, 'mm:ss.0' times without the hour, quoted label name, two trailing junk rows."""
    t0 = pd.Timestamp(origin)
    cols = ["1_AIT_001_PV", "2_FIT_001_PV", "2_LS_001_AL"]
    lines = [",".join(str(i) for i in range(len(cols) + 4)),
             "Row ,Date ,Time," + ",".join(cols) + ',"Attack LABLE (1:No Attack, -1:Attack)"']
    for r in rows:
        t = t0 + pd.Timedelta(seconds=r - 1)
        label = (labels or {}).get(r, 1)
        lines.append(f"{r},{t.month}/{t.day}/{t:%y},{t:%M:%S}.0,{r * 0.5},{r % 7},,{label}")
    lines += ["," * (len(cols) + 3) + "1"] * 2
    path.write_bytes(("\r\n".join(lines)).encode())


def test_wadi_2019_export_rebuilds_the_hour_from_row(tmp_path: Path) -> None:
    rows = [*range(1, 31), *range(21590, 21621)]  # 18:00 and the first midnight (Row 21601)
    _wadi_2019_file(tmp_path / "WADI_attackdataLABLE.csv", rows,
                    labels={r: -1 for r in range(21595, 21605)})
    cfg = DatasetConfig(name="wadi", paths=["WADI_attackdataLABLE.csv"])
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    ts = ds.frame[TIMESTAMP]
    assert ts.iloc[0] == pd.Timestamp("2017-10-09 18:00:00") and len(ds.frame) == len(rows)
    assert ts.iloc[-1] == pd.Timestamp("2017-10-10 00:00:19")
    assert int(ds.labels.sum()) == 10  # -1 marks attacks, the junk rows are gone
    assert ds.feature_cols == ["1_AIT_001_PV", "2_FIT_001_PV"]  # empty column dropped
    wadi = ds.fingerprint["read"]["per_file"]["WADI_attackdataLABLE.csv"]
    assert wadi["timestamp_source"] == "row_number" and wadi["date_changes_checked"] == 1
    assert wadi["rows_without_row_number_dropped"] == 2


def test_wadi_2019_export_rejects_rows_that_disagree_with_their_row_number(
        tmp_path: Path) -> None:
    _wadi_2019_file(tmp_path / "w.csv", [*range(1, 20), *range(21590, 21610)])
    data = (tmp_path / "w.csv").read_bytes()
    assert b"\r\n21600,10/9/17,59:59.0," in data
    (tmp_path / "w.csv").write_bytes(data.replace(b"\r\n21600,10/9/17,59:59.0,",
                                                  b"\r\n21600,10/9/17,59:58.0,"))
    cfg = DatasetConfig(name="wadi", paths=["w.csv"])
    with pytest.raises(ValueError, match="do not match the file at Row 21600"):
        load_dataset(cfg, resolver(tmp_path), 0)
    _wadi_2019_file(tmp_path / "late.csv", [*range(1, 20), *range(21590, 21610)],
                    origin="2017-10-09 17:00:00")  # a copy that does not start at 18:00
    with pytest.raises(ValueError, match="do not match"):  # the midnight date change shows it
        load_dataset(cfg.model_copy(update={"paths": ["late.csv"]}), resolver(tmp_path), 0)
    ok = load_dataset(cfg.model_copy(update={"paths": ["late.csv"],
                                             "row_time_origin": "2017-10-09 17:00:00"}),
                      resolver(tmp_path), 0)
    assert ok.frame[TIMESTAMP].iloc[0] == pd.Timestamp("2017-10-09 17:00:00")


def test_wadi_interval_file_replaces_the_built_in_labels(tmp_path: Path) -> None:
    _wadi_2019_file(tmp_path / "w.csv", list(range(1, 61)), labels={5: -1})
    pd.DataFrame({"start": ["2017-10-09 18:00:30"], "end": ["2017-10-09 18:00:39"]}).to_csv(
        tmp_path / "attacks.csv", index=False)
    cfg = DatasetConfig(name="wadi", paths=["w.csv"], attack_intervals_path="attacks.csv")
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert ds.labels.tolist() == [0] * 30 + [1] * 10 + [0] * 20
    assert not [c for c in ds.feature_cols if "LABLE" in c]  # never an input (label leak)
    assert ds.fingerprint["release_check"]["compared"] == ["rows"]


def _ton_iot_full_file(path: Path, ts0: int, n: int, uid: bool = False,
                       bad_bytes: bool = False) -> None:
    """Replica of Processed_Network_dataset/Network_dataset_N.csv (uid only in file 6)."""
    head = ["ts", *(["uid"] if uid else []), "src_ip", "src_port", "dst_ip", "dst_port",
            "proto", "service", "duration", "src_bytes", "dst_bytes", "conn_state",
            "http_referrer", "label", "type"]
    rows = []
    for i in range(n):
        attack = i % 10 == 0
        src_bytes = "0.0.0.0" if bad_bytes and i == 3 else str(100 + i)
        rows.append([str(ts0 + 2 * i), *([f"C{i}x"] if uid else []), f"192.168.1.{i % 5}",
                     str(40000 + i), "192.168.1.1", "80", "tcp", "-" if i % 2 else "http",
                     "0.5", src_bytes, "10", "SF", "-", "1" if attack else "0",
                     "scanning" if attack else "normal"])
    path.write_text("\n".join(",".join(r) for r in [head, *rows]) + "\n")


def test_ton_iot_full_files_are_globbed_in_numeric_order(tmp_path: Path) -> None:
    net = tmp_path / "network"
    net.mkdir()
    t0 = 1556000000
    _ton_iot_full_file(net / "Network_dataset_1.csv", t0, 50)
    _ton_iot_full_file(net / "Network_dataset_2.csv", t0 + 100, 50, uid=True, bad_bytes=True)
    _ton_iot_full_file(net / "Network_dataset_10.csv", t0 + 200, 50)
    cfg = DatasetConfig(name="ton_iot", paths=["network/Network_dataset_*.csv"],
                        timestamp_col="ts", timestamp_unit="s", label_col="label",
                        positive_labels=[1], attack_type_col="type",
                        feature_include=["duration", "src_bytes", "dst_bytes"],
                        count_distinct_cols=["src_ip", "dst_port", "service"],
                        resample_seconds=10, resample_aggs=["sum"])
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert [Path(f["path"]).name for f in ds.fingerprint["files"]] == [
        "Network_dataset_1.csv", "Network_dataset_2.csv", "Network_dataset_10.csv"]
    assert ds.fingerprint["non_numeric_values_coerced"] == {"src_bytes": 1}
    assert "uid" not in ds.frame.columns and "service__nunique" in ds.feature_cols
    assert ds.frame[TIMESTAMP].is_monotonic_increasing and ds.labels.sum() > 0
    with pytest.raises(FileNotFoundError, match="no raw data file matches"):
        load_dataset(cfg.model_copy(update={"paths": ["network/nothing_*.csv"]}),
                     resolver(tmp_path), 0)


def test_ton_iot_train_test_sample_is_rejected_with_an_explanation(tmp_path: Path) -> None:
    head = "src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,label,type"
    row = "192.168.1.37,4444,192.168.1.193,49178,tcp,-,290.371539,101568,1,backdoor"
    (tmp_path / "train_test_network.csv").write_bytes(
        b"\xef\xbb\xbf" + f"{head}\r\n{row}\r\n".encode())
    cfg = DatasetConfig(name="ton_iot", paths=["train_test_network.csv"], timestamp_col="ts",
                        timestamp_unit="s", label_col="label", resample_seconds=10,
                        feature_include=["duration", "src_bytes"])
    with pytest.raises(ValueError, match="Train_Test' network sample"):
        load_dataset(cfg, resolver(tmp_path), 0)
    with pytest.raises(ValueError, match="Train_Test' network sample"):  # without usecols too
        load_dataset(cfg.model_copy(update={"feature_include": None}), resolver(tmp_path), 0)


def _swat_rows(n: int, attack: range) -> list[list]:
    t0 = pd.Timestamp("2015-12-31 23:59:50")
    out = []
    for i in range(n):
        t = t0 + pd.Timedelta(seconds=i)
        stamp = f" {t.day}/{t.month}/{t.year} {t.hour % 12 or 12}:{t:%M:%S} {t:%p}"
        label = ("A ttack" if i == attack.start else "Attack") if i in attack else "Normal"
        out.append([stamp, 2.4 + i / 100, 500.0 + i, 2, label])
    return out


def test_swat_official_xlsx_is_read_through_a_cache(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Combined Data"
    ws.append([None] * 5)  # row 1 is blank in the official workbook
    ws.append([" Timestamp", "FIT101", "LIT101", " MV101", "Normal/Attack"])
    for r in _swat_rows(30, range(12, 18)):
        ws.append(r)
    (tmp_path / "SWaT").mkdir()
    wb.save(tmp_path / "SWaT" / "SWaT_Dataset_Attack_v0.xlsx")
    cfg = DatasetConfig(name="swat", paths=["SWaT/SWaT_Dataset_Attack_v0.xlsx"])
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert ds.feature_cols == ["FIT101", "LIT101", "MV101"]
    assert ds.labels.tolist() == [0] * 12 + [1] * 6 + [0] * 12  # 'A ttack' counts
    assert ds.frame[TIMESTAMP].iloc[10] == pd.Timestamp("2016-01-01 00:00:00")  # 1/1/2016
    excel = ds.fingerprint["read"]["excel"][0]
    assert Path(excel["cache"]).exists() and excel["sha256"] == sha256_file(
        tmp_path / "SWaT" / "SWaT_Dataset_Attack_v0.xlsx")
    again = load_dataset(cfg, resolver(tmp_path), 0)  # served from the cache
    pd.testing.assert_frame_equal(again.frame, ds.frame)
    assert ds.fingerprint["release_check"]["release"] is None  # not the official span


def test_swat_csv_copy_is_compared_with_the_official_release(tmp_path: Path) -> None:
    rows = _swat_rows(20, range(5, 9))
    frame = pd.DataFrame(rows, columns=[" Timestamp", "FIT101", "LIT101", " MV101",
                                        "Normal/Attack"])
    frame.to_csv(tmp_path / "merged.csv", index=False)
    from cpscopilot.data import loaders

    known = {"release": "test release", "rows": 21, "attack_rows": 4,
             "start": "2015-12-31 23:59:50", "end": "2016-01-01 00:00:09"}
    original = loaders._KNOWN_RELEASES["swat"]
    loaders._KNOWN_RELEASES["swat"] = [known]
    try:
        ds = load_dataset(DatasetConfig(name="swat", paths=["merged.csv"]), resolver(tmp_path), 0)
    finally:
        loaders._KNOWN_RELEASES["swat"] = original
    check = ds.fingerprint["release_check"]
    assert check["release"] == "test release" and not check["matches_release"]
    assert check["differences"] == ["rows: 20 vs 21 in the official file"]


def test_numeric_labels_match_whatever_their_text_form() -> None:
    from cpscopilot.data.loaders import _binary_labels

    assert _binary_labels(pd.Series([1.0, -1.0, 1.0]), [-1]).tolist() == [0, 1, 0]
    assert _binary_labels(pd.Series(["1", "-1"]), ["-1"]).tolist() == [0, 1]
    assert _binary_labels(pd.Series(["Normal", "A ttack"]), ["Attack"]).tolist() == [0, 1]


def test_compare_script_confirms_a_faithful_copy_and_reports_differences(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    openpyxl = pytest.importorskip("openpyxl")
    spec = importlib.util.spec_from_file_location(
        "compare_raw_files", Path(__file__).resolve().parents[1] / "scripts"
        / "compare_raw_files.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    rows = _swat_rows(40, range(10, 20))
    header = [" Timestamp", "FIT101", "LIT101", " MV101", "Normal/Attack"]
    wb = openpyxl.Workbook()
    wb.active.append([None] * 5)
    wb.active.append(header)
    for r in rows:
        wb.active.append(r)
    wb.save(tmp_path / "official.xlsx")
    copy = pd.DataFrame(rows, columns=header)
    copy[" MV101"] = copy[" MV101"].astype(float)  # '2.0' instead of '2', as in merged.csv
    pd.concat([copy[copy["Normal/Attack"] == "Normal"], copy[copy["Normal/Attack"] != "Normal"]]
              ).to_csv(tmp_path / "merged.csv", index=False)  # normal rows first
    swat = (Path(__file__).resolve().parents[1] / "configs/experiments/swat.yaml").as_posix()
    config = tmp_path / "swat.yaml"  # project root = tmp_path: the Excel cache goes there
    config.write_text(f"extends: {swat}\n")
    monkeypatch.chdir(tmp_path)
    config = str(config)
    assert compare.main(["--config", config, str(tmp_path / "official.xlsx"),
                         str(tmp_path / "merged.csv")]) == 0
    assert "IDENTICAL: 40 rows" in capsys.readouterr().out
    copy.loc[3, "LIT101"] += 1.0
    copy.drop(index=7).to_csv(tmp_path / "edited.csv", index=False)
    assert compare.main(["--config", config, str(tmp_path / "official.xlsx"),
                         str(tmp_path / "edited.csv")]) == 1
    out = capsys.readouterr().out
    assert "only in the first file: 1" in out and "column LIT101: 1 rows differ" in out


def test_wadi_14_day_and_attack_files_load_together(tmp_path: Path) -> None:
    _wadi_2019_file(tmp_path / "attack.csv", list(range(1, 41)))
    days = tmp_path / "WADI_14days_new.csv"  # starts 25 Sep, header on line 1, no labels
    t0 = pd.Timestamp("2017-09-25 18:00:00")
    lines = ["Row,Date,Time,1_AIT_001_PV,2_FIT_001_PV,2_LS_001_AL"]
    for r in [*range(1, 21), *range(600_000, 600_010)]:  # the 'unstable period' gap
        t = t0 + pd.Timedelta(seconds=r - 1)
        year = f"{t:%Y}" if t.month == 9 else f"{t:%y}"  # mixed 4- and 2-digit years
        lines.append(f"{r},{t.month}/{t.day}/{year},{t:%M:%S}.0,1.#QNAN,{r % 3},")
    days.write_text("\n".join(lines))
    pd.DataFrame({"start": ["2017-10-09 18:00:10"], "end": ["2017-10-09 18:00:14"]}).to_csv(
        tmp_path / "attacks.csv", index=False)
    cfg = DatasetConfig(name="wadi", paths=["WADI_14days_new.csv", "attack.csv"],
                        attack_intervals_path="attacks.csv")
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    ts = ds.frame[TIMESTAMP]
    assert ts.iloc[0] == pd.Timestamp("2017-09-25 18:00:00")
    assert pd.Timestamp("2017-10-02 16:39:59") in set(ts)  # Row 600000
    assert ts.iloc[-1] == pd.Timestamp("2017-10-09 18:00:39") and int(ds.labels.sum()) == 5


def test_labels_with_an_empty_cell_keep_the_other_attacks() -> None:
    from cpscopilot.data.loaders import _binary_labels

    assert _binary_labels(pd.Series([1.0, -1.0, np.nan]), [-1]).tolist() == [0, 1, 0]
    assert _binary_labels(pd.Series([0, 1, None]), [1]).tolist() == [0, 1, 0]


def test_ton_iot_text_columns_are_stack_independent(tmp_path: Path) -> None:
    net = tmp_path / "n"
    net.mkdir()
    t0 = 1556000000
    _ton_iot_full_file(net / "Network_dataset_1.csv", t0, 30, bad_bytes=True)
    blank = pd.read_csv(net / "Network_dataset_1.csv", dtype=str)
    blank["ts"] = (blank["ts"].astype(int) + 100).astype(str)
    blank["service"] = ""  # a file where a count-distinct column is entirely blank
    blank.loc[[2, 5], "src_ip"] = ""
    blank.to_csv(net / "Network_dataset_2.csv", index=False)
    common = dict(name="ton_iot", timestamp_col="ts", timestamp_unit="s", label_col="label",
                  positive_labels=[1], attack_type_col="type",
                  count_distinct_cols=["src_ip", "service"], resample_seconds=10,
                  resample_aggs=["sum"])
    fast = DatasetConfig(paths=["n/Network_dataset_*.csv"],
                         feature_include=["duration", "src_bytes"], **common)
    slow = DatasetConfig(paths=["n/Network_dataset_*.csv"], feature_include=["re:^duration$",
                                                                             "re:^src_bytes$"],
                         **common)
    plain = slow.model_copy(update={"read_csv_kwargs": {"dtype": {}}})  # text as objects
    a, b, c = (load_dataset(x, resolver(tmp_path), 0) for x in (fast, slow, plain))
    pd.testing.assert_frame_equal(a.frame, b.frame, check_dtype=False)
    pd.testing.assert_frame_equal(a.frame, c.frame, check_dtype=False)
    assert a.fingerprint["non_numeric_values_coerced"] == {"src_bytes": 2}
    f = a.frame.set_index(TIMESTAMP)
    # blank cells are not a distinct host or service
    assert f["service__nunique"].loc[pd.Timestamp(t0 + 110, unit="s"):].max() <= 0


def test_ton_iot_duration_default_applies_with_an_explicit_feature_list(tmp_path: Path) -> None:
    _ton_iot_full_file(tmp_path / "Network_dataset_1.csv", 1556000000, 30)
    df = pd.read_csv(tmp_path / "Network_dataset_1.csv")
    df["duration"] = 25.0
    df.to_csv(tmp_path / "Network_dataset_1.csv", index=False)
    cfg = DatasetConfig(name="ton_iot", paths=["Network_dataset_1.csv"], timestamp_col="ts",
                        timestamp_unit="s", label_col="label", positive_labels=[1],
                        feature_include=["src_bytes"], resample_seconds=10)
    ds = load_dataset(cfg, resolver(tmp_path), 0)  # event_duration_col defaults to duration
    assert ds.frame[TIMESTAMP].iloc[0] == pd.Timestamp(1556000000 + 30, unit="s")


def test_identical_files_matched_twice_are_rejected(tmp_path: Path) -> None:
    _ton_iot_full_file(tmp_path / "Network_dataset_1.csv", 1556000000, 20)
    (tmp_path / "Network_dataset_1 (1).csv").write_bytes(
        (tmp_path / "Network_dataset_1.csv").read_bytes())
    cfg = DatasetConfig(name="ton_iot", paths=["Network_dataset_*.csv"], timestamp_col="ts",
                        timestamp_unit="s", label_col="label", resample_seconds=10)
    with pytest.raises(ValueError, match="identical files would be counted twice"):
        load_dataset(cfg, resolver(tmp_path), 0)


def test_explicit_timestamp_format_wins_over_iso_guessing(tmp_path: Path) -> None:
    pd.DataFrame({"t": ["2017-01-02 00:00:00", "2017-01-02 00:00:10"], "x": [1.0, 2.0],
                  "label": [0, 1]}).to_csv(tmp_path / "f.csv", index=False)
    cfg = DatasetConfig(name="generic_csv", paths=["f.csv"], timestamp_col="t",
                        timestamp_format="%Y-%d-%m %H:%M:%S", label_col="label")
    ds = load_dataset(cfg, resolver(tmp_path), 0)
    assert ds.frame[TIMESTAMP].iloc[0] == pd.Timestamp("2017-02-01")


def test_swat_copies_with_separators_padded_duplicates_and_index(tmp_path: Path) -> None:
    rows = _swat_rows(30, range(10, 15))
    frame = pd.DataFrame(rows, columns=[" Timestamp", "FIT101", "LIT101", " MV101",
                                        "Normal/Attack"])
    frame.to_csv(tmp_path / "semi.csv", sep=";", decimal=",", index=False)
    semi = DatasetConfig(name="swat", paths=["semi.csv"],
                         read_csv_kwargs={"sep": ";", "decimal": ","})
    assert load_dataset(semi, resolver(tmp_path), 0).frame["LIT101"].iloc[0] == 500.0
    with pytest.raises(ValueError, match="';' separators"):
        load_dataset(semi.model_copy(update={"read_csv_kwargs": {}}), resolver(tmp_path), 0)
    # normal + attack files concatenated without stripping names, saved with the index
    halves = frame.copy()
    halves["MV101"] = np.nan
    halves.loc[20:, "MV101"] = halves.loc[20:, " MV101"]
    halves.loc[20:, " MV101"] = np.nan
    halves.to_csv(tmp_path / "merged.csv")  # index=True: an 'Unnamed: 0' column
    ds = load_dataset(DatasetConfig(name="swat", paths=["merged.csv"]), resolver(tmp_path), 0)
    assert ds.feature_cols == ["FIT101", "LIT101", "MV101"]  # merged, index not a feature
    assert ds.frame["MV101"].notna().all()


def test_excel_cache_is_specific_to_the_header_rule(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.append([None, "P1", None, None, None])  # stage banner (Normal_v1 layout)
    wb.active.append([" Timestamp", "FIT101", "LIT101", " MV101", "Normal/Attack"])
    for r in _swat_rows(20, range(5, 8)):
        wb.active.append(r)
    wb.save(tmp_path / "normal.xlsx")
    generic = DatasetConfig(name="generic_csv", paths=["normal.xlsx"], timestamp_col="Timestamp",
                            timestamp_format="%d/%m/%Y %I:%M:%S %p", label_col="Normal/Attack",
                            positive_labels=["Attack"])
    assert load_dataset(generic, resolver(tmp_path), 0).feature_cols == [
        "FIT101", "LIT101", "MV101"]  # the banner row is not taken as the header
    swat = load_dataset(DatasetConfig(name="swat", paths=["normal.xlsx"]), resolver(tmp_path), 0)
    assert int(swat.labels.sum()) == 3
    assert len(list((tmp_path / "data/interim/excel_cache").glob("normal.*.csv"))) == 2
