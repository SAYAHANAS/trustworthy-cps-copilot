"""Historical feature vectors must never contain future observations.

These tests attack the featurizer and the per-split preparation from several
angles: perturbing or deleting the future, injecting spikes, recomputing
trailing windows by hand, splitting at data gaps and perturbing *other* splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpscopilot.config import FeatureConfig, SplitConfig
from cpscopilot.data.features import CausalWindowFeaturizer, segment_ids
from cpscopilot.data.leakage import audit_feature_causality
from cpscopilot.data.loaders import resample_causal
from cpscopilot.data.schema import LABEL, TIMESTAMP
from cpscopilot.data.splitting import TemporalSplitter
from cpscopilot.pipeline.runner import prepare_split
from tests.conftest import make_series_frame

COLS = ["S0", "S1", "S2"]
CFG = FeatureConfig(windows=[3, 12], stats=["mean", "std", "min", "max", "range", "slope"],
                    diff_lags=[1, 4])


@pytest.fixture()
def fz() -> CausalWindowFeaturizer:
    return CausalWindowFeaturizer(CFG)


@pytest.mark.parametrize("t0", [0, 5, 11, 12, 150, 377, 598])
def test_future_perturbation_leaves_past_features_unchanged(series_frame, fz, t0) -> None:
    full, _ = fz.transform(series_frame, COLS)
    perturbed = series_frame.copy()
    future = perturbed.index[t0 + 1:]
    rng = np.random.default_rng(t0)
    perturbed.loc[future, COLS] = rng.normal(0, 1e9, size=(len(future), len(COLS)))
    perturbed.loc[future[::3], COLS] = np.nan
    pert, _ = fz.transform(perturbed, COLS)
    np.testing.assert_array_equal(full.iloc[: t0 + 1].to_numpy(), pert.iloc[: t0 + 1].to_numpy())


@pytest.mark.parametrize("t0", [13, 64, 300, 599])
def test_truncated_history_gives_identical_features(series_frame, fz, t0) -> None:
    full, _ = fz.transform(series_frame, COLS)
    truncated, _ = fz.transform(series_frame.iloc[: t0 + 1], COLS)
    np.testing.assert_array_equal(full.iloc[: t0 + 1].to_numpy(), truncated.to_numpy())


def test_spike_affects_only_its_own_and_later_rows(series_frame, fz) -> None:
    k = 250
    base, _ = fz.transform(series_frame, COLS)
    spiked_frame = series_frame.copy()
    # ~100x the series scale; much larger spikes leave visible round-off in pandas'
    # online rolling variance (sum of squares), which is history- not future-dependent.
    spiked_frame.loc[k, "S0"] = 1e3
    spiked, _ = fz.transform(spiked_frame, COLS)
    a, b = base.to_numpy(), spiked.to_numpy()
    same = (a == b) | (np.isnan(a) & np.isnan(b))  # warm-up NaNs compare equal
    diff = ~same.all(axis=1)
    assert not diff[:k].any(), "a future spike changed a historical feature vector"
    assert diff[k]
    # the spike leaves every trailing window once it is older than the widest window
    # (up to float round-off of the online rolling sums)
    window_cols = [c for c in base.columns if "_w" in c]
    late = slice(k + max(CFG.windows) + max(CFG.diff_lags), None)
    np.testing.assert_allclose(base[window_cols].iloc[late].to_numpy(),
                               spiked[window_cols].iloc[late].to_numpy(), rtol=1e-5, atol=1e-5)


def test_windows_are_trailing_not_centered(fz) -> None:
    frame = make_series_frame(n=80)
    frame[COLS] = frame[COLS].ffill().fillna(0.0)
    feats, _ = fz.transform(frame, COLS)
    x = frame["S1"].to_numpy()
    for t in (11, 40, 79):
        window = x[t - 11: t + 1]
        assert feats.loc[t, "S1__mean_w12"] == pytest.approx(window.mean(), rel=1e-5)
        assert feats.loc[t, "S1__max_w12"] == pytest.approx(window.max(), rel=1e-5)
        assert feats.loc[t, "S1__std_w12"] == pytest.approx(window.std(), rel=1e-4, abs=1e-5)
        slope = np.polyfit(np.arange(12), window, 1)[0]
        assert feats.loc[t, "S1__slope_w12"] == pytest.approx(slope, rel=1e-3, abs=1e-4)
        assert feats.loc[t, "S1__diff4"] == pytest.approx(x[t] - x[t - 4], rel=1e-5)


def test_warmup_rows_are_flagged_invalid(series_frame, fz) -> None:
    _, valid = fz.transform(series_frame, COLS)
    assert fz.warmup_steps == 11
    assert not valid[: fz.warmup_steps].any()
    assert valid[fz.warmup_steps:].all()


def test_windows_never_span_a_data_gap(fz) -> None:
    frame = make_series_frame(n=200)
    frame.loc[100:, TIMESTAMP] = frame.loc[100:, TIMESTAMP] + pd.Timedelta(hours=2)
    segs = segment_ids(frame[TIMESTAMP], 10.0, 5.0)
    assert segs[99] == 0 and segs[100] == 1
    joint, valid = fz.transform(frame, COLS, segs)
    alone, _ = fz.transform(frame.iloc[100:], COLS)
    np.testing.assert_array_equal(joint.iloc[100:].to_numpy(), alone.to_numpy())
    assert not valid[100: 100 + fz.warmup_steps].any()


def _prepared(frame: pd.DataFrame):
    split = TemporalSplitter(SplitConfig(), 15).split(frame)
    return split, {
        name: prepare_split(name, split[name], COLS, CausalWindowFeaturizer(CFG),
                            {"h5": 6}, "any", 10.0, 5.0, [])
        for name in ("train", "calibration", "test")
    }


def test_splits_are_featurised_in_isolation() -> None:
    frame = make_series_frame(n=1500, attack_every=60)
    split, prepared = _prepared(frame)
    tampered = frame.copy()
    tampered.loc[: split["calibration"].index.min() - 1, COLS] = 1e9  # train + embargo gap
    _, prepared_t = _prepared(tampered)
    for name in ("calibration", "test"):
        np.testing.assert_array_equal(prepared[name].X.to_numpy(), prepared_t[name].X.to_numpy())
    # and the earliest usable calibration row has a full window of calibration rows only
    cal = prepared["calibration"]
    assert cal.dropped_warmup == CausalWindowFeaturizer(CFG).warmup_steps


def test_prepared_rows_contain_no_future_observations() -> None:
    frame = make_series_frame(n=900, attack_every=60)
    split, prepared = _prepared(frame)
    rng = np.random.default_rng(0)
    for name, part in prepared.items():
        seg = split[name]
        for row_id in rng.choice(part.row_ids, size=5, replace=False):
            pos = seg.index.get_loc(row_id)
            history_only, _ = CausalWindowFeaturizer(CFG).transform(seg.iloc[: pos + 1], COLS)
            np.testing.assert_array_equal(part.X.loc[row_id].to_numpy(),
                                          history_only.iloc[-1].to_numpy())


def test_runtime_causality_audit_passes(series_frame, fz) -> None:
    report = audit_feature_causality(fz, series_frame, COLS, 8, np.random.default_rng(1))
    assert report["passed"]
    assert report["max_truncation_abs_error"] == 0.0
    assert report["max_future_perturbation_abs_error"] == 0.0


def test_resampling_is_right_closed_and_causal() -> None:
    ts = pd.to_datetime(["2026-01-01 00:00:01", "2026-01-01 00:00:10",
                         "2026-01-01 00:00:10.5", "2026-01-01 00:00:20"], format="ISO8601")
    frame = pd.DataFrame({TIMESTAMP: ts, "x": [1.0, 2.0, 100.0, 4.0], LABEL: [0, 0, 1, 0],
                          "attack_type": ["normal", "normal", "dos", "normal"]})
    out, names = resample_causal(frame, ["x"], 10, ["mean", "max"])
    out = out.set_index(TIMESTAMP)
    # bin labelled 00:00:10 summarises (00:00:00, 00:00:10] only - not the 10.5 s event
    assert out.loc["2026-01-01 00:00:10", "x__mean"] == pytest.approx(1.5)
    assert out.loc["2026-01-01 00:00:10", LABEL] == 0
    assert out.loc["2026-01-01 00:00:20", "x__max"] == 100.0
    assert out.loc["2026-01-01 00:00:20", LABEL] == 1
    assert out.loc["2026-01-01 00:00:20", "attack_type"] == "dos"
    assert "event_count" in names
