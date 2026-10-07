"""Future-horizon targets: strictly future, never across segments or splits."""

from __future__ import annotations

import numpy as np
import pytest

from cpscopilot.config import FeatureConfig, SplitConfig
from cpscopilot.data.features import CausalWindowFeaturizer
from cpscopilot.data.labels import future_labels, horizon_to_steps, segmentwise_future_labels
from cpscopilot.data.leakage import audit_future_labels
from cpscopilot.data.splitting import TemporalSplitter
from cpscopilot.pipeline.runner import prepare_split
from tests.conftest import make_series_frame


def brute(y: np.ndarray, h: int, mode: str) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for t in range(len(y)):
        if t + h >= len(y) or (mode == "onset" and y[t] == 1):
            continue
        out[t] = y[t + 1: t + h + 1].max()
    return out


@pytest.mark.parametrize("h", [1, 3, 30])
@pytest.mark.parametrize("mode", ["any", "onset"])
def test_future_labels_match_brute_force(h: int, mode: str) -> None:
    rng = np.random.default_rng(h)
    y = (rng.random(400) < 0.1).astype(int)
    np.testing.assert_array_equal(future_labels(y, h, mode), brute(y, h, mode))


def test_current_label_is_not_part_of_the_future_window() -> None:
    y = np.array([0, 0, 1, 0, 0, 0])
    out = future_labels(y, 1, "any")
    assert out[1] == 1  # attack at t=2 is in (1, 2]
    assert out[2] == 0  # attack *at* t=2 is not in (2, 3]


def test_tail_is_undefined() -> None:
    out = future_labels(np.zeros(10), 4)
    assert np.isnan(out[-4:]).all() and not np.isnan(out[:-4]).any()


def test_onset_mode_excludes_ongoing_attacks() -> None:
    y = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    out = future_labels(y, 2, "onset")
    assert np.isnan(out[2]) and np.isnan(out[3])
    assert out[0] == 1 and out[1] == 1 and out[4] == 0


def test_labels_do_not_cross_segments() -> None:
    y = np.array([0, 0, 0, 1, 1, 0])
    segs = np.array([0, 0, 0, 1, 1, 1])
    out = segmentwise_future_labels(y, segs, 2)
    assert np.isnan(out[1]) and np.isnan(out[2])  # would need rows of segment 1
    assert out[0] == 0


def test_train_labels_never_see_calibration_attacks() -> None:
    frame = make_series_frame(n=1000, attack_every=1000, attack_len=1)
    frame["label"] = 0
    frame.loc[[100, 300], "label"] = 1
    split = TemporalSplitter(SplitConfig(min_positive_rows=0), 5).split(frame)
    cal_start = split["calibration"].index.min()
    frame.loc[cal_start: cal_start + 3, "label"] = 1  # attack right after the boundary
    split = TemporalSplitter(SplitConfig(min_positive_rows=0), 5).split(frame)
    train = prepare_split("train", split["train"], ["S0"],
                          CausalWindowFeaturizer(FeatureConfig(windows=[2], stats=["mean"])),
                          {"h": 20}, "any", 10.0, 5.0, [])
    tail = train.y_future["h"][-20:]
    assert np.isnan(tail).all(), "train labels near the boundary must be undefined"
    assert np.nanmax(train.y_future["h"]) == 1  # earlier in-split attacks are still labelled


def test_label_audit_detects_corruption() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    good = future_labels(y, 5)
    assert audit_future_labels(y, good, 5, "any", 50, rng)["passed"]
    leaky = good.copy()
    leaky[-5:] = 0  # labels peeking beyond the end of the split
    assert not audit_future_labels(y, leaky, 5, "any", 50, rng)["passed"]


def test_horizon_conversion() -> None:
    assert horizon_to_steps(5, 10) == 30
    assert horizon_to_steps(0.5, 1) == 30
    with pytest.raises(ValueError):
        horizon_to_steps(5, 7)
