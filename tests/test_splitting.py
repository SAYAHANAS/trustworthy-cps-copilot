"""Chronological split bounds and embargo guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpscopilot.config import SplitConfig
from cpscopilot.data.schema import LABEL, TIMESTAMP
from cpscopilot.data.splitting import SPLIT_ORDER, SplitError, TemporalSplitter
from tests.conftest import make_series_frame


@pytest.mark.parametrize("embargo", [0, 1, 17, 60])
@pytest.mark.parametrize("fracs", [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.7, 0.15, 0.15)])
def test_split_bounds_are_chronological_and_disjoint(embargo: int, fracs: tuple) -> None:
    frame = make_series_frame(n=1200, attack_every=40, attack_len=5)
    cfg = SplitConfig(train_frac=fracs[0], calibration_frac=fracs[1], test_frac=fracs[2])
    split = TemporalSplitter(cfg, embargo).split(frame)
    tr, ca, te = (split[n] for n in SPLIT_ORDER)

    # strict temporal order: every train time < every calibration time < every test time
    assert tr[TIMESTAMP].max() < ca[TIMESTAMP].min()
    assert ca[TIMESTAMP].max() < te[TIMESTAMP].min()
    # disjoint rows, contiguous blocks, and exactly `embargo` purged rows per boundary
    assert not (set(tr.index) & set(ca.index)) and not (set(ca.index) & set(te.index))
    for part in (tr, ca, te):
        assert np.all(np.diff(part.index.to_numpy()) == 1)
    assert ca.index.min() - tr.index.max() - 1 == embargo
    assert te.index.min() - ca.index.max() - 1 == embargo
    assert len(tr) + len(ca) + len(te) + 2 * embargo == len(frame)
    assert te.index.max() == frame.index.max()  # test is the most recent period
    audit = split.audit(sampling_seconds=10.0)
    assert audit["passed"]
    assert audit["discarded_rows"] == 2 * embargo
    assert all(g >= embargo * 10.0 for g in audit["boundary_gaps_seconds"])


def test_split_fractions_are_respected() -> None:
    frame = make_series_frame(n=1000, attack_every=40)
    split = TemporalSplitter(SplitConfig(train_frac=0.6, calibration_frac=0.2,
                                         test_frac=0.2), 0).split(frame)
    sizes = {n: len(split[n]) for n in SPLIT_ORDER}
    assert sizes == {"train": 600, "calibration": 200, "test": 200}


def test_unsorted_frame_is_rejected() -> None:
    frame = make_series_frame(n=300).iloc[::-1]
    with pytest.raises(SplitError):
        TemporalSplitter(SplitConfig(), 0).split(frame)


def test_split_without_attacks_is_rejected() -> None:
    frame = make_series_frame(n=300)
    frame[LABEL] = 0
    with pytest.raises(SplitError, match="attack rows"):
        TemporalSplitter(SplitConfig(min_positive_rows=1), 0).split(frame)


def test_timestamp_strategy_with_embargo() -> None:
    frame = make_series_frame(n=900, attack_every=40)
    start = frame[TIMESTAMP].iloc[0]
    cfg = SplitConfig(strategy="timestamp", train_end=str(start + pd.Timedelta(seconds=4000)),
                      calibration_end=str(start + pd.Timedelta(seconds=6500)))
    split = TemporalSplitter(cfg, 12).split(frame)
    assert split["train"][TIMESTAMP].max() < pd.Timestamp(cfg.train_end)
    assert split["calibration"][TIMESTAMP].min() >= pd.Timestamp(cfg.train_end) + \
        pd.Timedelta(seconds=12 * 10)
    assert split["calibration"][TIMESTAMP].max() < pd.Timestamp(cfg.calibration_end)
    assert split.audit(10.0)["passed"]


def test_too_large_embargo_fails() -> None:
    with pytest.raises(SplitError):
        TemporalSplitter(SplitConfig(), 200).split(make_series_frame(n=300))
