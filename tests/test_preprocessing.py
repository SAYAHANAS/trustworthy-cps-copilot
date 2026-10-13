"""Imputation/scaling/encoding statistics come from the training split only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpscopilot.data.leakage import FitLedger, LeakageError, index_fingerprint
from cpscopilot.data.preprocessing import LeakageSafePreprocessor, TrainOnlyOneHotEncoder


def frames():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(5, 2, 200), "b": rng.normal(0, 1, 200),
                          "const": 1.0}, index=np.arange(200))
    train.loc[3, "a"] = np.nan
    test = pd.DataFrame({"a": rng.normal(500, 50, 50), "b": [np.inf] + [0.0] * 49,
                         "const": rng.normal(0, 1, 50)}, index=np.arange(300, 350))
    return train, test


def test_fit_is_refused_outside_train() -> None:
    train, _ = frames()
    for split in ("calibration", "test"):
        with pytest.raises(LeakageError):
            LeakageSafePreprocessor().fit(train, split, FitLedger())


def test_statistics_are_train_only_and_frozen() -> None:
    train, test = frames()
    ledger = FitLedger()
    pre = LeakageSafePreprocessor().fit(train, "train", ledger)
    filled = train["a"].fillna(train["a"].median())
    assert pre.center_[0] == pytest.approx(filled.mean())
    assert pre.scale_[0] == pytest.approx(filled.std(ddof=0))
    before = (pre.median_.copy(), pre.center_.copy(), pre.scale_.copy())
    out_test = pre.transform(test)
    for a, b in zip(before, (pre.median_, pre.center_, pre.scale_)):
        np.testing.assert_array_equal(a, b)  # transform never refits
    assert out_test[:, 0].mean() > 100  # test shift is visible, not normalised away
    out_train = pre.transform(train)
    np.testing.assert_allclose(out_train.mean(axis=0), 0, atol=1e-5)
    assert np.isfinite(out_test).all()  # inf/NaN imputed with train medians


def test_constant_columns_are_decided_on_train() -> None:
    train, test = frames()
    pre = LeakageSafePreprocessor().fit(train, "train", FitLedger())
    assert "const" not in pre.kept_
    assert pre.transform(test).shape[1] == 2


def test_ledger_records_the_training_rows() -> None:
    train, _ = frames()
    ledger = FitLedger()
    LeakageSafePreprocessor().fit(train, "train", ledger)
    entry = ledger.entries[0]
    assert entry["split"] == "train"
    assert entry["index_sha256"] == index_fingerprint(train.index)
    ledger.assert_not_fitted_on("test")
    ledger.record("bad", "test", 1)
    with pytest.raises(LeakageError):
        ledger.assert_not_fitted_on("test")


def test_one_hot_vocabulary_comes_from_train() -> None:
    train = pd.DataFrame({"proto": ["tcp", "udp", "tcp", "icmp"]})
    test = pd.DataFrame({"proto": ["tcp", "quic"]})
    enc = TrainOnlyOneHotEncoder(["proto"]).fit(train, "train", FitLedger())
    out = enc.transform(test)
    assert "proto=quic" not in out.columns
    assert out["proto=__other__"].tolist() == [0.0, 1.0]
    assert out["proto=tcp"].tolist() == [1.0, 0.0]
    with pytest.raises(LeakageError):
        TrainOnlyOneHotEncoder(["proto"]).fit(test, "test", FitLedger())
