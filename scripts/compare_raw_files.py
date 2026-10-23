#!/usr/bin/env python3
"""Check that two copies of a dataset hold the same rows, labels and values.

Typical use: confirm that a CSV export of SWaT is a faithful copy of the official
workbook before using it::

    python scripts/compare_raw_files.py --config configs/experiments/swat.yaml \
        data/raw/SWaT/SWaT_Dataset_Attack_v0.xlsx data/raw/SWaT/merged.csv

Both files are loaded with the dataset's loader (without resampling) and compared
on timestamps, labels and every common numeric column. Exit code 0 means
identical (up to ``--tolerance``), 1 means differences were found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpscopilot.config import load_config  # noqa: E402
from cpscopilot.data.loaders import load_dataset  # noqa: E402
from cpscopilot.data.schema import LABEL, TIMESTAMP  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment YAML of the dataset")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="largest accepted absolute difference of a value")
    args = parser.parse_args(argv)
    frames, problems = [], []
    for name, path in (("first", args.first), ("second", args.second)):
        cfg = load_config(args.config, [f"dataset.paths=[{Path(path).resolve().as_posix()}]",
                                        "dataset.resample_seconds=null"])
        ds = load_dataset(cfg.dataset, cfg.resolve_path, cfg.seed)
        frames.append(ds.frame.set_index(TIMESTAMP))
        print(f"{path}: {len(ds.frame):,} rows, {int(ds.labels.sum()):,} attack rows, "
              f"{ds.frame[TIMESTAMP].min()} -> {ds.frame[TIMESTAMP].max()}")
        if ds.fingerprint.get("duplicate_timestamps"):
            problems.append(f"{ds.fingerprint['duplicate_timestamps']:,} duplicate timestamps "
                            f"in the {name} file")
    a, b = frames
    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    if len(only_a) or len(only_b):
        problems.append(f"timestamps only in the first file: {len(only_a):,}, only in the "
                        f"second: {len(only_b):,} (e.g. {list(only_a[:2]) + list(only_b[:2])})")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    n_label = int((a[LABEL] != b[LABEL]).sum())
    if n_label:
        problems.append(f"labels differ on {n_label:,} rows")
    numeric = pd.api.types.is_numeric_dtype
    cols = [c for c in a.columns if c in b.columns and c != LABEL
            and numeric(a[c]) and numeric(b[c])]
    for col in cols:
        diff = np.abs(a[col].to_numpy(dtype=float) - b[col].to_numpy(dtype=float))
        both_nan = np.isnan(a[col].to_numpy(dtype=float)) & np.isnan(b[col].to_numpy(dtype=float))
        bad = np.nan_to_num(diff, nan=np.inf)[~both_nan] > args.tolerance
        if bad.any():
            problems.append(f"column {col}: {int(bad.sum()):,} rows differ "
                            f"(largest difference {np.nanmax(diff):.6g})")
    missing = sorted(set(a.columns) ^ set(b.columns))
    if missing:
        problems.append(f"columns present in only one file: {missing}")
    if problems:
        print("DIFFERENT:\n  " + "\n  ".join(problems))
        return 1
    print(f"IDENTICAL: {len(common):,} rows, labels and {len(cols)} numeric columns match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
