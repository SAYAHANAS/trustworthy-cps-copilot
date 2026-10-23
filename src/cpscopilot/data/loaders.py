"""Dataset loaders producing a canonical :class:`~cpscopilot.data.schema.CPSDataset`.

Supported sources:

* ``synthetic`` - :mod:`cpscopilot.data.synthetic` (no files needed)
* ``swat`` - iTrust SWaT: the official ``.xlsx`` (read directly, cached as CSV)
  or a CSV export (``Timestamp``, sensor tags, ``Normal/Attack``)
* ``wadi`` - iTrust WADI: the 2017 export (``Date`` + ``h:mm:ss.000 AM/PM``,
  historian tag prefixes, attack-interval file) and the 2019 export
  (``WADI_attackdataLABLE.csv``: ``M/D/YY`` dates and ``mm:ss.0`` times without
  the hour, so timestamps are rebuilt from ``Row`` and checked row by row)
* ``ton_iot`` - UNSW TON_IoT processed network files ``Network_dataset_*.csv``
  (irregular flow records aggregated into fixed causal bins)
* ``generic_csv`` - anything else, driven entirely by the config

Header lines are located by content (WADI and SWaT files carry preamble rows),
``paths`` entries may be glob patterns, and files that cannot carry a timeline
(the TON_IoT ``train_test_network.csv`` sample) are rejected with an explanation.

Raw files are opened read-only and fingerprinted with SHA-256 so the runner
can prove after the run that they were not modified. Loaders never write.

Resampling is causal: bin ``(t - Δ, t]`` is labelled with its *right* edge, so
the aggregated value at ``t`` only summarises observations up to ``t``. Two
details keep that true for real data:

* **Outages are not filled.** A run of empty bins longer than
  ``max_gap_factor`` sampling periods is dropped, so the remaining timestamps
  jump and :func:`cpscopilot.data.features.segment_ids` starts a new segment;
  no window, difference, forward-fill or future label spans the outage, and no
  outage bin becomes a (fake) benign row. Shorter dropouts are kept as empty
  bins: for polled sensors they inherit the previous bin's attack state, for
  event logs (``record_kind: event``) an empty bin is "no traffic" and is an
  attack only between attack bins.
* **Flow records are placed at their observable time.** A connection log
  summarises the whole connection (bytes, duration, final state), which is only
  known when it ends; with ``event_duration_col`` each record is shifted to
  ``ts + duration`` before binning (attack intervals still match start times).

Decisions about which columns are usable (missingness) are made later on the
training split only (``features.min_train_coverage``), never on the whole file.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from cpscopilot.config import DatasetConfig
from cpscopilot.data.schema import ATTACK_TYPE, BENIGN_TYPE, LABEL, TIMESTAMP, CPSDataset
from cpscopilot.data.synthetic import generate_synthetic_cps
from cpscopilot.utils.io import sha256_file
from cpscopilot.utils.logging import get_logger

__all__ = ["expand_paths", "fingerprint_files", "load_dataset", "resample_causal"]

log = get_logger("data.loaders")

_WADI_PREFIX = re.compile(r"^.*\\")
_META_DEFAULT = [ATTACK_TYPE, "phase", "campaign_id"]
_GLOB = re.compile(r"[*?\[]")
_UNNAMED = re.compile(r"^Unnamed: \d+$")
_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$")
# WADI Time cells: 2017 export "6:00:00.000 PM"; 2019 export "00:00.0" (mm:ss, hour lost)
_WADI_CLOCK = re.compile(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?\s*[AaPp][Mm]$")
_WADI_MMSS = re.compile(r"^(\d{1,2}):(\d{2})\.\d+$")
_WADI_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$")
_WADI_ROW_ORIGIN_HOUR = 18  # every WADI export starts at 18:00:00 local time (2017 files)

# Official releases, used to warn when a file differs from what was distributed.
_KNOWN_RELEASES: dict[str, list[dict[str, Any]]] = {
    "swat": [{"release": "SWaT A1&A2 Dec 2015 attack file (SWaT_Dataset_Attack_v0)",
              "rows": 449_919, "attack_rows": 54_621,
              "start": "2015-12-28 10:00:00", "end": "2016-01-02 14:59:59"}],
    "wadi": [{"release": "WADI A2 2019 labelled attack file (WADI_attackdataLABLE.csv)",
              "rows": 172_801, "attack_rows": 9_977,
              "start": "2017-10-09 18:00:00", "end": "2017-10-11 18:00:00"}],
}
_TON_IOT_FILE_ROWS = {**{n: 1_000_000 for n in range(1, 23)}, 23: 339_021}


# ----------------------------------------------------------------- helpers
def fingerprint_files(paths: list[Path]) -> list[dict[str, Any]]:
    """SHA-256 digest, size and mtime of each raw input file."""
    return [
        {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size,
         "mtime_ns": p.stat().st_mtime_ns}
        for p in paths
    ]


def _normalise_token(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _binary_labels(series: pd.Series, positive: list[Any]) -> np.ndarray:
    if positive:
        numeric = pd.to_numeric(series, errors="coerce")
        missing = series.isna().to_numpy()
        if missing.any():
            log.warning("%d rows have an empty label cell; they are counted as normal",
                        int(missing.sum()))
        targets = {_normalise_token(v) for v in positive}
        labels = series.map(lambda v: _normalise_token(v) in targets).to_numpy(dtype=bool,
                                                                                copy=True)
        targets_num = [v for v in (_as_number(p) for p in positive) if v is not None]
        if targets_num:  # "-1", -1 and -1.0 are the same label wherever the cell is a number
            is_num = numeric.notna().to_numpy()
            labels[is_num] = numeric[is_num].isin(targets_num).to_numpy()
        return labels.astype(np.int8)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("label column is not numeric; set dataset.positive_labels")
    return (numeric.to_numpy() != 0).astype(np.int8)


def _missing_column_message(where: str, wanted: str, columns: list[str]) -> str:
    """Explain a missing column, recognising files that cannot be used as they are."""
    shown = ", ".join(columns[:12]) + (", ..." if len(columns) > 12 else "")
    msg = f"{where}: column {wanted!r} not found. Columns found: {shown}"
    lower = {c.lower() for c in columns}
    if columns and columns[0].lower() == "src_ip" and {"label", "type"} <= lower:
        msg += (
            "\nThis is the TON_IoT 'Train_Test' network sample (train_test_network.csv). It "
            "has no time column at all and its rows are grouped by attack type, so it cannot "
            "be used for time-based experiments. Download the full processed network files "
            "Network_dataset_1.csv ... Network_dataset_23.csv (folder Processed_datasets/"
            "Processed_Network_dataset) into data/raw/TON_IoT/network/; "
            "configs/experiments/ton_iot.yaml reads data/raw/TON_IoT/network/"
            "Network_dataset_*.csv.")
    elif len(columns) == 1 and (";" in columns[0] or "\t" in columns[0]):
        msg += ("\nThe whole header was read as one column: the file was probably re-saved "
                "by Excel with ';' (or tab) separators. Use the original download instead.")
    else:
        near = [c for c in columns if wanted.lower() in c.lower() or c.lower() in wanted.lower()]
        if near:
            msg += f"\nSimilar columns: {near[:5]} (set the column name in the config)."
    return msg


def _require_columns(raw: pd.DataFrame, names: list[str], where: str) -> None:
    for name in names:
        if name not in raw.columns:
            raise ValueError(_missing_column_message(where, name, [str(c) for c in raw.columns]))


def _parse_timestamps(raw: pd.DataFrame, cfg: DatasetConfig) -> pd.Series:
    if cfg.timestamp_cols:
        _require_columns(raw, list(cfg.timestamp_cols), f"dataset {cfg.name!r}")
        cols = [raw[c].astype(str).str.strip() for c in cfg.timestamp_cols]
        text = cols[0]
        for col in cols[1:]:
            text = text + " " + col
    elif cfg.timestamp_col:
        _require_columns(raw, [cfg.timestamp_col], f"dataset {cfg.name!r}")
        text = raw[cfg.timestamp_col]
    else:
        raise ValueError("dataset.timestamp_col or dataset.timestamp_cols must be set")
    if pd.api.types.is_datetime64_any_dtype(text):
        return pd.Series(text.to_numpy(), index=raw.index)
    if cfg.timestamp_unit:
        return pd.to_datetime(pd.to_numeric(text, errors="coerce"), unit=cfg.timestamp_unit)
    if text.dtype == object or pd.api.types.is_string_dtype(text):
        text = text.astype(str).str.strip()
        iso = bool(len(text)) and bool(text.str.match(_ISO_DATETIME).all())
        if cfg.timestamp_format:
            try:
                return pd.to_datetime(text, format=cfg.timestamp_format)
            except ValueError:
                if not iso:
                    raise
            # e.g. Excel date cells written out as ISO text: unambiguous, so accepted
            return pd.to_datetime(text, format="ISO8601")
        if iso:
            return pd.to_datetime(text, format="ISO8601")
    return pd.to_datetime(text, format=cfg.timestamp_format, dayfirst=cfg.dayfirst)


def _select_features(raw: pd.DataFrame, cfg: DatasetConfig, reserved: set[str]) -> list[str]:
    if cfg.feature_include:
        selected: list[str] = []
        for spec in cfg.feature_include:
            if spec.startswith("re:"):
                pattern = re.compile(spec[3:])
                selected.extend(c for c in raw.columns if pattern.search(c) and c not in selected)
            elif spec in raw.columns:
                selected.append(spec)
            else:
                raise KeyError(f"feature_include column {spec!r} not found")
    else:
        selected = list(raw.columns)
    excluded = set(cfg.feature_exclude) | reserved
    unnamed = [c for c in selected if _UNNAMED.match(str(c)) and c not in excluded]
    if unnamed:  # e.g. the row index written by pandas: it would leak position in time
        log.warning("ignoring columns without a header name: %s", unnamed[:5])
    return [c for c in selected if c not in excluded and c not in unnamed]


def _parse_interval_times(values: pd.Series, cfg: DatasetConfig) -> pd.Series:
    text = values.astype(str).str.strip()
    if cfg.attack_intervals_format:
        return pd.to_datetime(text, format=cfg.attack_intervals_format)
    try:  # ISO 8601 is unambiguous: never apply dayfirst to it
        return pd.to_datetime(text, format="ISO8601")
    except ValueError:
        return pd.to_datetime(text, dayfirst=cfg.dayfirst)


def _apply_attack_intervals(frame: pd.DataFrame, path: Path,
                            cfg: DatasetConfig) -> list[dict[str, Any]]:
    """Label rows inside each ``[start, end]`` interval; every interval must match rows.

    Interval times are parsed with ``attack_intervals_format``, as ISO 8601, or with
    the dataset's ``dayfirst`` convention, never with a guessed format, because
    attack tables and data files often use different day/month orders. Intervals
    are matched against record start times (before any ``event_duration_col`` shift).
    """
    intervals = pd.read_csv(path)
    required = {"start", "end"}
    if not required <= set(intervals.columns):
        raise ValueError(f"attack interval file needs columns {sorted(required)}")
    starts = _parse_interval_times(intervals["start"], cfg)
    ends = _parse_interval_times(intervals["end"], cfg)
    types = intervals["attack_type"] if "attack_type" in intervals else pd.Series(
        ["attack"] * len(intervals))
    ts = frame[TIMESTAMP]
    report = []
    for i, (s, e, a) in enumerate(zip(starts, ends, types)):
        if e < s:
            raise ValueError(f"attack interval {i} ends before it starts: {s} > {e}")
        mask = (ts >= s) & (ts <= e)
        matched = int(mask.sum())
        if matched == 0:
            raise ValueError(
                f"attack interval {i} ({s} .. {e}) matches no rows (data spans "
                f"{ts.min()} .. {ts.max()}); check attack_intervals_format / dayfirst")
        frame.loc[mask, LABEL] = 1
        frame.loc[mask, ATTACK_TYPE] = str(a)
        report.append({"start": str(s), "end": str(e), "attack_type": str(a),
                       "rows_matched": matched})
    return report


def _record_kind(cfg: DatasetConfig) -> str:
    if cfg.record_kind is not None:
        return cfg.record_kind
    return "event" if cfg.event_duration_col or cfg.name == "ton_iot" else "polled"


def _mode_attack_type(values: pd.Series) -> str:
    attacks = values[values != BENIGN_TYPE]
    if attacks.empty:
        return BENIGN_TYPE
    counts = attacks.value_counts()
    top = counts[counts == counts.max()].index
    return str(sorted(top)[0])  # deterministic tie-break


def resample_causal(
    frame: pd.DataFrame,
    feature_cols: list[str],
    seconds: int,
    aggs: list[str],
    count_distinct_cols: list[str] | None = None,
    max_gap_bins: float | None = None,
    stats: dict[str, Any] | None = None,
    record_kind: str = "polled",
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate rows into right-closed, right-labelled bins of ``seconds``.

    Returns the resampled frame and its feature names. The label of a bin is the
    maximum label inside it (any attack activity), the attack type is the most
    frequent non-benign type, and ``event_count`` counts raw rows per bin. A sum is
    0 in an empty bin and NaN when the bin has rows but none carries the field, so
    all aggregates of a field share its missingness.

    Args:
        max_gap_bins: runs of ``L`` consecutive empty bins with ``L + 1 >
            max_gap_bins`` (i.e. outages that would start a new segment) are
            dropped instead of being emitted as empty rows. ``None`` keeps all.
        stats: optional dict that receives bin counts for provenance.
        record_kind: how kept empty bins are labelled. ``"polled"``: a lost poll
            inherits the previous bin's label/attack type (an attack does not
            pause because one poll was lost). ``"event"``: an empty bin means no
            traffic and is an attack only when attack bins precede and follow it.
    """
    rule = f"{int(seconds)}s"
    indexed = frame.set_index(TIMESTAMP)
    grouper = indexed.resample(rule, label="right", closed="right")
    parts: list[pd.DataFrame] = []
    names: list[str] = []
    for agg in aggs:
        if agg == "sum":
            block = grouper[feature_cols].sum(min_count=1)
        else:
            block = getattr(grouper[feature_cols], agg)()
        if len(aggs) > 1:
            block = block.add_suffix(f"__{agg}")
        parts.append(block)
        names.extend(block.columns)
    counts = grouper[LABEL].count().rename("event_count").astype(float)
    parts.append(counts.to_frame())
    names.append("event_count")
    for col in count_distinct_cols or []:
        nunique = grouper[col].nunique().rename(f"{col}__nunique").astype(float)
        parts.append(nunique.to_frame())
        names.append(nunique.name)
    out = pd.concat(parts, axis=1)
    empty = counts.to_numpy() == 0
    sum_cols = ([c for c in names if c.endswith("__sum")] if len(aggs) > 1
                else list(feature_cols) if aggs == ["sum"] else [])
    out.loc[empty, sum_cols] = 0.0  # no rows in the bin: the sum is 0, not missing
    # The first bin always holds the first record, so the forward fill is defined.
    observed = grouper[LABEL].max()
    if record_kind == "event":
        label = np.minimum(observed.ffill(), observed.bfill()).fillna(0)
    else:
        label = observed.ffill().fillna(0)
    out[LABEL] = label.astype(np.int8)
    if ATTACK_TYPE in indexed:
        types = grouper[ATTACK_TYPE].apply(_mode_attack_type).reindex(out.index)
        inherited = types.where(~empty).ffill().where(out[LABEL].to_numpy() == 1)
        out[ATTACK_TYPE] = types.where(~empty, inherited).fillna(BENIGN_TYPE)
    drop = np.zeros(len(out), dtype=bool)
    if max_gap_bins is not None and empty.any():
        anchor = np.cumsum(~empty)  # empty bins share the id of the preceding filled bin
        run_len = pd.Series(empty.astype(int)).groupby(anchor).transform("sum").to_numpy()
        drop = empty & (run_len + 1 > max_gap_bins)
    if stats is not None:
        stats.update({"bins_total": int(len(out)), "bins_empty": int(empty.sum()),
                      "bins_dropped_outage": int(drop.sum())})
    out = out[~drop]
    out = out.reset_index().rename(columns={"index": TIMESTAMP})
    return out, names


def _finalise(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cfg: DatasetConfig,
    source: str,
    fingerprint: dict[str, Any],
    meta_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> CPSDataset:
    categorical_cols = [c for c in (categorical_cols or []) if c in frame]
    frame = frame.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    n_dup = int(frame[TIMESTAMP].duplicated().sum())
    fingerprint["duplicate_timestamps"] = n_dup
    if cfg.resample_seconds:
        if categorical_cols:
            raise ValueError(
                "categorical_cols cannot be averaged into time bins; list them under "
                "dataset.count_distinct_cols when resample_seconds is set"
            )
        resample_stats: dict[str, Any] = {}
        frame, feature_cols = resample_causal(
            frame, feature_cols, cfg.resample_seconds, list(cfg.resample_aggs),
            cfg.count_distinct_cols, max_gap_bins=cfg.max_gap_factor, stats=resample_stats,
            record_kind=_record_kind(cfg),
        )
        resample_stats["record_kind"] = _record_kind(cfg)
        fingerprint["resampling"] = resample_stats
        if resample_stats.get("bins_dropped_outage"):
            log.info("dropped %d empty bins inside data outages", resample_stats[
                "bins_dropped_outage"])
        meta_cols = [c for c in meta_cols if c in frame.columns]
    elif n_dup:
        log.warning("%d duplicate timestamps found; keeping the last observation", n_dup)
        frame = frame.drop_duplicates(TIMESTAMP, keep="last").reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    all_nan = [c for c in feature_cols if frame[c].isna().all()]
    if all_nan:
        log.warning("dropping %d all-NaN feature columns: %s", len(all_nan), all_nan[:5])
        feature_cols = [c for c in feature_cols if c not in all_nan]
    if cfg.max_rows:
        frame = frame.iloc[: cfg.max_rows].reset_index(drop=True)
    if cfg.sampling_seconds:
        sampling = float(cfg.sampling_seconds)
    elif cfg.resample_seconds:
        sampling = float(cfg.resample_seconds)
    else:
        deltas = frame[TIMESTAMP].diff().dt.total_seconds().dropna()
        sampling = float(deltas.median()) if len(deltas) else 1.0
        log.warning("sampling period inferred from the data (%.3f s); set "
                    "dataset.sampling_seconds to fix it explicitly", sampling)
    if ATTACK_TYPE not in frame:
        frame[ATTACK_TYPE] = np.where(frame[LABEL] == 1, "attack", BENIGN_TYPE)
    meta_cols = sorted(set(meta_cols) | {ATTACK_TYPE})
    keep = [TIMESTAMP, *feature_cols, *categorical_cols, LABEL, *meta_cols]
    frame = frame[keep].copy()
    return CPSDataset(frame=frame, feature_cols=list(feature_cols), sampling_seconds=sampling,
                      source=source, fingerprint=fingerprint, meta_cols=meta_cols,
                      categorical_cols=categorical_cols)


def _natural_key(path: Path) -> tuple[str, list[tuple[int, Any]]]:
    """Sort key comparing digit runs as numbers (Network_dataset_2 < Network_dataset_10)."""
    parts = [(0, int(t)) if t.isdigit() else (1, t.lower())
             for t in re.split(r"(\d+)", path.name) if t]
    return str(path.parent).lower(), parts


def expand_paths(entries: list[str], resolve: Callable[[str], Path]) -> list[Path]:
    """Resolve ``dataset.paths``; entries with ``* ? [`` are globbed and sorted numerically."""
    paths: list[Path] = []
    for entry in entries:
        parts = re.split(r"[\\/]", str(entry))
        first = next((i for i, part in enumerate(parts) if _GLOB.search(part)), None)
        if first is None:
            paths.append(resolve(entry))
            continue
        base = resolve("/".join(parts[:first]) or ".")
        matches = sorted((m for m in base.glob("/".join(parts[first:])) if m.is_file()),
                         key=_natural_key)
        if not matches:
            raise FileNotFoundError(f"no raw data file matches {entry!r} (looked in {base})")
        paths.extend(matches)
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            raise ValueError(f"raw data file listed more than once: {path}")
        seen.add(path)
    return paths


def _clean_name(name: Any) -> str:
    return str(name).replace("\ufeff", "").strip()


def _header_predicate(cfg: DatasetConfig) -> Callable[[list[str]], bool] | None:
    """Recognises the header line of datasets whose files carry preamble lines."""
    if cfg.name == "wadi":
        # 2019 label file: a '0,1,...,130' row first; 2017 14-day file: 4 lines first
        return lambda f: [_clean_name(x).lower() for x in f[:3]] == ["row", "date", "time"]
    if cfg.name == "swat":
        target = _clean_name(cfg.timestamp_col or "Timestamp").lower()
        return lambda f: any(_clean_name(x).lower() == target for x in f)
    return None


def _locate_header(lines: list[list[str]], accept: Callable[[list[str]], bool],
                   where: str) -> int:
    for i, fields in enumerate(lines):
        if accept(fields):
            return i
    preview = "\n".join("  " + ",".join(f)[:120] for f in lines[:5])
    raise ValueError(f"{where}: no header line found in the first {len(lines)} lines. "
                     f"First lines:\n{preview}")


def _needed_columns(cfg: DatasetConfig) -> set[str] | None:
    """Columns to read when the feature list is explicit (saves memory on large files)."""
    if not cfg.feature_include or any(f.startswith("re:") for f in cfg.feature_include):
        return None
    cols = {*cfg.feature_include, *cfg.categorical_cols, *cfg.count_distinct_cols,
            *cfg.timestamp_cols}
    cols |= {c for c in (cfg.timestamp_col, cfg.label_col, cfg.attack_type_col,
                         cfg.event_duration_col) if c}
    return cols


def _optional_columns(cfg: DatasetConfig) -> set[str]:
    """Columns read when present although not required (defaults resolved after reading)."""
    return {"duration"} if cfg.name == "ton_iot" and cfg.event_duration_col is None else set()


def _merge_duplicate_columns(table: pd.DataFrame, where: str) -> pd.DataFrame:
    """Merge columns whose names differ only by surrounding spaces (' MV101' / 'MV101').

    This happens when files with differently padded headers were concatenated; the
    columns are complementary (one is empty where the other is filled).
    """
    names = list(table.columns)
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if not duplicated:
        return table
    cols: dict[str, pd.Series] = {}
    for name in dict.fromkeys(names):
        idx = [i for i, n in enumerate(names) if n == name]
        block = table.iloc[:, idx]
        if len(idx) > 1 and (block.notna().sum(axis=1) > 1).any():
            raise ValueError(f"{where}: column {name!r} appears {len(idx)} times (ignoring "
                             "spaces) with values in the same rows; keep only one of them")
        cols[name] = block.bfill(axis=1).iloc[:, 0] if len(idx) > 1 else block.iloc[:, 0]
    log.warning("%s: merged columns whose names differ only by spaces: %s", where,
                duplicated[:10])
    return pd.DataFrame(cols)


def _read_csv_table(path: Path, cfg: DatasetConfig) -> pd.DataFrame:
    kwargs: dict[str, Any] = {"low_memory": False, **cfg.read_csv_kwargs}
    accept = _header_predicate(cfg)
    if accept is not None and "skiprows" not in kwargs and "header" not in kwargs:
        delimiter = kwargs.get("sep") or kwargs.get("delimiter") or ","
        with path.open("r", encoding=kwargs.get("encoding", "utf-8-sig"), errors="replace",
                       newline="") as fh:
            head = [row for _, row in zip(range(25), csv.reader(fh, delimiter=delimiter))]
        try:
            kwargs["skiprows"] = _locate_header(head, accept, str(path))
        except ValueError as exc:
            if delimiter == "," and any(";" in ",".join(r) for r in head[:5]):
                raise ValueError(f"{exc}\nThe lines contain ';': the file was probably "
                                 "re-saved by Excel with ';' separators; use the original "
                                 "download (or set read_csv_kwargs sep/decimal)") from exc
            raise
    header_kwargs = {k: v for k, v in kwargs.items() if k not in ("low_memory", "dtype")}
    raw_names = list(pd.read_csv(path, nrows=0, **header_kwargs).columns)
    header = [_clean_name(c) for c in raw_names]
    needed = _needed_columns(cfg)
    if needed is not None:
        missing = sorted(needed - set(header))
        if missing:
            raise ValueError(_missing_column_message(str(path), missing[0], header))
        wanted = needed | _optional_columns(cfg)
        kwargs["usecols"] = [r for r, c in zip(raw_names, header) if c in wanted]
    categorical = set(cfg.categorical_cols) | set(cfg.count_distinct_cols)
    if cfg.attack_type_col:
        categorical.add(cfg.attack_type_col)
    if "dtype" not in kwargs:  # string columns as categories: far less memory on big logs
        kwargs["dtype"] = {r: "category" for r, c in zip(raw_names, header) if c in categorical}
    table = pd.read_csv(path, **kwargs)
    table.columns = [_clean_name(c) for c in table.columns]
    return _merge_duplicate_columns(table, str(path))


def _coerce_numeric(table: pd.DataFrame, columns: list[str], counts: dict[str, int]) -> None:
    """Convert text-typed numeric fields in place, counting tokens that are not numbers.

    Done per file so that one stray token (e.g. '0.0.0.0' in TON_IoT ``src_bytes``)
    does not turn a column of millions of rows into Python objects.
    """
    for col in columns:
        if col not in table or pd.api.types.is_numeric_dtype(table[col]):
            continue
        values = pd.to_numeric(table[col], errors="coerce")
        text = table[col][values.isna() & table[col].notna()].astype(str).str.strip()
        n_bad = int((~text.isin(["", "-"])).sum())
        if n_bad:
            counts[col] = counts.get(col, 0) + n_bad
        table[col] = values


def _excel_header_rule(fields: list[str]) -> bool:
    """Generic header row: at least half of the cells filled (skips banner rows)."""
    filled = sum(1 for x in fields if _clean_name(x) and _clean_name(x).lower() != "nan")
    return filled >= max(1, len(fields) // 2)


def _read_excel_table(path: Path, cfg: DatasetConfig,
                      resolve: Callable[[str], Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the first sheet of an Excel file through a CSV cache keyed by its SHA-256.

    Excel is slow to parse (minutes for the 450k-row SWaT workbook), so the sheet
    is converted once; the raw workbook is only read, and the cache is reused
    only for byte-identical sources read with the same header rule.
    """
    digest = sha256_file(path)
    rule = re.sub(r"[^A-Za-z0-9]+", "-", f"{cfg.name}-{cfg.timestamp_col or ''}").strip("-")
    cache = resolve(cfg.excel_cache_dir) / f"{path.stem}.{digest[:16]}.{rule}.csv"
    if not cache.exists():
        log.info("converting %s to %s (one-off, may take a few minutes)", path.name, cache)
        grid = pd.read_excel(path, sheet_name=0, header=None, dtype=object)
        accept = _header_predicate(cfg) or _excel_header_rule
        head = [["" if pd.isna(v) else str(v) for v in row]
                for row in grid.head(25).itertuples(index=False)]
        h = _locate_header(head, accept, str(path))
        table = grid.iloc[h + 1:].reset_index(drop=True)
        table.columns = [_clean_name(c) for c in grid.iloc[h]]
        table = table.dropna(how="all")
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
        table.to_csv(tmp, index=False)
        os.replace(tmp, cache)  # atomic: a crash never leaves a truncated cache
    table = pd.read_csv(cache, low_memory=False)
    table.columns = [_clean_name(c) for c in table.columns]
    return _merge_duplicate_columns(table, str(path)), {
        "source": str(path), "sha256": digest, "cache": str(cache),
        "cache_sha256": sha256_file(cache)}


def _concat_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate by column name, keeping categorical columns categorical (memory)."""
    if len(tables) == 1:
        return tables[0]
    names = list(dict.fromkeys(c for t in tables for c in t.columns))
    cats = [c for c in names
            if all(c in t and isinstance(t[c].dtype, pd.CategoricalDtype) for t in tables)]
    out = pd.concat([t.drop(columns=cats) for t in tables], ignore_index=True)
    for c in cats:  # same category dtype everywhere (an all-blank file has object categories)
        parts = [pd.Categorical.from_codes(t[c].array.codes, categories=pd.Index(
            [str(v) for v in t[c].array.categories], dtype=object)) for t in tables]
        out[c] = union_categoricals(parts, ignore_order=True)
    return out[[c for c in names if c in out.columns]]


def _read_tables(cfg: DatasetConfig, resolve: Callable[[str], Path],
                 per_table: Callable[[pd.DataFrame, Path, dict[str, Any]], pd.DataFrame]
                 | None = None) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    """Read every input file, apply ``per_table`` to each, then concatenate."""
    if not cfg.paths:
        raise ValueError(f"dataset {cfg.name!r} requires dataset.paths")
    paths = expand_paths(list(cfg.paths), resolve)
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"raw data file not found: {p}")
    files = fingerprint_files(paths)
    by_digest: dict[str, list[str]] = {}
    for f in files:
        by_digest.setdefault(f["sha256"], []).append(Path(f["path"]).name)
    copies = [names for names in by_digest.values() if len(names) > 1]
    if copies:
        raise ValueError(f"identical files would be counted twice: {copies}; remove the "
                         "duplicate download(s) or narrow dataset.paths")
    tables = []
    info: dict[str, Any] = {"files": files, "rows_per_file": {}, "excel": [], "per_file": {}}
    coerced: dict[str, int] = {}
    for p in paths:
        if p.suffix.lower() in _EXCEL_SUFFIXES:
            table, excel = _read_excel_table(p, cfg, resolve)
            info["excel"].append(excel)
        else:
            table = _read_csv_table(p, cfg)
            if _needed_columns(cfg) is not None:
                _coerce_numeric(table, list(cfg.feature_include or []), coerced)
        info["rows_per_file"][p.name] = len(table)
        if per_table is not None:
            details: dict[str, Any] = {}
            table = per_table(table, p, details)
            if details:
                info["per_file"][p.name] = details
        tables.append(table)
    for key in ("excel", "per_file"):
        if not info[key]:
            del info[key]
    if coerced:
        info["non_numeric_values_coerced"] = coerced
    return _concat_tables(tables), paths, info


def _clean_text_column(values: pd.Series) -> Any:
    """Stripped strings with missing cells kept missing (same on pandas 2 and 3).

    Categorical input stays categorical (one code per row).
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        cats = values.cat.categories.astype(str).str.strip()
        if cats.is_unique:
            return values.cat.rename_categories(cats).array
    missing = values.isna().to_numpy()
    out = np.array(values.astype(str).str.strip().astype(object), dtype=object)  # writable
    out[missing] = np.nan
    return out


def _check_release(name: str, frame: pd.DataFrame,
                   own_labels: bool = True) -> dict[str, Any]:
    """Compare the loaded rows with the officially distributed file they resemble.

    ``own_labels=False`` (labels from an interval file) skips the attack-row count,
    which then describes the intervals rather than the file.
    """
    ts = pd.to_datetime(frame[TIMESTAMP])
    stats = {"rows": int(len(frame)), "attack_rows": int(frame[LABEL].sum()),
             "start": str(ts.min()), "end": str(ts.max()),
             "duplicate_timestamps": int(ts.duplicated().sum())}
    compared = ("rows", "attack_rows") if own_labels else ("rows",)
    for known in _KNOWN_RELEASES[name]:
        if stats["start"] != known["start"] or stats["end"] != known["end"]:
            continue
        diffs = [f"{k}: {stats[k]:,} vs {known[k]:,} in the official file"
                 for k in compared if stats[k] != known[k]]
        if stats["duplicate_timestamps"]:
            diffs.append(f"duplicate_timestamps: {stats['duplicate_timestamps']:,} vs 0")
        if diffs:
            log.warning("%s spans the same period as the %s but differs (%s); it may be an "
                        "edited or merged copy: compare it with the original download",
                        name, known["release"], "; ".join(diffs))
        else:
            log.info("%s matches the %s (%s rows, %s attack rows)", name, known["release"],
                     f"{stats['rows']:,}", f"{stats['attack_rows']:,}")
        return {**stats, "release": known["release"], "matches_release": not diffs,
                "differences": diffs, "compared": list(compared)}
    official = "; ".join(f"{k['release']}: {k['rows']:,} rows, {k['attack_rows']:,} attack rows, "
                         f"{k['start']} -> {k['end']}" for k in _KNOWN_RELEASES[name])
    log.warning("%s: the file spans %s -> %s with %s rows, unlike the official release (%s). "
                "Fine for another release or a deliberate subset; otherwise the copy may be "
                "truncated or edited", name, stats["start"], stats["end"], f"{stats['rows']:,}",
                official)
    return {**stats, "release": None, "matches_release": False, "differences": [],
            "compared": list(compared), "note": f"official: {official}"}


def _wadi_dates(date: pd.Series, row: pd.Series) -> pd.Series:
    """Parse WADI Date cells (M/D/YYYY in 2017, M/D/YY or M/D/YYYY mixed in 2019)."""
    parts = date.str.extract(_WADI_DATE)
    bad = parts.isna().any(axis=1).to_numpy()
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"WADI Date {date.iloc[i]!r} at Row {row.iloc[i]} is not M/D/YY or "
                         "M/D/YYYY; the file may have been re-saved with another date format")
    month, day, year = (parts[k].astype(int) for k in (0, 1, 2))
    year = year.where(year >= 100, year + 2000)
    try:
        return pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}))
    except ValueError as exc:
        raise ValueError(f"WADI Date column has an impossible month/day ({exc}); WADI dates "
                         "are month-first (10/9/17 = 9 Oct 2017)") from exc


def _wadi_timestamps(raw: pd.DataFrame, cfg: DatasetConfig,
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Timestamps of a WADI export, from Date + Time or, for the 2019 export, from Row.

    The 2019 (A2) files store Time as Excel's ``mm:ss.0`` rendering, so the hour is
    lost in every row. WADI is sampled at 1 Hz and ``Row`` keeps the original
    numbering, hence ``time = origin + (Row - 1) s`` with the origin at 18:00:00 on
    the date of Row 1. Every row's date and minute:second are checked against the
    rebuilt time; the date changes at midnight also pin down the hour.
    """
    _require_columns(raw, ["Row", "Date", "Time"], "WADI file")
    row_num = pd.to_numeric(raw["Row"], errors="coerce")
    junk = row_num.isna().to_numpy()
    info: dict[str, Any] = {"rows_without_row_number_dropped": int(junk.sum())}
    if junk.any():  # the 2019 label file ends with two rows of commas plus a label
        log.info("WADI: dropping %d rows without a Row number", int(junk.sum()))
        raw = raw.loc[~junk].reset_index(drop=True)
        row_num = row_num[~junk].reset_index(drop=True)
    if (row_num % 1 != 0).any():
        raise ValueError("WADI Row column contains non-integer values")
    row = row_num.astype(np.int64)
    date = raw["Date"].astype(str).str.strip()
    time = raw["Time"].astype(str).str.strip()
    dates = _wadi_dates(date, row)
    clock = time.str.match(_WADI_CLOCK).to_numpy()
    mmss = time.str.match(_WADI_MMSS).to_numpy()
    if clock.all():  # 2017 export: "6:00:00.000 PM"
        fmt = "%I:%M:%S.%f %p" if time.str.contains(".", regex=False).all() else "%I:%M:%S %p"
        tod = pd.to_datetime(time.str.upper(), format=fmt)
        ts = dates + (tod - tod.dt.normalize())
        info["timestamp_source"] = "date_and_time_columns"
    elif mmss.all():  # 2019 export: "00:00.0", hour not stored
        if cfg.row_time_origin:
            origin = pd.Timestamp(cfg.row_time_origin)
        else:
            first = (row == 1).to_numpy()
            if not first.any():
                raise ValueError(
                    "WADI 2019 export: the Time column has no hour (mm:ss.0) and Row 1 is not "
                    "in the file, so the time of the first row is unknown; set "
                    "dataset.row_time_origin to the date and time of Row 1")
            origin = dates[first].iloc[0] + pd.Timedelta(hours=_WADI_ROW_ORIGIN_HOUR)
        ts = origin + pd.to_timedelta(row - 1, unit="s")
        found = time.str.extract(_WADI_MMSS).astype(int)
        ok = ((ts.dt.minute == found[0]) & (ts.dt.second == found[1])
              & (ts.dt.normalize() == dates)).to_numpy()
        if not ok.all():
            i = int(np.argmin(ok))
            raise ValueError(
                f"WADI timestamps rebuilt from Row do not match the file at Row {row.iloc[i]}: "
                f"expected {ts.iloc[i]:%m/%d/%y} {ts.iloc[i]:%M:%S} ({origin} + Row - 1 s), "
                f"found Date={date.iloc[i]!r} Time={time.iloc[i]!r}. Rows were removed, "
                "re-ordered or re-dated in this copy; use the original export, or set "
                "dataset.row_time_origin if the file does not start at 18:00:00.")
        midnights = int((dates.diff().dt.days.fillna(0) != 0).sum())
        if not midnights:
            log.warning("WADI: no midnight inside the file, so the 18:00:00 start of Row 1 "
                        "cannot be checked against the dates")
        log.info("WADI 2019 export: Time has no hour; timestamps rebuilt as %s + (Row - 1) s "
                 "and checked on all %d rows (%d date changes)", origin, len(row), midnights)
        info.update({"timestamp_source": "row_number", "row_1_time": str(origin),
                     "rows_checked": int(len(row)), "date_changes_checked": midnights})
    else:
        i = int(np.argmin(clock | mmss)) if (clock | mmss).any() else 0
        raise ValueError(
            f"WADI Time column: value {time.iloc[i]!r} at Row {row.iloc[i]} matches neither "
            "the 2017 export ('6:00:00.000 PM') nor the 2019 export ('00:00.0'), or the "
            "column mixes both; the file may have been re-saved by a spreadsheet program")
    raw = raw.copy()  # one consolidated block before adding a column (wide files)
    raw["timestamp_rebuilt"] = ts.to_numpy()
    return raw, info


def _from_table(raw: pd.DataFrame, cfg: DatasetConfig, source: str,
                paths: list[Path], resolve: Callable[[str], Path],
                extra: dict[str, Any] | None = None) -> CPSDataset:
    if cfg.label_col is None and cfg.attack_intervals_path is None:
        raise ValueError("set dataset.label_col or dataset.attack_intervals_path")
    timestamps = _parse_timestamps(raw, cfg)
    columns: dict[str, Any] = {TIMESTAMP: timestamps.to_numpy()}
    if cfg.label_col is not None:
        _require_columns(raw, [cfg.label_col], f"dataset {cfg.name!r}")
        columns[LABEL] = _binary_labels(raw[cfg.label_col], cfg.positive_labels)
    else:
        columns[LABEL] = np.zeros(len(raw), dtype=np.int8)
    if cfg.attack_type_col and cfg.attack_type_col in raw:
        types = raw[cfg.attack_type_col].astype(str).str.strip().str.lower().to_numpy()
        columns[ATTACK_TYPE] = np.where(columns[LABEL] == 1, types, BENIGN_TYPE)
    else:
        columns[ATTACK_TYPE] = np.where(columns[LABEL] == 1, "attack", BENIGN_TYPE)
    reserved = {cfg.label_col or "", cfg.attack_type_col or "", cfg.timestamp_col or "",
                *cfg.timestamp_cols, *cfg.categorical_cols, *cfg.count_distinct_cols}
    feature_cols: list[str] = []
    coerced: dict[str, int] = dict((extra or {}).get("read", {}).get(
        "non_numeric_values_coerced", {}))  # counted per file by the reader
    for col in _select_features(raw, cfg, reserved):
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.notna().sum() == 0:  # no numeric value anywhere (text column)
            log.info("skipping non-numeric column %r", col)
            continue
        if not pd.api.types.is_numeric_dtype(raw[col]):  # text tokens, e.g. '0.0.0.0'
            text = raw[col][values.isna() & raw[col].notna()].astype(str).str.strip()
            n_bad = int((~text.isin(["", "-"])).sum())
            if n_bad:
                coerced[col] = n_bad
        # Partial missingness is judged on the training split only (runner).
        columns[col] = values.to_numpy(dtype=np.float64)
        feature_cols.append(col)
    if coerced:
        log.warning("non-numeric values treated as missing: %s", coerced)
    for col in [*cfg.categorical_cols, *cfg.count_distinct_cols]:
        if col in raw and col not in columns:
            columns[col] = _clean_text_column(raw[col])
    frame = pd.DataFrame(columns)
    extra = dict(extra or {})
    read = dict(extra.get("read", {}))
    files = read.pop("files", None)  # hashed before reading by _read_tables
    if read:
        extra["read"] = read
    fingerprint: dict[str, Any] = {
        "files": files if files is not None else fingerprint_files(paths), "loader": source,
        **extra}
    if coerced:
        fingerprint["non_numeric_values_coerced"] = coerced
    if cfg.attack_intervals_path:
        fingerprint["attack_intervals"] = _apply_attack_intervals(
            frame, resolve(cfg.attack_intervals_path), cfg)
    if cfg.name in _KNOWN_RELEASES:
        fingerprint["release_check"] = _check_release(
            cfg.name, frame, own_labels=not cfg.attack_intervals_path)
    if cfg.event_duration_col:
        # Intervals describe when records happened (above); a flow record is only
        # observable once the connection has ended, so it is binned at that time.
        duration = pd.to_numeric(raw[cfg.event_duration_col], errors="coerce")
        frame[TIMESTAMP] = frame[TIMESTAMP] + pd.to_timedelta(
            duration.fillna(0.0).clip(lower=0.0).to_numpy(), unit="s")
    return _finalise(frame, feature_cols, cfg, source, fingerprint, [ATTACK_TYPE],
                     categorical_cols=list(cfg.categorical_cols))


# ----------------------------------------------------------------- sources
def _load_synthetic(cfg: DatasetConfig, seed: int) -> CPSDataset:
    frame, campaigns = generate_synthetic_cps(cfg.synthetic, seed)
    feature_cols = [c for c in frame.columns if c not in {TIMESTAMP, LABEL, *_META_DEFAULT}]
    digest = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    import hashlib

    fingerprint = {
        "loader": "synthetic",
        "generator_seed": seed,
        "frame_sha256": hashlib.sha256(digest.tobytes()).hexdigest(),
        "campaigns": [c.__dict__ for c in campaigns],
        "files": [],
    }
    cfg_local = cfg.model_copy(update={"sampling_seconds": float(cfg.synthetic.sampling_seconds)})
    return _finalise(frame, feature_cols, cfg_local, "synthetic", fingerprint, _META_DEFAULT)


def _swat_defaults(cfg: DatasetConfig) -> DatasetConfig:
    return cfg.model_copy(update={
        "timestamp_col": cfg.timestamp_col or "Timestamp",
        "timestamp_format": cfg.timestamp_format or "%d/%m/%Y %I:%M:%S %p",
        "label_col": cfg.label_col or "Normal/Attack",
        "positive_labels": cfg.positive_labels or ["Attack", "A ttack"],
        "dayfirst": True,
    })


def _wadi_defaults(cfg: DatasetConfig, raw: pd.DataFrame) -> DatasetConfig:
    built_in = [c for c in raw.columns if c.lower().startswith("attack")]
    label_col = cfg.label_col
    if label_col is None and cfg.attack_intervals_path is None:
        # with an interval file the intervals define the labels (e.g. to relabel 2019 files)
        label_col = built_in[0] if built_in else None
    update: dict[str, Any] = {
        "label_col": label_col,
        # a label column is never an input, even when the intervals replace it
        "feature_exclude": [*cfg.feature_exclude, "Row", "Date", "Time", *built_in],
    }
    if not (cfg.timestamp_col or cfg.timestamp_cols):
        update["timestamp_col"] = "timestamp_rebuilt"
    if label_col is not None:
        _require_columns(raw, [label_col], "WADI file")
        values = set(pd.to_numeric(raw[label_col], errors="coerce").dropna().unique())
        if not values <= {1.0, -1.0} and not cfg.positive_labels:
            raise ValueError(f"WADI label column {label_col!r} has values {sorted(values)[:5]}; "
                             "expected 1 (normal) and -1 (attack): set dataset.positive_labels")
        if not cfg.positive_labels:
            update["positive_labels"] = [-1]
    return cfg.model_copy(update=update)


def _check_ton_iot_files(info: dict[str, Any]) -> None:
    """Warn when a full TON_IoT network file has fewer rows than distributed."""
    short = {}
    for name, rows in info.get("rows_per_file", {}).items():
        m = re.fullmatch(r"Network_dataset_(\d+)\.csv", name)
        expected = _TON_IOT_FILE_ROWS.get(int(m.group(1))) if m else None
        if expected is not None and rows != expected:
            short[name] = f"{rows:,} rows vs {expected:,}"
    if short:
        log.warning("TON_IoT files differ from the distributed row counts (truncated or "
                    "partial download?): %s", short)
        info["row_count_differences"] = short


def load_dataset(cfg: DatasetConfig, resolve: Callable[[str], Path], seed: int) -> CPSDataset:
    """Load and canonicalise the configured dataset.

    Args:
        cfg: dataset section of the experiment config.
        resolve: maps config paths to absolute paths (``ExperimentConfig.resolve_path``).
        seed: experiment seed (used by the synthetic generator only).
    """
    if cfg.name == "synthetic":
        return _load_synthetic(cfg, seed)
    if cfg.name == "swat":
        cfg = _swat_defaults(cfg)
    per_table = None
    if cfg.name == "wadi":
        rebuild = not (cfg.timestamp_col or cfg.timestamp_cols)

        def per_table(table: pd.DataFrame, path: Path, details: dict[str, Any]) -> pd.DataFrame:
            # per file: the 14-day and attack files each have their own Row numbering
            table.columns = [_WADI_PREFIX.sub("", c) for c in table.columns]
            if rebuild:
                try:
                    table, wadi_info = _wadi_timestamps(table, cfg)
                except ValueError as exc:
                    raise ValueError(f"{path.name}: {exc}") from exc
                details.update(wadi_info)
            return table

    raw, paths, info = _read_tables(cfg, resolve, per_table)
    if cfg.name == "swat":
        return _from_table(raw, cfg, "swat", paths, resolve, {"read": info})
    if cfg.name == "wadi":
        return _from_table(raw, _wadi_defaults(cfg, raw), "wadi", paths, resolve, {"read": info})
    if cfg.name == "ton_iot":
        if not cfg.resample_seconds:
            raise ValueError("TON_IoT events are irregular: set dataset.resample_seconds")
        if cfg.event_duration_col is None and "duration" in raw.columns:
            cfg = cfg.model_copy(update={"event_duration_col": "duration"})
        _check_ton_iot_files(info)
        return _from_table(raw, cfg, "ton_iot", paths, resolve, {"read": info})
    return _from_table(raw, cfg, "generic_csv", paths, resolve, {"read": info})
