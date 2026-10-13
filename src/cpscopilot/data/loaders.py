"""Dataset loaders producing a canonical :class:`~cpscopilot.data.schema.CPSDataset`.

Supported sources:

* ``synthetic`` - :mod:`cpscopilot.data.synthetic` (no files needed)
* ``swat`` - iTrust SWaT CSV exports (``Timestamp``, sensor tags, ``Normal/Attack``)
* ``wadi`` - iTrust WADI CSV exports (``Date`` + ``Time``, long historian tag
  names, optional ``Attack LABLE (1:No Attack, -1:Attack)`` column or an
  external attack-interval file)
* ``ton_iot`` - UNSW TON_IoT network/telemetry CSVs (irregular events are
  aggregated into fixed causal bins)
* ``generic_csv`` - anything else, driven entirely by the config

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

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cpscopilot.config import DatasetConfig
from cpscopilot.data.schema import ATTACK_TYPE, BENIGN_TYPE, LABEL, TIMESTAMP, CPSDataset
from cpscopilot.data.synthetic import generate_synthetic_cps
from cpscopilot.utils.io import sha256_file
from cpscopilot.utils.logging import get_logger

__all__ = ["fingerprint_files", "load_dataset", "resample_causal"]

log = get_logger("data.loaders")

_WADI_PREFIX = re.compile(r"^.*\\")
_META_DEFAULT = [ATTACK_TYPE, "phase", "campaign_id"]


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


def _binary_labels(series: pd.Series, positive: list[Any]) -> np.ndarray:
    if positive:
        targets = {_normalise_token(v) for v in positive}
        return series.map(lambda v: _normalise_token(v) in targets).to_numpy(dtype=np.int8)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("label column is not numeric; set dataset.positive_labels")
    return (numeric.to_numpy() != 0).astype(np.int8)


def _parse_timestamps(raw: pd.DataFrame, cfg: DatasetConfig) -> pd.Series:
    if cfg.timestamp_cols:
        cols = [raw[c].astype(str).str.strip() for c in cfg.timestamp_cols]
        text = cols[0]
        for col in cols[1:]:
            text = text + " " + col
    elif cfg.timestamp_col:
        text = raw[cfg.timestamp_col]
    else:
        raise ValueError("dataset.timestamp_col or dataset.timestamp_cols must be set")
    if cfg.timestamp_unit:
        return pd.to_datetime(pd.to_numeric(text, errors="coerce"), unit=cfg.timestamp_unit)
    if text.dtype == object or pd.api.types.is_string_dtype(text):
        text = text.astype(str).str.strip()
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
    return [c for c in selected if c not in excluded]


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


def _read_tables(cfg: DatasetConfig,
                 resolve: Callable[[str], Path]) -> tuple[pd.DataFrame, list[Path]]:
    if not cfg.paths:
        raise ValueError(f"dataset {cfg.name!r} requires dataset.paths")
    paths = [resolve(p) for p in cfg.paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"raw data file not found: {p}")
    tables = []
    for p in paths:
        kwargs = {"low_memory": False, **cfg.read_csv_kwargs}
        table = pd.read_csv(p, **kwargs)
        table.columns = [str(c).strip() for c in table.columns]
        tables.append(table)
    return pd.concat(tables, ignore_index=True), paths


def _from_table(raw: pd.DataFrame, cfg: DatasetConfig, source: str,
                paths: list[Path], resolve: Callable[[str], Path]) -> CPSDataset:
    if cfg.label_col is None and cfg.attack_intervals_path is None:
        raise ValueError("set dataset.label_col or dataset.attack_intervals_path")
    timestamps = _parse_timestamps(raw, cfg)
    columns: dict[str, Any] = {TIMESTAMP: timestamps.to_numpy()}
    if cfg.label_col is not None:
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
    for col in _select_features(raw, cfg, reserved):
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.notna().sum() == 0:  # no numeric value anywhere (text column)
            log.info("skipping non-numeric column %r", col)
            continue
        # Partial missingness is judged on the training split only (runner).
        columns[col] = values.to_numpy(dtype=np.float64)
        feature_cols.append(col)
    for col in [*cfg.categorical_cols, *cfg.count_distinct_cols]:
        if col in raw and col not in columns:
            columns[col] = raw[col].astype(str).str.strip().to_numpy()
    frame = pd.DataFrame(columns)
    fingerprint: dict[str, Any] = {"files": fingerprint_files(paths), "loader": source}
    if cfg.attack_intervals_path:
        fingerprint["attack_intervals"] = _apply_attack_intervals(
            frame, resolve(cfg.attack_intervals_path), cfg)
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
    label_col = cfg.label_col
    if label_col is None:
        label_col = next((c for c in raw.columns if c.lower().startswith("attack")), None)
    update: dict[str, Any] = {
        "timestamp_cols": cfg.timestamp_cols or ["Date", "Time"],
        "timestamp_format": cfg.timestamp_format or "%m/%d/%Y %I:%M:%S.%f %p",
        "label_col": label_col,
        "feature_exclude": [*cfg.feature_exclude, "Row"],
    }
    if label_col is not None and not cfg.positive_labels:
        update["positive_labels"] = [-1, "-1"]
    return cfg.model_copy(update=update)


def load_dataset(cfg: DatasetConfig, resolve: Callable[[str], Path], seed: int) -> CPSDataset:
    """Load and canonicalise the configured dataset.

    Args:
        cfg: dataset section of the experiment config.
        resolve: maps config paths to absolute paths (``ExperimentConfig.resolve_path``).
        seed: experiment seed (used by the synthetic generator only).
    """
    if cfg.name == "synthetic":
        return _load_synthetic(cfg, seed)
    raw, paths = _read_tables(cfg, resolve)
    if cfg.name == "swat":
        return _from_table(raw, _swat_defaults(cfg), "swat", paths, resolve)
    if cfg.name == "wadi":
        raw.columns = [_WADI_PREFIX.sub("", c) for c in raw.columns]
        return _from_table(raw, _wadi_defaults(cfg, raw), "wadi", paths, resolve)
    if cfg.name == "ton_iot":
        if not cfg.resample_seconds:
            raise ValueError("TON_IoT events are irregular: set dataset.resample_seconds")
        if cfg.event_duration_col is None and "duration" in raw.columns:
            cfg = cfg.model_copy(update={"event_duration_col": "duration"})
        return _from_table(raw, cfg, "ton_iot", paths, resolve)
    return _from_table(raw, cfg, "generic_csv", paths, resolve)
