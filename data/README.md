# Data

Raw datasets go under `data/raw/` and are **never committed** (see `.gitignore`)
and **never modified**: loaders open them read-only, every run records their
SHA-256 in `data_fingerprint.json`, re-hashes them at the end
(`raw_data_integrity.json`) and fails if anything changed. Making the raw files
read-only (`chmod -R a-w data/raw`) is recommended.

| Dataset | Source (request access from the owner) | Expected path (default config) |
|---|---|---|
| SWaT (A1 & A2, Dec 2015) | iTrust, Singapore University of Technology and Design | `data/raw/SWaT/SWaT_Dataset_Attack_v0.xlsx` (or a CSV export) |
| WADI (2019 release A2) | iTrust, Singapore University of Technology and Design | `data/raw/WADI/WADI_attackdataLABLE.csv` |
| TON_IoT (network, full processed files) | UNSW Canberra Cyber | `data/raw/TON_IoT/network/Network_dataset_1.csv` ... `_23.csv` |
| Any CSV | your own telemetry | set `dataset.name: generic_csv` and the column options |

`paths` entries may be glob patterns (`Network_dataset_*.csv`); matches are sorted
with numbers compared numerically. Run `cpscopilot audit --config <config>` (or
`python scripts/run_study.py --datasets <name> --audit-only`) before any experiment:
it prints the rows and attacks, the release check below and the chronological split
with the attack rows of each part. Windows users: see [`docs/WINDOWS.md`](../docs/WINDOWS.md).

**Release check.** For SWaT and WADI the loader compares the file with the
officially distributed one (rows, attack rows, time span) and warns when a copy
differs (edited, merged or truncated); the result is stored in
`data_fingerprint.json` (`release_check`).

Notes per dataset:

* **SWaT.** iTrust distributes the 2015 data as Excel workbooks. The official
  `SWaT_Dataset_Attack_v0.xlsx` is read directly (first row blank, header on row 2,
  timestamps stored as text such as ` 28/12/2015 10:00:00 AM`). Reading a 450 000-row
  workbook takes several minutes, so it is converted once to CSV under
  `data/interim/excel_cache/`, keyed by the workbook's SHA-256; the workbook itself
  is only read. The loader strips padded column names (` MV101`), parses
  `d/m/YYYY h:mm:ss AM/PM` day-first, and maps both `Attack` and the typo `A ttack`
  (37 rows) to the positive class. Official file: 449 919 rows, 54 621 attack rows,
  28/12/2015 10:00:00 AM to 2/1/2016 2:59:59 PM, one 81-second gap. The normal-period
  files contain no attacks, so the supervised chronological split uses the attack file.
  `python scripts/compare_raw_files.py --config configs/experiments/swat.yaml <xlsx> <csv>`
  checks that a CSV copy has the same timestamps, labels and values as the workbook.
* **WADI.** Two releases exist and both are supported.
  - 2019 (A2) `WADI_attackdataLABLE.csv`: a `0,1,...,130` index line precedes the
    header (`Row ,Date ,Time,...`), dates are `M/D/YY` and **Time is stored as
    `mm:ss.0` without the hour** in every row (this is the official file, not a
    damaged copy). Timestamps are therefore rebuilt from `Row` (1 Hz) as
    2017-10-09 18:00:00 + (Row - 1) s, and every row's date and minute:second are
    checked against the rebuilt time; the date changes at midnight confirm the hour.
    The two trailing rows made of commas are dropped. Labels: `1` normal, `-1`
    attack (9 977 rows). Known label issues of this release: segments start about
    3 s late and attack 13 is labelled at 2017-10-11 00:07:33-00:10:55 instead of
    12:07:30-12:10:52. Setting `attack_intervals_path` replaces the built-in labels by
    your interval file (CSV with `start,end[,attack_type]`).
  - 2017 (A1) `WADI_attackdata.csv`: historian tag prefixes
    (`\\WIN-...\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_001_PV`, removed), times such as
    `6:00:00.000 PM`, no label column: provide `attack_intervals_path`. Interval
    times are parsed with `attack_intervals_format` if set, else as ISO 8601
    (`2017-10-09 19:25:00`), else with the dataset's `dayfirst`. Every interval must
    match at least one row, otherwise loading fails.
  - The attacks fall on 9 Oct evening, 10 Oct 10:24-17:49 and 11 Oct 10:55-15:37,
    so `wadi.yaml` splits at fixed times (train to 10 Oct 13:00, calibration to
    11 Oct 06:00): a 60/20/20 fraction split would leave almost no attack in the
    calibration part.
* **TON_IoT.** Use the full processed network files `Network_dataset_1.csv` ...
  `Network_dataset_23.csv` (folder `Processed_datasets/Processed_Network_dataset`,
  22.3 million flow records, 3.4 GB). The `Train_Test_datasets` sample
  `train_test_network.csv` has **no time column** (its rows are grouped by attack
  type) and is rejected with an explanation. Flow records are irregular events: each
  is placed at the time it becomes observable (`ts + duration`, `event_duration_col`)
  and aggregated into right-closed bins (`resample_seconds`) with sums/means/maxima,
  event counts and distinct counts of hosts/ports/services. Only the configured
  columns are read and text columns are kept as categories to limit memory; text
  tokens in numeric fields (e.g. `0.0.0.0` in `src_bytes`) are counted and treated
  as missing. Gaps longer than `max_gap_factor` bins are outages: their empty bins
  are dropped and a new segment starts. Shorter silent stretches are kept; with
  `record_kind: event` (the TON_IoT default) a silent bin is labelled an attack only
  when attack bins precede and follow it, and attack intervals are matched to record
  start times. Caveats for experiments: the capture covers about 6.5 days of traffic
  spread over 2-29 April 2019, about 96 % of the flows are attacks, and each attack
  type occurs in a single time window, so a chronological split evaluates attack
  types that were not seen in training.

Without any downloaded data, `dataset.name: synthetic` simulates a three-stage
water-treatment plant with labelled attack campaigns (see
`src/cpscopilot/data/synthetic.py`); `cpscopilot synth` writes it to CSV.
