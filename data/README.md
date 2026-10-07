# Data

Raw datasets go under `data/raw/` and are **never committed** (see `.gitignore`)
and **never modified**: loaders open them read-only, every run records their
SHA-256 in `data_fingerprint.json`, re-hashes them at the end
(`raw_data_integrity.json`) and fails if anything changed. Making the raw files
read-only (`chmod -R a-w data/raw`) is recommended.

| Dataset | Source (request access from the owner) | Expected path (default config) |
|---|---|---|
| SWaT (A1, Dec 2015) | iTrust, Singapore University of Technology and Design | `data/raw/SWaT/SWaT_Dataset_Attack_v0.csv` |
| WADI (2017/2019) | iTrust, Singapore University of Technology and Design | `data/raw/WADI/WADI_attackdataLABLE.csv` |
| TON_IoT (network) | UNSW Canberra Cyber | `data/raw/TON_IoT/network/train_test_network.csv` |
| Any CSV | your own telemetry | set `dataset.name: generic_csv` and the column options |

Notes per dataset:

* **SWaT** is distributed as `.xlsx`; export the attack sheet to CSV. The loader
  strips the padded column names, parses `dd/mm/YYYY hh:mm:ss AM/PM`, and maps both
  `Attack` and the known typo `A ttack` to the positive class. The normal-period
  file contains no attacks, so the supervised chronological split uses the attack
  period; run `cpscopilot audit --config configs/experiments/swat.yaml` first to
  check that every split contains attacks.
* **WADI** column names carry a long historian prefix
  (`\\WIN-...\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_001_PV`), which is removed. The
  2019 export's label column uses `1` = normal and `-1` = attack. For exports
  without labels, provide `attack_intervals_path` (CSV with `start,end[,attack_type]`).
* **TON_IoT** flow records are irregular events; they are aggregated into
  right-closed bins (`resample_seconds`) with sums/means/maxima, event counts and
  distinct counts of hosts/ports/services. Use the full chronological network
  CSVs for temporal experiments - the sampled `train_test_network.csv` is not
  contiguous in time.

Without any downloaded data, `dataset.name: synthetic` simulates a three-stage
water-treatment plant with labelled attack campaigns (see
`src/cpscopilot/data/synthetic.py`); `cpscopilot synth` writes it to CSV.
