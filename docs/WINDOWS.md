# Running the study on Windows (PowerShell)

Everything works from PowerShell with Python 3.11-3.13; `make` is not needed.

## 1. Install

```powershell
cd C:\path\to\trustworthy-cps-copilot
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-lock.txt
pip install -e ".[dev,llm]"         # installs the `cpscopilot` command
python -m pytest                    # all tests should pass
```

## 2. Place the raw data (never open the CSV files in Excel and save them)

| Dataset | Put this file / these files | Notes |
|---|---|---|
| SWaT | `data\raw\SWaT\SWaT_Dataset_Attack_v0.xlsx` | the official workbook is read directly; the first run converts it once (about 5-10 min) into `data\interim\excel_cache\` |
| WADI | `data\raw\WADI\WADI_attackdataLABLE.csv` | the 2019 release file, as downloaded |
| TON_IoT | `data\raw\TON_IoT\network\Network_dataset_1.csv` ... `Network_dataset_23.csv` | the full processed network files (3.4 GB); `train_test_network.csv` has no time column and cannot be used |

Make the raw files read-only (the runs check that they never change):

```powershell
attrib +R data\raw\* /S
```

A CSV copy of SWaT works too; run SWaT on its own then, since `--set` applies to
every dataset of the command:
`python scripts\run_study.py --datasets swat --set "dataset.paths=[data/raw/SWaT/merged.csv]"`.
To check that it is a faithful copy of the official workbook (same timestamps, labels
and values):

```powershell
python scripts\compare_raw_files.py --config configs/experiments/swat.yaml data\raw\SWaT\SWaT_Dataset_Attack_v0.xlsx data\raw\SWaT\merged.csv
```

## 3. Check each dataset (fast, no training)

```powershell
python scripts\run_study.py --datasets swat wadi --audit-only
```

For every dataset this prints the number of rows and attacks, whether the file
matches the official release, and the chronological train / calibration / test
split with the attack rows in each part. Every part must contain attacks.

## 4. Run the study

```powershell
# synthetic plant + SWaT + WADI, 5 seeds each, offline template copilot
python scripts\run_study.py --datasets synthetic swat wadi --seeds 0 1 2 3 4

# the hallucination-audit sensitivity experiment
python scripts\run_study.py --datasets fault_injection --seeds 42

# the same with Claude as copilot (responses are cached in llm_cache\)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python scripts\run_study.py --datasets swat wadi --backend anthropic
```

The script audits the data, runs each seed, aggregates the seeds and renders the
figures. An interrupted study can be restarted with the same command: finished
runs are recognised and reused. Results go to `runs\` (one directory per run),
`runs\aggregates\` (mean, s.d. and 95 % interval over seeds) and `figures\`.

Single commands, if you prefer:

```powershell
cpscopilot audit --config configs/experiments/wadi.yaml
cpscopilot run --config configs/experiments/wadi.yaml --seeds 0 1 2 3 4
cpscopilot verify runs\<run_id>
python scripts\generate_paper_figures.py runs\<run_1> runs\<run_2> --formats pdf svg
```

## PowerShell equivalents

| Linux / Makefile | PowerShell |
|---|---|
| `make test` | `python -m pytest` |
| `make smoke` | `cpscopilot run --config configs/experiments/synthetic_smoke.yaml` |
| `make synthetic-seeds` | `python scripts\run_study.py --datasets synthetic` |
| `make fault-injection` | `python scripts\run_study.py --datasets fault_injection --seeds 42` |
| `chmod -R a-w data/raw` | `attrib +R data\raw\* /S` |
| `cmd1 && cmd2` | `cmd1; if ($?) { cmd2 }` (Windows PowerShell 5) or `cmd1 && cmd2` (PowerShell 7) |
| `export KEY=value` | `$env:KEY = "value"` |
| `--set "a.b=x y"` | same; keep the double quotes around values with spaces |

## Expected run times

The cost grows with the number of 10 s time bins and of input columns: WADI has
about 17 000 bins (124 columns), SWaT about 45 000 (51 columns) and the synthetic
plant about 121 000. Most of the time goes into training the XGBoost ensembles.
As a reference, one WADI seed took about 23 minutes on a shared 4-core cloud
machine; expect roughly 10-30 minutes per seed for WADI and SWaT on a laptop, more
for the synthetic plant, plus the one-off SWaT Excel conversion (about 6 minutes).
Start with one seed (`--seeds 0`) to see the time on your machine; `--set
runtime.n_jobs=8` uses more cores if you have them. The runs log each stage with
its duration.
