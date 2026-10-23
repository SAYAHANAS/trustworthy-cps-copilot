#!/usr/bin/env python3
"""Run the whole study: data audit, every seed, aggregation and figures, per dataset.

A cross-platform replacement for the Makefile targets (it works the same in
Windows PowerShell)::

    python scripts/run_study.py --datasets synthetic swat wadi
    python scripts/run_study.py --datasets wadi --seeds 0 1 2 --backend anthropic
    python scripts/run_study.py --datasets swat wadi --audit-only

For each dataset the script

1. audits the data (loading, release check, chronological split, attacks in
   every split) and skips the dataset if the audit fails; synthetic data, which
   depends on the seed, is audited for every seed;
2. runs one experiment per seed. A finished run is reused only if it has the
   same configuration hash and seed, the same raw-data files (SHA-256), the
   same knowledge base and the same framework code, and if its files still
   match their manifest; so an interrupted study resumes where it stopped and
   a changed input is always recomputed. A failing seed is reported and the
   other seeds still run;
3. aggregates the finished seeds (mean, s.d., 95 % t-interval) and renders the
   figures.

Nothing is overwritten: runs, aggregates and figures go to new timestamped
directories. ``--set key=value`` applies to every dataset. Ctrl+C stops the
whole study (finished runs are kept and reused next time).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpscopilot.cli import audit_report, write_aggregate  # noqa: E402
from cpscopilot.config import ExperimentConfig, config_hash, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "synthetic": "configs/experiments/synthetic_full.yaml",
    "fault_injection": "configs/experiments/synthetic_fault_injection.yaml",
    "swat": "configs/experiments/swat.yaml",
    "wadi": "configs/experiments/wadi.yaml",
    "ton_iot": "configs/experiments/ton_iot.yaml",
}


class StudyInterrupted(Exception):
    """Ctrl+C: stop everything (XGBoost may re-raise the interrupt as its own error)."""


_INTERRUPTED = False


def _on_interrupt(signum: int, frame: Any) -> None:
    global _INTERRUPTED
    _INTERRUPTED = True
    raise KeyboardInterrupt


def _config_path(name: str) -> Path:
    key = name.lower()
    if key in CONFIGS:
        return ROOT / CONFIGS[key]
    path = Path(name)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"unknown dataset {name!r}: use one of {', '.join(CONFIGS)} "
                                "or the path of a config file")
    return path


def _check_llm_credentials(cfg: ExperimentConfig) -> None:
    """Fail before any training when the Claude backend cannot authenticate."""
    if not cfg.copilot.enabled or cfg.copilot.llm.backend != "anthropic":
        return
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("the anthropic backend needs the anthropic package: "
                           'pip install -e ".[llm]"') from exc
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "copilot.llm.backend=anthropic but ANTHROPIC_API_KEY is not set in this terminal. "
            'PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."   bash: export '
            "ANTHROPIC_API_KEY=sk-ant-...   (or pass --skip-key-check if you authenticate "
            "another way)")


def _summarise_audit(report: dict[str, Any]) -> str:
    ds = report["dataset"]
    lines = [f"  rows after resampling: {ds['n_rows']:,}  features: {ds['n_features']}  "
             f"attack rows: {ds['attack_rows']:,} ({ds['attack_prevalence']:.1%}), "
             f"{ds['attack_episodes']} episodes, {ds['start']} -> {ds['end']}"]
    release = report["data_checks"].get("release_check")
    if release:
        if release.get("release"):
            verdict = "identical counts" if release["matches_release"] else "; ".join(
                release["differences"])
            lines.append(f"  official release: {release['release']}: {verdict}")
        else:
            lines.append(f"  official release: no match ({release.get('note', '')})")
    split = report["split_audit"]
    if "error" in split:
        lines.append(f"  split: FAILED - {split['error']}")
    else:
        for name, s in split["splits"].items():
            lines.append(f"  {name:<12} {s['start_ts']} -> {s['end_ts']}  rows {s['n_rows']:,}"
                         f"  attack rows {s['n_positive']:,}")
        lines.append(f"  split checks passed: {split['passed']}")
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(cfg: ExperimentConfig, data_files: list[dict[str, Any]]) -> dict[str, Any]:
    """What a reusable run must share with the run we would start now."""
    from cpscopilot.copilot.knowledge_base import kb_fingerprint
    from cpscopilot.utils.io import code_fingerprint

    kb = None
    if cfg.copilot.enabled:
        kb = kb_fingerprint(cfg.resolve_path(cfg.copilot.knowledge_base_dir))
    return {"config_hash": config_hash(cfg), "data": sorted(f["sha256"] for f in data_files),
            "kb": kb, "code": code_fingerprint()}


def _finished_runs(cfg: ExperimentConfig, identity: dict[str, Any]) -> dict[int, Path]:
    """Verified run directories with the same configuration, data, KB and code, by seed."""
    from cpscopilot.utils.run_manager import verify_run

    root = cfg.resolve_path(cfg.output_root)
    found: dict[int, Path] = {}
    if not root.exists():
        return found
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        try:
            metrics = _read_json(metrics_path)
            if metrics.get("config_hash") != identity["config_hash"]:
                continue
            files = _read_json(run_dir / "data_fingerprint.json")["fingerprint"].get("files", [])
            kb_path = run_dir / "knowledge_base.json"
            kb = _read_json(kb_path)["files"] if kb_path.exists() else None
            code = _read_json(run_dir / "environment.json").get("code_sha256")
            same = (sorted(f["sha256"] for f in files) == identity["data"]
                    and kb == identity["kb"] and code == identity["code"])
            if same and verify_run(run_dir)["ok"]:
                found[int(metrics["seed"])] = run_dir
        except (OSError, ValueError, KeyError, TypeError) as exc:  # damaged run: ignore it
            print(f"  ignoring {run_dir.name}: {type(exc).__name__}: {exc}")
    return found


def run_dataset(name: str, seeds: list[int], overrides: list[str], audit_only: bool,
                figures: bool, key_check: bool) -> dict[str, Any]:
    from cpscopilot.pipeline.runner import ExperimentRunner
    from cpscopilot.plotting.figures import generate_all

    path = _config_path(name)
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"\n=== {name}  ({shown})")
    cfg = load_config(path, overrides)
    if key_check and not audit_only:
        _check_llm_credentials(cfg)
    per_seed_audit = cfg.dataset.name == "synthetic"  # the generated data depend on the seed
    report: dict[str, Any] = {}
    if not per_seed_audit:
        t0 = time.time()
        report = audit_report(cfg)
        print(_summarise_audit(report) + f"\n  audit took {time.time() - t0:.0f} s")
        if not report["split_audit"]["passed"]:
            return {"dataset": name, "status": "audit failed", "runs": []}
    if audit_only and not per_seed_audit:
        return {"dataset": name, "status": "audit passed", "runs": []}
    reusable: dict[int, Path] | None = None
    run_dirs: list[Path] = []
    failed: list[int] = []
    for seed in seeds:
        seed_cfg = load_config(path, [*overrides, f"seed={seed}"])
        try:
            if per_seed_audit:
                report = audit_report(seed_cfg)
                print(f"  seed {seed} data:\n" + _summarise_audit(report))
                if not report["split_audit"]["passed"]:
                    failed.append(seed)
                    continue
                if audit_only:
                    continue
            if reusable is None:  # the code, data and KB digests are the same for every seed
                reusable = _finished_runs(cfg, _identity(cfg, report["data_files"]))
            if seed in reusable:
                run_dir = reusable[seed]
                print(f"  seed {seed}: reusing {run_dir.name}")
            else:
                t0 = time.time()
                run_dir = ExperimentRunner(seed_cfg).run().run_dir
                print(f"  seed {seed}: {run_dir.name} ({(time.time() - t0) / 60:.1f} min)")
            run_dirs.append(run_dir)
        except KeyboardInterrupt as exc:
            raise StudyInterrupted from exc
        except Exception:
            if _INTERRUPTED:
                raise StudyInterrupted from None
            traceback.print_exc()
            print(f"  seed {seed}: FAILED (see the error above); continuing with the others")
            failed.append(seed)
    if audit_only:
        status = "audit passed" if not failed else f"audit failed for seeds {failed}"
        return {"dataset": name, "status": status, "runs": []}
    status = "done" if not failed else f"done except seeds {failed}"
    if not run_dirs:
        return {"dataset": name, "status": f"failed (seeds {failed})", "runs": []}
    out: dict[str, Any] = {"dataset": name, "status": status,
                           "runs": [str(r) for r in run_dirs]}
    if len(run_dirs) > 1:
        out["aggregate"] = str(write_aggregate(run_dirs, cfg.name))
        print(f"  aggregate: {out['aggregate']}")
    if figures:
        out["figures"] = str(generate_all(run_dirs, formats=["pdf", "svg"], latex="auto"))
        print(f"  figures: {out['figures']}")
    det = [_read_json(r / "metrics.json")["models"]["detection"]["temperature"]
           for r in run_dirs]
    pr_auc = [round(d["classification"]["pr_auc"], 4) for d in det]
    ece = [round(d["calibration"]["ece"], 4) for d in det]
    print(f"  detection PR-AUC per seed: {pr_auc}  ECE (temperature-scaled): {ece}")
    return out


def main(argv: list[str] | None = None) -> int:
    import signal

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "swat", "wadi"],
                        help=f"names ({', '.join(CONFIGS)}) or config paths")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--backend", choices=["template", "anthropic"], default=None,
                        help="copilot LLM backend (default: the config's, i.e. template)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override applied to every dataset (repeatable)")
    parser.add_argument("--audit-only", action="store_true", help="check the data, run nothing")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-key-check", action="store_true",
                        help="do not require ANTHROPIC_API_KEY for the anthropic backend")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error(f"--seeds repeats a seed: {args.seeds}")
    overrides = list(args.set)
    if args.backend:
        overrides.append(f"copilot.llm.backend={args.backend}")
    previous = signal.signal(signal.SIGINT, _on_interrupt)
    results = []
    try:
        for name in args.datasets:
            try:
                results.append(run_dataset(name, args.seeds, overrides, args.audit_only,
                                           not args.no_figures, not args.skip_key_check))
            except (KeyboardInterrupt, StudyInterrupted):
                raise
            except Exception as exc:  # one dataset failing must not stop the others
                if _INTERRUPTED:
                    raise StudyInterrupted from exc
                traceback.print_exc()
                results.append({"dataset": name, "status": f"error: {exc}", "runs": []})
    except (KeyboardInterrupt, StudyInterrupted):
        print("\ninterrupted: finished runs are kept and will be reused by the same command")
        results.append({"dataset": "-", "status": "interrupted", "runs": []})
        code = 130
    else:
        ok = ("done", "audit passed")
        code = 0 if all(r["status"] in ok for r in results) else 1
    finally:
        signal.signal(signal.SIGINT, previous)
    print("\n=== summary")
    for r in results:
        print(f"  {r['dataset']:<16} {r['status']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
