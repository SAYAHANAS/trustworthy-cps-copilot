"""Command-line interface.

Examples::

    cpscopilot run --config configs/experiments/synthetic_smoke.yaml
    cpscopilot run --config configs/experiments/swat.yaml --seeds 0 1 2 3 4
    cpscopilot run --config configs/experiments/synthetic_full.yaml \\
        --set copilot.llm.backend=anthropic
    cpscopilot audit --config configs/experiments/swat.yaml
    cpscopilot aggregate runs/2026...s0_xxx runs/2026...s1_xxx
    cpscopilot figures runs/2026...s0_xxx --formats pdf svg
    cpscopilot verify runs/2026...s0_xxx
    cpscopilot synth --config configs/experiments/synthetic_full.yaml --out data/raw/synth.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cpscopilot.config import load_config

__all__ = ["main"]


def _cmd_run(args: argparse.Namespace) -> int:
    from cpscopilot.pipeline.aggregate import aggregate_runs
    from cpscopilot.pipeline.runner import ExperimentRunner
    from cpscopilot.utils.io import write_json_exclusive
    from cpscopilot.utils.run_manager import slugify, utc_stamp

    overrides = list(args.set or [])
    seeds = args.seeds or [None]
    run_dirs = []
    for seed in seeds:
        extra = [f"seed={seed}"] if seed is not None else []
        cfg = load_config(args.config, overrides + extra)
        output_root = Path(args.output_root) if args.output_root else None
        result = ExperimentRunner(cfg, output_root=output_root).run()
        run_dirs.append(result.run_dir)
        print(result.run_dir)
    if len(run_dirs) > 1:
        summary, table = aggregate_runs(run_dirs)
        out = run_dirs[0].parent / "aggregates" / f"{utc_stamp()}_{slugify(cfg.name)}"
        out.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(out / "summary.json", summary)
        with (out / "metrics.csv").open("x", encoding="utf-8") as fh:
            table.to_csv(fh, index=False)
        print(f"aggregate: {out}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Data loading, splitting and leakage audit only (no training)."""
    from cpscopilot.data.features import CausalWindowFeaturizer
    from cpscopilot.data.labels import horizon_to_steps
    from cpscopilot.data.loaders import load_dataset
    from cpscopilot.data.splitting import TemporalSplitter
    from cpscopilot.models.prediction import horizon_key

    cfg = load_config(args.config, args.set)
    ds = load_dataset(cfg.dataset, cfg.resolve_path, cfg.seed)
    featurizer = CausalWindowFeaturizer(cfg.features)
    steps = {horizon_key(h): horizon_to_steps(h, ds.sampling_seconds)
             for h in cfg.labels.horizons_minutes}
    embargo = cfg.split.embargo_steps if cfg.split.embargo_steps is not None else max(
        featurizer.warmup_steps + 1, max(steps.values()))
    split = TemporalSplitter(cfg.split, embargo).split(ds.frame)
    report = {"dataset": ds.summary(), "horizon_steps": steps,
              "split_audit": split.audit(ds.sampling_seconds)}
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["split_audit"]["passed"] else 1


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from cpscopilot.pipeline.aggregate import aggregate_runs

    summary, table = aggregate_runs([Path(r) for r in args.runs], allow_mixed=args.allow_mixed,
                                    allow_duplicates=args.allow_duplicates)
    print(json.dumps(summary, indent=2))
    if args.out:
        with Path(args.out).open("x", encoding="utf-8") as fh:
            table.to_csv(fh, index=False)
    else:
        print(table.to_string(index=False))
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from cpscopilot.plotting.figures import generate_all

    out = generate_all([Path(r) for r in args.runs], Path(args.out) if args.out else None,
                       formats=args.formats, latex=args.latex,
                       allow_duplicates=args.allow_duplicates)
    print(out)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from cpscopilot.utils.run_manager import verify_run

    report = verify_run(Path(args.run))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def _cmd_synth(args: argparse.Namespace) -> int:
    from cpscopilot.data.synthetic import generate_synthetic_cps

    cfg = load_config(args.config, args.set)
    frame, _ = generate_synthetic_cps(cfg.dataset.synthetic, cfg.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:  # never overwrite raw data
        frame.to_csv(fh, index=False)
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpscopilot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="experiment YAML")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="dotted override, repeatable (e.g. detection.ensemble_size=3)")

    p = sub.add_parser("run", help="run an experiment (optionally over several seeds)")
    config_args(p)
    p.add_argument("--seeds", type=int, nargs="+", help="run once per seed and aggregate")
    p.add_argument("--output-root", help="override output_root")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("audit", help="load + split + leakage checks only")
    config_args(p)
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("aggregate", help="mean/std/CI of metrics across runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", help="CSV path (must not exist)")
    p.add_argument("--allow-mixed", action="store_true")
    p.add_argument("--allow-duplicates", action="store_true",
                   help="accept repeated seeds within one configuration (replicas are not "
                        "independent samples)")
    p.set_defaults(func=_cmd_aggregate)

    p = sub.add_parser("figures", help="publication figures from run directories")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", help="new output directory (default figures/<stamp>_<hash>)")
    p.add_argument("--formats", nargs="+", default=["pdf", "svg"],
                   choices=["pdf", "svg", "pgf", "png", "eps"])
    p.add_argument("--latex", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--allow-duplicates", action="store_true",
                   help="accept repeated seeds within one configuration (replicas are not "
                        "independent samples)")
    p.set_defaults(func=_cmd_figures)

    p = sub.add_parser("verify", help="re-hash a run directory against its manifest")
    p.add_argument("run")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("synth", help="materialise the synthetic dataset as CSV")
    config_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_synth)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
