"""
CLI entry point for caushap-run.

Usage
-----
# Single run:
caushap-run --config A4_full --dataset nf_cic2018 --seeds 42

# All 72 ablation runs:
caushap-run --all

# Specific subset:
caushap-run --config A4_full A3_moocf --dataset nf_cic2018 nf_unsw15 --seeds 42 43

# Fast smoke-test (200 explained flows per run):
caushap-run --config A4_full --dataset nf_cic2018 --seeds 42 --n-explain 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from caushap_nids.experiments.runner import (
    CONFIG_NAMES,
    DATASETS,
    DEFAULT_SEEDS,
    run_ablation_matrix,
    run_single,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caushap-run",
        description="Run caushap_nids ablation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",    nargs="+", choices=list(CONFIG_NAMES), default=None,
                   help="Ablation config(s) to run. Defaults to all 6.")
    p.add_argument("--dataset",   nargs="+", choices=list(DATASETS), default=None,
                   help="Dataset key(s). Defaults to all 4.")
    p.add_argument("--seeds",     nargs="+", type=int, default=None,
                   help="Random seeds. Defaults to [42, 43, 44].")
    p.add_argument("--all",       action="store_true",
                   help="Run complete 72-cell ablation matrix (overrides --config/--dataset/--seeds).")
    p.add_argument("--data-dir",  default="data",       help="Path to data directory.")
    p.add_argument("--artifacts", default="artifacts",  help="Path to artifacts directory.")
    p.add_argument("--configs",   default="configs",    help="Path to configs directory.")
    p.add_argument("--n-explain", type=int, default=200,
                   help="Max anomalous flows to explain per run (default 200; use 0 for all).")
    p.add_argument("--quiet",     action="store_true",  help="Suppress per-step output.")
    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    configs  = tuple(CONFIG_NAMES) if args.all else tuple(args.config or CONFIG_NAMES)
    datasets = tuple(DATASETS)     if args.all else tuple(args.dataset or DATASETS)
    seeds    = tuple(DEFAULT_SEEDS) if args.all else tuple(args.seeds or DEFAULT_SEEDS)
    n_explain = args.n_explain

    data_dir     = Path(args.data_dir)
    artifact_dir = Path(args.artifacts)
    configs_dir  = Path(args.configs)

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 2

    verbose = not args.quiet

    if len(configs) == 1 and len(datasets) == 1 and len(seeds) == 1:
        result = run_single(
            configs[0], datasets[0], seeds[0],
            data_dir=data_dir,
            artifact_dir=artifact_dir,
            configs_dir=configs_dir,
            n_explain=n_explain,
            verbose=verbose,
        )
        if result.status != "ok":
            print(f"FAILED: {result.error}", file=sys.stderr)
            return 1
        _print_summary([result])
    else:
        results = run_ablation_matrix(
            configs=configs,
            datasets=datasets,
            seeds=seeds,
            data_dir=data_dir,
            artifact_dir=artifact_dir,
            configs_dir=configs_dir,
            n_explain=n_explain,
            verbose=verbose,
        )
        _print_summary(results)
        failed = [r for r in results if r.status != "ok"]
        if failed:
            print(f"\n{len(failed)} run(s) failed.", file=sys.stderr)
            return 1

    return 0


def _print_summary(results: list) -> None:
    ok  = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status != "ok")
    print(f"\n{'═'*60}")
    print(f"  Runs: {len(results)}  OK: {ok}  Failed: {err}")
    if results:
        f1s = []
        for r in results:
            if r.status == "ok" and "macro_f1" in r.detection:
                f1s.append(r.detection["macro_f1"][0])
        if f1s:
            import numpy as np
            print(f"  Macro-F1  mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    sys.exit(main())
