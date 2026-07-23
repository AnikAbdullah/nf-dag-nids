#!/usr/bin/env python3
"""Parallel, memory-aware driver for the 72-run ablation matrix on Apple Silicon.

Why this exists
---------------
On an M4 the AE is a tiny 39-dim net and the XAI layers (causal Shapley, NSGA-II
counterfactuals, concept abduction) issue *hundreds of thousands* of single-flow
``detector.score`` calls. On MPS each tiny call pays kernel-launch latency and runs
**5-12x slower than CPU** (measured). So this driver:

  • forces **CPU** (``CAUSHAP_DEVICE=cpu``) — faster here, and frees us to run
    several dataset streams in parallel with no GPU contention;
  • runs one subprocess per **(dataset, seed)** work-unit. Each unit invokes
    ``caushap-run`` for all 6 configs of that unit, so the data loads once and the
    AE+IF detector trains once and is reused across configs (same as
    ``run_ablation_matrix``);
  • caps each unit to a few threads so N units don't oversubscribe the 10 cores;
  • lets at most one heavy ``nf_cic2018`` unit run at a time (16 GB RAM guard);
  • is **resume-safe** — the underlying runner skips any (config,dataset,seed)
    whose ``artifacts/results/.../result.json`` already matches. Re-run any time.

Usage (from the project root, venv active)
------------------------------------------
    python scripts/run_matrix_mac.py                 # full 72-run matrix, 4 parallel units
    python scripts/run_matrix_mac.py --jobs 5        # more parallelism (watch RAM)
    python scripts/run_matrix_mac.py --datasets edge_iiotset 5g_nidd nf_unsw15
    python scripts/run_matrix_mac.py --n-explain 200 # default; matches the paper matrix

Per-unit logs land in ``artifacts/logs/matrix_<dataset>_<seed>.log``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("nf_cic2018", "nf_unsw15", "edge_iiotset", "5g_nidd")
SEEDS = (42, 43, 44)
HEAVY = {"nf_cic2018"}  # ~17M rows — serialize these among themselves (RAM guard)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4,
                    help="Max concurrent (dataset,seed) units (default 4; 16 GB-safe).")
    ap.add_argument("--n-explain", type=int, default=200,
                    help="Anomalous flows to explain per cell (default 200, paper matrix).")
    ap.add_argument("--threads-per-job", type=int, default=2,
                    help="Thread cap per unit so units don't oversubscribe cores (default 2).")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--fresh", action="store_true",
                    help="Delete artifacts/results/ first so all cells recompute with the "
                         "current frozen code (use for the authoritative paper run; the "
                         "detector cache in artifacts/models/ is kept). Omit to resume.")
    args = ap.parse_args()

    if args.fresh:
        import shutil
        results_dir = ROOT / "artifacts" / "results"
        if results_dir.exists():
            # Some result.json on disk predate the 2026-05-17 module freezes but still carry
            # criteria-v7/n_explain=200, so the runner's checkpoint would silently reuse them
            # and mix pre-freeze XAI numbers into the ablation. --fresh removes them so every
            # cell is recomputed; the (frozen, deterministic) detector cache is left intact.
            n = sum(1 for _ in results_dir.rglob("result.json"))
            shutil.rmtree(results_dir)
            print(f"--fresh: cleared artifacts/results/ ({n} old result.json removed; "
                  f"detector cache kept).")

    log_dir = ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Heavy units first so the long pole (nf_cic2018) overlaps with the small ones
    # instead of forming a serial tail; the mutex still keeps only one heavy unit live.
    units = [(d, s) for d in args.datasets for s in args.seeds]
    units.sort(key=lambda u: (u[0] not in HEAVY, u[0], u[1]))

    env = dict(os.environ)
    env.setdefault("CAUSHAP_DEVICE", "cpu")
    t = str(args.threads_per_job)
    for k in ("CAUSHAP_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = t

    pending = list(units)
    running: list[tuple[subprocess.Popen, tuple[str, int], object]] = []
    total = len(units)
    done = failed = 0
    t0 = time.perf_counter()

    def heavy_live() -> bool:
        return any(u[0] in HEAVY for _, u, _ in running)

    def launch(unit: tuple[str, int]) -> None:
        d, s = unit
        log = log_dir / f"matrix_{d}_{s}.log"
        fh = open(log, "w")
        cmd = [sys.executable, "-m", "caushap_nids.experiments.cli",
               "--dataset", d, "--seeds", str(s), "--n-explain", str(args.n_explain)]
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
        running.append((p, unit, fh))
        print(f"[{time.strftime('%H:%M:%S')}] ▶ launched {d:13s} seed={s}  "
              f"(log: {log.relative_to(ROOT)})", flush=True)

    print(f"Matrix driver: {total} units (dataset×seed) × 6 configs = "
          f"{total*6} cells | jobs={args.jobs} | device={env['CAUSHAP_DEVICE']} | "
          f"threads/unit={t} | n_explain={args.n_explain}\n", flush=True)

    while pending or running:
        # Fill the pool with the first launchable unit (skip a heavy one if another is live).
        while pending and len(running) < args.jobs:
            pick = None
            for i, u in enumerate(pending):
                if u[0] in HEAVY and heavy_live():
                    continue
                pick = i
                break
            if pick is None:
                break  # only heavy units left and one is already running
            launch(pending.pop(pick))

        time.sleep(2)

        still = []
        for p, u, fh in running:
            if p.poll() is None:
                still.append((p, u, fh))
                continue
            fh.close()
            done += 1
            ok = p.returncode == 0
            failed += 0 if ok else 1
            tag = "OK" if ok else f"FAIL(rc={p.returncode})"
            mins = (time.perf_counter() - t0) / 60
            print(f"[{time.strftime('%H:%M:%S')}] ■ {u[0]:13s} seed={u[1]} → {tag}  "
                  f"[{done}/{total} units, {mins:.1f} min elapsed]", flush=True)
        running = still

    mins = (time.perf_counter() - t0) / 60
    print(f"\n{'═'*64}")
    print(f"  Done: {done}/{total} units in {mins:.1f} min  |  failed units: {failed}")
    print(f"  Results: artifacts/results/   Per-unit logs: artifacts/logs/")
    print(f"  Next: open notebooks/08_results_tables.ipynb (no GPU) → results_tables/")
    print(f"{'═'*64}")
    if failed:
        print("  ⚠ Some units reported a non-zero exit — check their logs, then "
              "re-run this script (it resumes from result.json).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
