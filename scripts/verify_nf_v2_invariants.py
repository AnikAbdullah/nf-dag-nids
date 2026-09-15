"""Verify candidate NF-v2 physical invariants against real traffic.

Reports the per-invariant violation rate on each dataset.  Only invariants that
hold exactly (or near-exactly) should be cited as deterministic constraints or
used as the graph-independent feasibility oracle for the DAG ablation.

Usage:
    python scripts/verify_nf_v2_invariants.py [--rows N] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS
from caushap_nids.xai_layers.multi_obj_cf.plausibility import CANDIDATE_INVARIANTS, _idx

DATASETS = {
    "nf_cic2018": "NF-CSE-CIC-IDS2018-V2.parquet",
    "nf_unsw15": "NF-UNSW-NB15-v2.parquet",
    "edge_iiotset": "Edge-IIoTset.parquet",
    "5g_nidd": "5G-NIDD.parquet",
}


def verify(path: Path, n_rows: int) -> dict[str, float]:
    df = pl.read_parquet(path, columns=list(NF_V2_FEATURE_COLS)).head(n_rows)
    X = df.to_numpy().astype(np.float64)
    ix = _idx(list(NF_V2_FEATURE_COLS))

    rates: dict[str, float] = {}
    for inv in CANDIDATE_INVARIANTS:
        violations = sum(1 for row in X if not inv.check(row, ix))
        rates[inv.name] = violations / len(X)
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=500_000)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    results: dict[str, dict[str, float]] = {}
    for name, fname in DATASETS.items():
        path = args.data_dir / fname
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        print(f"[run ] {name} ({args.rows:,} rows)...")
        results[name] = verify(path, args.rows)

    if not results:
        print("No datasets found.")
        return

    cols = list(results)
    width = max(len(inv.name) for inv in CANDIDATE_INVARIANTS) + 2
    print(f"\n{'invariant':<{width}}" + "".join(f"{c:>16}" for c in cols))
    print("-" * (width + 16 * len(cols)))
    for inv in CANDIDATE_INVARIANTS:
        row = "".join(f"{results[c][inv.name]:>15.6%}" + " " for c in cols)
        print(f"{inv.name:<{width}}{row}")

    print("\nExact on every dataset checked (violation rate 0):")
    for inv in CANDIDATE_INVARIANTS:
        if all(results[c][inv.name] == 0.0 for c in cols):
            print(f"  - {inv.name}: {inv.description}")

    print("\nNOT exact — do not cite as deterministic:")
    for inv in CANDIDATE_INVARIANTS:
        worst = max(results[c][inv.name] for c in cols)
        if worst > 0.0:
            print(f"  - {inv.name}: worst violation rate {worst:.4%}")


if __name__ == "__main__":
    main()
