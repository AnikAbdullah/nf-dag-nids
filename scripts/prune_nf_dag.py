"""Prune NF-DAG-v1 to the edges real traffic supports.

A plain effect-size threshold is not enough, and would go wrong in both
directions:

  * it would KEEP `RETRANSMITTED_IN_BYTES -> ICMP_TYPE` (|r| = 0.99), which is
    domain nonsense held up by structural zeros -- both fields are 0 on almost
    every flow, so they correlate without ever co-varying.
  * it would DROP `PROTOCOL -> ICMP_TYPE` (|r| = 0.04), which looks dead only
    because CIC2018 carries almost no ICMP, leaving the child near-constant.

So an edge is dropped only when it is weak *and* the data had a fair chance to
show it, and separately when its apparent strength survives only on the rows
where neither endpoint varies:

  drop-weak      |partial r| < `--min-r` on every dataset where BOTH endpoints
                 actually vary (std above `--min-std`).  Edges with no such
                 dataset are kept and reported as untested rather than judged.
  drop-spurious  |partial r| >= `--min-r` overall but collapses below it once
                 rows with both endpoints at zero are excluded.

Usage:
    python scripts/prune_nf_dag.py --out artifacts/nf_dag_v1_pruned.graphml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl

from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS
from caushap_nids.xai_layers.multi_obj_cf.plausibility import FeatureTransform

from audit_dag_edge_dependence import partial_corr  # noqa: E402

NAMES = list(NF_V2_FEATURE_COLS)
DATASETS = {
    "nf_cic2018": "NF-CSE-CIC-IDS2018-V2.parquet",
    "nf_unsw15": "NF-UNSW-NB15-v2.parquet",
}

#: Edges that encode a deterministic export invariant verified at a 0% violation
#: rate (see plausibility.VERIFIED_INVARIANTS).  These are true by construction
#: of the NetFlow format, and partial correlation is simply the wrong instrument
#: for them: an inequality constraint can be linearly weak among the rows where
#: both endpoints vary while still never being violated.  Never pruned.
INVARIANT_EDGES: frozenset[tuple[str, str]] = frozenset({
    ("FLOW_DURATION_MILLISECONDS", "DURATION_IN"),
    ("FLOW_DURATION_MILLISECONDS", "DURATION_OUT"),
})


def _load_benign(path: Path, rows: int) -> tuple[np.ndarray, np.ndarray]:
    df = pl.scan_parquet(path).head(rows).collect()
    label_col = "Label" if "Label" in df.columns else "label"
    y = df[label_col].to_numpy()
    X_raw = df.select(NAMES).to_numpy().astype(np.float64)
    ft = FeatureTransform.fit(X_raw[y == 0])
    return ft.transform(X_raw)[y == 0], X_raw[y == 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dag", default="artifacts/nf_dag_v1.graphml")
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--min-r", type=float, default=0.10)
    ap.add_argument("--min-std", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("artifacts/nf_dag_v1_pruned.graphml"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/dag_prune_report.json"))
    args = ap.parse_args()

    dag = nx.read_graphml(args.dag)
    ix = {f: i for i, f in enumerate(NAMES)}

    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fname in DATASETS.items():
        p = Path("data") / fname
        if p.exists():
            data[name] = _load_benign(p, args.rows)
            print(f"loaded {name}: {len(data[name][0]):,} benign rows")
    print()

    decisions: dict[str, dict] = {}
    for u, v, attrs in dag.edges(data=True):
        key = f"{u}->{v}"
        rec: dict = {"edge_type": attrs.get("edge_type", "?"), "per_dataset": {}}
        tested_anywhere = False
        strong_anywhere = False

        for ds, (X, X_raw) in data.items():
            if u not in ix or v not in ix:
                continue
            su, sv = float(np.std(X[:, ix[u]])), float(np.std(X[:, ix[v]]))
            varies = su > args.min_std and sv > args.min_std
            r = partial_corr(X, ix, u, v, [p for p in dag.predecessors(v) if p != u])
            rec["per_dataset"][ds] = {
                "partial_r": None if np.isnan(r) else round(float(r), 4),
                "both_vary": varies,
                "std_parent": round(su, 5),
                "std_child": round(sv, 5),
            }
            if varies:
                tested_anywhere = True
                if not np.isnan(r) and r >= args.min_r:
                    strong_anywhere = True

        # Structural-zero check on the datasets where the edge looked strong.
        # A real dependence must survive excluding the shared zeros EVERYWHERE it
        # looked strong; surviving on one dataset while collapsing on another is
        # the signature of a zero-driven artefact, not of a mechanism.
        spurious = False
        if strong_anywhere:
            survives_all, checked_any = True, False
            for ds, (X, X_raw) in data.items():
                info = rec["per_dataset"].get(ds, {})
                if not info.get("both_vary") or (info.get("partial_r") or 0) < args.min_r:
                    continue
                nz = (X_raw[:, ix[u]] != 0) | (X_raw[:, ix[v]] != 0)
                if nz.sum() < 1000:
                    continue
                r_nz = partial_corr(
                    X[nz], ix, u, v, [p for p in dag.predecessors(v) if p != u]
                )
                rec["per_dataset"][ds]["partial_r_nonzero_rows"] = (
                    None if np.isnan(r_nz) else round(float(r_nz), 4)
                )
                rec["per_dataset"][ds]["nonzero_rows"] = int(nz.sum())
                checked_any = True
                if np.isnan(r_nz) or r_nz < args.min_r:
                    survives_all = False
            spurious = checked_any and not survives_all

        if (u, v) in INVARIANT_EDGES:
            rec["decision"] = "keep"
            rec["reason"] = "deterministic export invariant (0% violations); not prunable by correlation"
        elif not tested_anywhere:
            rec["decision"], rec["reason"] = "keep", "untested: endpoints near-constant everywhere"
        elif spurious:
            rec["decision"], rec["reason"] = "drop", "structural-zero artefact"
        elif strong_anywhere:
            rec["decision"], rec["reason"] = "keep", "supported on at least one dataset"
        else:
            rec["decision"], rec["reason"] = "drop", f"|partial r| < {args.min_r} wherever tested"

        decisions[key] = rec

    kept = [k for k, r in decisions.items() if r["decision"] == "keep"]
    dropped = [k for k, r in decisions.items() if r["decision"] == "drop"]

    pruned = dag.copy()
    for key in dropped:
        u, v = key.split("->")
        pruned.remove_edge(u, v)

    print(f"kept {len(kept)} / {dag.number_of_edges()} edges; dropped {len(dropped)}\n")
    print("DROPPED:")
    for k in dropped:
        r = decisions[k]
        rs = {ds: d["partial_r"] for ds, d in r["per_dataset"].items()}
        print(f"  [{r['edge_type']:<9}] {k:<52} {r['reason']}  {rs}")

    print("\nKEPT but untested (endpoints near-constant in available data):")
    for k in kept:
        if "untested" in decisions[k]["reason"]:
            print(f"  [{decisions[k]['edge_type']:<9}] {k}")

    nx.write_graphml(pruned, args.out)
    args.report.write_text(json.dumps(
        {"kept": kept, "dropped": dropped, "decisions": decisions,
         "n_before": dag.number_of_edges(), "n_after": pruned.number_of_edges()},
        indent=2,
    ))
    print(f"\nwrote {args.out} ({pruned.number_of_edges()} edges) and {args.report}")


if __name__ == "__main__":
    main()
