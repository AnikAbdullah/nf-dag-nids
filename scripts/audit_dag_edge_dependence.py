"""Audit whether NF-DAG-v1's edges carry real statistical dependence.

Every downstream claim rests on NF-DAG-v1 being a better description of NetFlow
than an arbitrary graph, yet the paper never measures that directly: its
stratified validation checks each edge against the evidence class that proposed
it, so protocol edges are confirmed by the protocol semantics they came from.

This measures each edge against the data instead.  For an edge u -> v, the
effect size is the partial correlation between u and v controlling for v's other
parents -- how much u explains about v that v's remaining parents do not.  The
same statistic is computed for size-matched random graphs, which gives the null
distribution the expert graph has to beat.

Results break down by provenance class, so a graph that is sound in part can be
pruned to that part rather than defended or discarded whole.

Usage:
    python scripts/audit_dag_edge_dependence.py --rows 400000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl

from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS
from caushap_nids.xai_layers.multi_obj_cf.plausibility import FeatureTransform

NAMES = list(NF_V2_FEATURE_COLS)


def _residual(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residual of y after regressing out the columns of Z (intercept included)."""
    if Z.size == 0:
        return y - y.mean()
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ beta


def partial_corr(X: np.ndarray, ix: dict[str, int], u: str, v: str,
                 controls: list[str]) -> float:
    """|partial correlation| between u and v given `controls`."""
    if u not in ix or v not in ix:
        return float("nan")
    ctrl = [c for c in controls if c in ix and c not in (u, v)]
    Z = X[:, [ix[c] for c in ctrl]] if ctrl else np.empty((len(X), 0))
    ru = _residual(X[:, ix[u]], Z)
    rv = _residual(X[:, ix[v]], Z)
    du, dv = np.std(ru), np.std(rv)
    if du < 1e-12 or dv < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(ru, rv)[0, 1]))


def edge_effect_sizes(dag: nx.DiGraph, X: np.ndarray, ix: dict[str, int]) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for u, v in dag.edges():
        others = [p for p in dag.predecessors(v) if p != u]
        out[(u, v)] = partial_corr(X, ix, u, v, others)
    return out


def random_dag_like(dag: nx.DiGraph, seed: int) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    nodes = list(dag.nodes())
    order = list(rng.permutation(len(nodes)))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    pairs = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    for c in rng.choice(len(pairs), size=dag.number_of_edges(), replace=False):
        i, j = pairs[c]
        g.add_edge(nodes[order[i]], nodes[order[j]])
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NF-CSE-CIC-IDS2018-V2.parquet")
    ap.add_argument("--dag", default="artifacts/nf_dag_v1.graphml")
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("artifacts/dag_edge_audit.json"))
    args = ap.parse_args()

    df = pl.scan_parquet(Path("data") / args.dataset).head(args.rows).collect()
    label_col = "Label" if "Label" in df.columns else "label"
    y = df[label_col].to_numpy()
    X_raw = df.select(NAMES).to_numpy().astype(np.float64)

    ft = FeatureTransform.fit(X_raw[y == 0])
    X = ft.transform(X_raw)[y == 0]          # benign only, matching how the DAG was built
    ix = {f: i for i, f in enumerate(NAMES)}
    print(f"benign rows: {len(X):,}\n")

    dag = nx.read_graphml(args.dag)
    real = edge_effect_sizes(dag, X, ix)
    types = {(u, v): d.get("edge_type", "?") for u, v, d in dag.edges(data=True)}

    # Null distribution from size-matched random graphs.
    null: list[float] = []
    for c in range(args.controls):
        null.extend(edge_effect_sizes(random_dag_like(dag, seed=1000 + c), X, ix).values())
    null_arr = np.array([v for v in null if not np.isnan(v)])
    real_arr = np.array([v for v in real.values() if not np.isnan(v)])

    print("=" * 72)
    print("EDGE DEPENDENCE: |partial correlation| given the child's other parents")
    print("=" * 72)
    print(f"{'set':<28}{'n':>6}{'median':>10}{'mean':>10}{'>0.1':>9}{'>0.3':>9}")
    print("-" * 72)

    def _row(label: str, a: np.ndarray) -> None:
        print(f"{label:<28}{len(a):>6}{np.median(a):>10.3f}{a.mean():>10.3f}"
              f"{np.mean(a > 0.1):>9.1%}{np.mean(a > 0.3):>9.1%}")

    _row("NF-DAG-v1 (all edges)", real_arr)
    _row(f"random graphs (n={args.controls})", null_arr)

    print()
    by_type: dict[str, list[float]] = defaultdict(list)
    for e, val in real.items():
        if not np.isnan(val):
            by_type[types[e]].append(val)
    for t in sorted(by_type):
        _row(f"  NF-DAG-v1 [{t}]", np.array(by_type[t]))

    # How many expert edges beat the random 95th percentile?
    cutoff = float(np.percentile(null_arr, 95))
    beats = float(np.mean(real_arr > cutoff))
    print(f"\nrandom 95th percentile: {cutoff:.3f}")
    print(f"NF-DAG-v1 edges above it: {beats:.1%} "
          f"({int(beats * len(real_arr))}/{len(real_arr)})  [5% expected by chance]")

    print("\nWeakest 10 edges in NF-DAG-v1 (candidates for pruning):")
    for (u, v), val in sorted(real.items(), key=lambda kv: kv[1])[:10]:
        print(f"  {val:.3f}  [{types[(u, v)]:<9}] {u} -> {v}")

    print("\nStrongest 10 edges:")
    for (u, v), val in sorted(real.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {val:.3f}  [{types[(u, v)]:<9}] {u} -> {v}")

    payload = {
        "real_edges": {f"{u}->{v}": val for (u, v), val in real.items()},
        "edge_types": {f"{u}->{v}": t for (u, v), t in types.items()},
        "null_median": float(np.median(null_arr)),
        "real_median": float(np.median(real_arr)),
        "random_p95": cutoff,
        "frac_real_above_random_p95": beats,
        "by_type_median": {t: float(np.median(v)) for t, v in by_type.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
