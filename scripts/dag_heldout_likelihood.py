"""Model-selection comparison of NF-DAG-v1 against random graphs on real traffic.

The edge-level test in `audit_dag_edge_dependence.py` scores edges one at a time.
This scores the graph as a whole, using the standard criterion for comparing
directed acyclic graphs: the held-out predictive log-likelihood of a linear
Gaussian structural equation model, fitted on one split of benign traffic and
evaluated on another.

Under that model each node is Gaussian given its parents, so the graph's
log-likelihood decomposes over nodes and a graph that encodes real conditional
structure predicts unseen flows better than one that does not. Held-out
evaluation makes the comparison fair between graphs with the same edge count
without needing a complexity penalty, and every graph compared here has exactly
the edge count of NF-DAG-v1.

Usage:
    python scripts/dag_heldout_likelihood.py --rows 400000 --controls 20
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

NAMES = list(NF_V2_FEATURE_COLS)
_RIDGE = 1e-3


def heldout_loglik(g: nx.DiGraph, X_tr: np.ndarray, X_te: np.ndarray,
                   ix: dict[str, int]) -> tuple[float, int]:
    """Total held-out Gaussian log-likelihood of the SEM implied by ``g``.

    Returns (mean per-row log-likelihood, number of fitted parameters).
    """
    total = np.zeros(len(X_te))
    n_params = 0

    for node in g.nodes():
        if node not in ix:
            continue
        j = ix[node]
        pa = [p for p in g.predecessors(node) if p in ix]
        n_params += len(pa) + 1

        y_tr, y_te = X_tr[:, j], X_te[:, j]
        if pa:
            A_tr = X_tr[:, [ix[p] for p in pa]]
            A_te = X_te[:, [ix[p] for p in pa]]
            A1 = np.column_stack([np.ones(len(A_tr)), A_tr])
            gram = A1.T @ A1 + _RIDGE * np.eye(A1.shape[1])
            w = np.linalg.solve(gram, A1.T @ y_tr)
            pred_tr = A1 @ w
            pred_te = np.column_stack([np.ones(len(A_te)), A_te]) @ w
        else:
            w = np.array([y_tr.mean()])
            pred_tr = np.full(len(y_tr), w[0])
            pred_te = np.full(len(y_te), w[0])

        sigma2 = max(float(np.mean((y_tr - pred_tr) ** 2)), 1e-12)
        total += -0.5 * (np.log(2 * np.pi * sigma2) + (y_te - pred_te) ** 2 / sigma2)

    return float(np.mean(total)), n_params


def random_dag_like(g: nx.DiGraph, seed: int) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes())
    order = list(rng.permutation(len(nodes)))
    h = nx.DiGraph()
    h.add_nodes_from(nodes)
    pairs = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    for c in rng.choice(len(pairs), size=g.number_of_edges(), replace=False):
        i, j = pairs[c]
        h.add_edge(nodes[order[i]], nodes[order[j]])
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NF-CSE-CIC-IDS2018-V2.parquet")
    ap.add_argument("--dag", default="artifacts/nf_dag_v1.graphml")
    ap.add_argument("--pruned-dag", default="artifacts/nf_dag_v1_pruned.graphml")
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/dag_heldout_likelihood.json"))
    args = ap.parse_args()

    df = pl.scan_parquet(Path("data") / args.dataset).head(args.rows).collect()
    label_col = "Label" if "Label" in df.columns else "label"
    y = df[label_col].to_numpy()
    X_raw = df.select(NAMES).to_numpy().astype(np.float64)

    ft = FeatureTransform.fit(X_raw[y == 0])
    X = ft.transform(X_raw)[y == 0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    cut = len(X) // 2
    X_tr, X_te = X[perm[:cut]], X[perm[cut:]]
    ix = {f: i for i, f in enumerate(NAMES)}
    print(f"benign rows: train {len(X_tr):,}  held-out {len(X_te):,}\n")

    dag = nx.read_graphml(args.dag)
    graphs: dict[str, nx.DiGraph] = {"NF-DAG-v1": dag}
    pruned_path = Path(args.pruned_dag)
    if pruned_path.exists():
        graphs["NF-DAG-v1 (pruned)"] = nx.read_graphml(pruned_path)
    empty = nx.DiGraph()
    empty.add_nodes_from(dag.nodes())
    graphs["no edges (independent)"] = empty

    results: dict[str, dict] = {}
    for name, g in graphs.items():
        ll, k = heldout_loglik(g, X_tr, X_te, ix)
        results[name] = {"heldout_loglik": ll, "edges": g.number_of_edges(), "params": k}
        print(f"  {name:<26} LL/row = {ll:9.4f}   edges={g.number_of_edges():3d}")

    ctrl = []
    for c in range(args.controls):
        h = random_dag_like(dag, seed=2000 + c)
        ll, _ = heldout_loglik(h, X_tr, X_te, ix)
        ctrl.append(ll)
    ctrl_arr = np.array(ctrl)
    results["random_controls"] = {
        "n": len(ctrl), "mean": float(ctrl_arr.mean()), "sd": float(ctrl_arr.std()),
        "best": float(ctrl_arr.max()), "worst": float(ctrl_arr.min()),
    }
    print(f"  {'random graphs (same |E|)':<26} LL/row = {ctrl_arr.mean():9.4f}"
          f"   sd={ctrl_arr.std():.4f}  best={ctrl_arr.max():.4f}")

    real = results["NF-DAG-v1"]["heldout_loglik"]
    z = (real - ctrl_arr.mean()) / ctrl_arr.std() if ctrl_arr.std() > 0 else float("inf")
    beat = int((ctrl_arr >= real).sum())

    print("\n" + "=" * 68)
    print(f"NF-DAG-v1 held-out LL/row       {real:.4f}")
    print(f"random-graph mean               {ctrl_arr.mean():.4f}  (sd {ctrl_arr.std():.4f})")
    print(f"advantage                       {real - ctrl_arr.mean():+.4f} nats/flow")
    print(f"z-score vs random distribution  {z:+.1f}")
    print(f"random graphs beating NF-DAG-v1 {beat}/{len(ctrl)}")
    print(f"vs no-edge baseline             "
          f"{real - results['no edges (independent)']['heldout_loglik']:+.4f} nats/flow")

    results["summary"] = {
        "advantage_nats_per_flow": real - float(ctrl_arr.mean()),
        "z_score": float(z),
        "n_random_beating_real": beat,
        "vs_empty": real - results["no edges (independent)"]["heldout_loglik"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
