"""Does the causal graph improve attribution correctness? A ground-truth test.

The random-DAG ablation on counterfactual feasibility cannot settle whether
NF-DAG-v1 is a good causal model, because feasibility is defined relative to the
graph being tested. Attribution can settle it, because for a known structural
causal model the true causal Shapley values exist in closed form, so estimates
can be scored against ground truth rather than against each other.

Construction:
  * a linear SCM  x = W^T x + e  over d features with known DAG G_true;
  * a linear detector f(x) = b^T x, for which the interventional value function
    v(S) = E[f(x) | do(x_S = x*_S)] is exact -- intervened features are held and
    the rest propagated through the SCM in topological order;
  * exact Shapley values by enumerating all 2^d coalitions.

The same estimator (caushap_nids.xai_layers.causal_shapley) is then run with the
true graph, with progressively corrupted graphs, and with no graph at all
(vanilla KernelSHAP), and each result is scored against the exact values. If the
graph carries information, attribution error should rise as the graph degrades.

Usage:
    python scripts/attribution_recovery_benchmark.py --features 10 --flows 30
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from caushap_nids.xai_layers.causal_shapley import causal_shapley, vanilla_kernel_shap


class LinearDetector:
    """f(x) = b^T x. Linear so that ground-truth Shapley values are exact."""

    def __init__(self, b: np.ndarray) -> None:
        self.b = np.asarray(b, dtype=np.float64)

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return X @ self.b


def random_scm(d: int, edge_prob: float, rng: np.random.Generator):
    """Random upper-triangular (hence acyclic) linear SCM over d features."""
    W = np.zeros((d, d))
    order = rng.permutation(d)
    g = nx.DiGraph()
    g.add_nodes_from(range(d))
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < edge_prob:
                i, j = int(order[a]), int(order[b])
                w = rng.uniform(0.4, 1.2) * rng.choice([-1.0, 1.0])
                W[i, j] = w
                g.add_edge(i, j)
    return W, g


def propagate(x_star: np.ndarray, S: tuple[int, ...], W: np.ndarray,
              topo: list[int], parents: dict[int, list[int]]) -> np.ndarray:
    """E[x | do(x_S = x*_S)] under the linear SCM (noise has zero mean)."""
    m = np.zeros(len(x_star))
    S_set = set(S)
    for j in topo:
        if j in S_set:
            m[j] = x_star[j]
        else:
            pa = parents[j]
            m[j] = sum(W[i, j] * m[i] for i in pa) if pa else 0.0
    return m


def exact_causal_shapley(x_star: np.ndarray, detector: LinearDetector, W: np.ndarray,
                         g: nx.DiGraph) -> np.ndarray:
    """Exact interventional Shapley values by enumerating all coalitions."""
    d = len(x_star)
    topo = list(nx.topological_sort(g))
    parents = {j: list(g.predecessors(j)) for j in g.nodes()}

    v: dict[tuple[int, ...], float] = {}
    for r in range(d + 1):
        for S in combinations(range(d), r):
            v[S] = float(detector.b @ propagate(x_star, S, W, topo, parents))

    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for r in range(d):
            for S in combinations(others, r):
                w = 1.0 / (d * comb(d - 1, r))
                phi[i] += w * (v[tuple(sorted(S + (i,)))] - v[tuple(sorted(S))])
    return phi


def corrupt(g: nx.DiGraph, k: int, rng: np.random.Generator) -> nx.DiGraph:
    """Apply k random edge operations (delete / reverse / add), keeping acyclicity."""
    h = g.copy()
    for _ in range(k):
        for _try in range(60):
            op = rng.choice(["delete", "reverse", "add"])
            edges = list(h.edges())
            if op == "delete" and edges:
                h.remove_edge(*edges[rng.integers(len(edges))])
                break
            if op == "reverse" and edges:
                u, v_ = edges[rng.integers(len(edges))]
                h.remove_edge(u, v_)
                h.add_edge(v_, u)
                if nx.is_directed_acyclic_graph(h):
                    break
                h.remove_edge(v_, u)
                h.add_edge(u, v_)
                continue
            if op == "add":
                nodes = list(h.nodes())
                u, v_ = rng.choice(nodes, size=2, replace=False)
                if h.has_edge(int(u), int(v_)):
                    continue
                h.add_edge(int(u), int(v_))
                if nx.is_directed_acyclic_graph(h):
                    break
                h.remove_edge(int(u), int(v_))
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--flows", type=int, default=30)
    ap.add_argument("--background", type=int, default=400)
    ap.add_argument("--edge-prob", type=float, default=0.30)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--corruption-repeats", type=int, default=3,
                    help="independent corrupted graphs per corruption level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/attribution_recovery.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = args.features

    W, g_true = random_scm(d, args.edge_prob, rng)
    topo = list(nx.topological_sort(g_true))
    parents = {j: list(g_true.predecessors(j)) for j in g_true.nodes()}
    print(f"SCM: {d} features, {g_true.number_of_edges()} edges")

    # Background sample from the SCM.
    def draw(n: int) -> np.ndarray:
        X = np.zeros((n, d))
        for j in topo:
            pa = parents[j]
            base = sum(W[i, j] * X[:, i] for i in pa) if pa else 0.0
            X[:, j] = base + rng.normal(scale=1.0, size=n)
        return X

    background = draw(args.background)
    anchors = draw(args.flows)
    detector = LinearDetector(rng.normal(size=d))
    names = [f"f{i}" for i in range(d)]

    def to_named(h: nx.DiGraph) -> nx.DiGraph:
        return nx.relabel_nodes(h, {i: names[i] for i in h.nodes()})

    # Several independent corruptions per level: a single draw is too noisy,
    # because k random edge operations sometimes leave the interventional
    # distribution essentially intact.
    corruption_levels = [1, 2, 3, 5, 8]
    configs: dict[str, nx.DiGraph] = {"true_graph": g_true}
    for k in corruption_levels:
        for r in range(args.corruption_repeats):
            configs[f"corrupt_{k}_r{r}"] = corrupt(g_true, k, rng)

    truth = np.array([
        exact_causal_shapley(x, detector, W, g_true) for x in anchors
    ])

    results: dict[str, dict] = {}
    per_flow_err: dict[str, list[float]] = {}

    for label, h in configs.items():
        errs, rhos = [], []
        for x, phi_true in zip(anchors, truth, strict=True):
            est = causal_shapley(
                detector, to_named(h), x, background,
                n_samples=args.n_samples,
            ).phi
            errs.append(float(np.mean(np.abs(est - phi_true))))
            rhos.append(float(spearmanr(est, phi_true).statistic))
        per_flow_err[label] = errs
        results[label] = {
            "mae": float(np.mean(errs)),
            "spearman": float(np.nanmean(rhos)),
            "edges": h.number_of_edges(),
        }
        print(f"  {label:<12} MAE={results[label]['mae']:.4f}  "
              f"rho={results[label]['spearman']:.4f}  edges={h.number_of_edges()}")

    # No-graph baseline: vanilla KernelSHAP.
    errs, rhos = [], []
    for x, phi_true in zip(anchors, truth, strict=True):
        est = vanilla_kernel_shap(
            detector, x, background, n_samples=args.n_samples, feature_names=names
        ).phi
        errs.append(float(np.mean(np.abs(est - phi_true))))
        rhos.append(float(spearmanr(est, phi_true).statistic))
    per_flow_err["vanilla_shap"] = errs
    results["vanilla_shap"] = {
        "mae": float(np.mean(errs)), "spearman": float(np.nanmean(rhos)), "edges": 0,
    }
    print(f"  {'vanilla_shap':<12} MAE={results['vanilla_shap']['mae']:.4f}  "
          f"rho={results['vanilla_shap']['spearman']:.4f}")

    # Corruption levels, pooled over the independent corrupted graphs.
    print("\n" + "=" * 70)
    print("ATTRIBUTION ERROR vs GRAPH CORRUPTION (pooled over repeats)")
    print("=" * 70)
    base = np.array(per_flow_err["true_graph"])
    print(f"  {'graph':<16}{'MAE':>10}{'rho':>9}{'vs true':>11}{'p':>11}")
    print(f"  {'true graph':<16}{base.mean():>10.4f}"
          f"{results['true_graph']['spearman']:>9.4f}{'--':>11}{'--':>11}")

    level_summary: dict[str, dict] = {}
    for k in corruption_levels:
        labels = [f"corrupt_{k}_r{r}" for r in range(args.corruption_repeats)]
        pooled = np.concatenate([per_flow_err[x] for x in labels])
        rep = np.mean([np.array(per_flow_err[x]) for x in labels], axis=0)
        p = float(wilcoxon(rep, base).pvalue) if not np.allclose(rep - base, 0) else 1.0
        red = 100.0 * (pooled.mean() - base.mean()) / pooled.mean() if pooled.mean() else 0.0
        rho = float(np.mean([results[x]["spearman"] for x in labels]))
        level_summary[f"corrupt_{k}"] = {
            "mae": float(pooled.mean()), "spearman": rho,
            "pct_error_reduction_from_true_graph": red, "p_vs_true": p,
        }
        print(f"  {f'{k} edge ops':<16}{pooled.mean():>10.4f}{rho:>9.4f}"
              f"{red:>10.1f}%{p:>11.3g}")

    alt = np.array(per_flow_err["vanilla_shap"])
    p = float(wilcoxon(alt, base).pvalue) if not np.allclose(alt - base, 0) else 1.0
    red = 100.0 * (alt.mean() - base.mean()) / alt.mean() if alt.mean() else 0.0
    level_summary["vanilla_shap"] = {
        "mae": float(alt.mean()), "spearman": results["vanilla_shap"]["spearman"],
        "pct_error_reduction_from_true_graph": red, "p_vs_true": p,
    }
    print(f"  {'no graph (SHAP)':<16}{alt.mean():>10.4f}"
          f"{results['vanilla_shap']['spearman']:>9.4f}{red:>10.1f}%{p:>11.3g}")
    results["_summary"] = level_summary

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
