from __future__ import annotations

import networkx as nx
import numpy as np


def validity(x_cf: np.ndarray, detector, threshold: float) -> float:
    """1.0 if detector score < threshold (benign), else 0.0. NSGA-II minimises: return 1 - validity."""
    score = float(detector.score(np.asarray(x_cf, dtype=np.float64).reshape(1, -1))[0])
    return 1.0 if score < threshold else 0.0


def proximity(x_orig: np.ndarray, x_cf: np.ndarray) -> float:
    """L1 distance. Objective: minimise."""
    return float(np.abs(np.asarray(x_orig) - np.asarray(x_cf)).sum())


def sparsity(x_orig: np.ndarray, x_cf: np.ndarray, eps: float = 1e-6) -> float:
    """Number of features changed. Objective: minimise."""
    return float((np.abs(np.asarray(x_orig) - np.asarray(x_cf)) > eps).sum())


def feasibility(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    dag: nx.DiGraph,
    feature_names: list[str],
) -> float:
    """Fraction of DAG edges violated by the counterfactual (co-change criterion).

    Violated = exactly one endpoint of a causal edge changes.

    DEPRECATED for new work; retained to reproduce previously published runs.
    The criterion is degenerate: it is satisfied both when nothing changes and
    when every feature changes (all edges then have both endpoints changed), and
    is worst in between -- on the 41-node NF-DAG-v1 it scores 1.000 at 0/41 and
    41/41 changed features and 0.512 at 20/41. A search maximising it is
    therefore driven towards changing every feature, which is the opposite of
    the minimal recourse a counterfactual is meant to describe.

    It is also unsuitable for comparing two candidate graphs, since each is
    scored against itself: a random graph connects features with no reason to
    co-move and so is intrinsically harder to satisfy, regardless of whether it
    is causally correct.

    Use ``structural.structural_violation`` instead, and
    ``plausibility.plausibility_rate`` for graph-independent comparison.

    0.0 = fully feasible; 1.0 = all edges violated. Objective: minimise.
    """
    n_edges = dag.number_of_edges()
    if n_edges == 0:
        return 0.0

    x_orig = np.asarray(x_orig)
    x_cf = np.asarray(x_cf)
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    violations = 0
    for parent, child in dag.edges():
        pi = feat_idx.get(parent)
        ci = feat_idx.get(child)
        if pi is None or ci is None:
            continue
        child_changed = abs(x_cf[ci] - x_orig[ci]) > 1e-6
        parent_changed = abs(x_cf[pi] - x_orig[pi]) > 1e-6
        if child_changed != parent_changed:
            violations += 1

    return violations / n_edges
