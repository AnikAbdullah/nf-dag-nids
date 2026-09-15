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
    """
    Fraction of DAG edges violated by the counterfactual.
    Violated = exactly one endpoint of a causal edge changes.  A changed child
    without its parent is unsupported; a changed parent without updating the
    child is also structurally inconsistent in the absence of an SEM update.
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
