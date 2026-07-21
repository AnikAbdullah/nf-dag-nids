from __future__ import annotations

import numpy as np
import networkx as nx


def dag_respecting_mutation(
    x: np.ndarray,
    dag: nx.DiGraph,
    feature_names: list[str],
    mutation_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    NSGA-II mutation operator that respects DAG structure.
    When a feature is selected for mutation, nearby ancestors and descendants
    are also perturbed so the resulting individual better preserves causal
    parent-child consistency.
    Returns a mutated copy of x.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = x.copy()
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    scale = max(float(np.std(x)), 0.1)

    for feat, idx in feat_idx.items():
        if rng.random() < mutation_rate:
            x[idx] += rng.normal(scale=0.1 * scale)
            for ancestor in nx.ancestors(dag, feat):
                ai = feat_idx.get(ancestor)
                if ai is not None:
                    x[ai] += rng.normal(scale=0.05 * scale)
            for descendant in nx.descendants(dag, feat):
                di = feat_idx.get(descendant)
                if di is not None:
                    x[di] += rng.normal(scale=0.05 * scale)

    return x


def check_feasibility(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    dag: nx.DiGraph,
    feature_names: list[str],
) -> dict[str, bool]:
    """
    Return {edge_str: is_feasible} for every DAG edge.
    An edge parent->child is feasible if both endpoints are unchanged or both
    endpoints are changed.  Changing only one endpoint is structurally
    inconsistent under the DAG.
    """
    x_orig = np.asarray(x_orig)
    x_cf = np.asarray(x_cf)
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    result: dict[str, bool] = {}
    for parent, child in dag.edges():
        edge_str = f"{parent}->{child}"
        pi = feat_idx.get(parent)
        ci = feat_idx.get(child)
        if pi is None or ci is None:
            result[edge_str] = True
            continue
        child_changed = abs(x_cf[ci] - x_orig[ci]) > 1e-6
        parent_changed = abs(x_cf[pi] - x_orig[pi]) > 1e-6
        result[edge_str] = child_changed == parent_changed

    return result
