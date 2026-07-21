from __future__ import annotations

from collections.abc import Callable, Hashable

import networkx as nx
import numpy as np


def cc_shapley_correction(
    phi: np.ndarray,
    dag: nx.DiGraph,
    x: np.ndarray,
    background: np.ndarray,
    detector_score_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: list[Hashable],
    n_samples: int = 200,
) -> np.ndarray:
    """Apply a collider-descendant correction to interventional Shapley values.

    This implementation is intentionally lightweight for Module 5a. It finds
    nodes with multiple parents, estimates how strongly each collider is
    coupled to its parents in the benign background, shrinks attribution on
    collider descendants, and redistributes the removed mass to collider
    parents. The correction preserves the total Shapley sum.
    """
    del detector_score_fn, n_samples  # reserved for later non-linear correction variants

    phi_arr = np.asarray(phi, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if phi_arr.ndim != 1:
        raise ValueError("phi must be a 1D array")
    if x_arr.shape != phi_arr.shape:
        raise ValueError("x and phi must have the same shape")
    if bg.ndim != 2 or bg.shape[1] != phi_arr.shape[0]:
        raise ValueError("background shape must be (N, len(phi))")
    if len(feature_names) != phi_arr.shape[0]:
        raise ValueError("feature_names length must match phi")

    g = _indexed_graph(dag, feature_names)
    colliders = [node for node in g.nodes if g.in_degree(node) >= 2]
    if not colliders:
        return phi_arr.copy()

    corrected = phi_arr.copy()
    original_total = float(np.sum(phi_arr))

    bg_mean = bg.mean(axis=0)
    bg_std = bg.std(axis=0) + 1e-12

    for collider in colliders:
        parents = sorted(g.predecessors(collider))
        descendants = sorted(nx.descendants(g, collider))
        if not descendants:
            continue

        coupling = _mean_abs_corr(bg, parents, collider)
        parent_idx = np.asarray(parents, dtype=int)
        deviation = np.mean(np.abs((x_arr[parent_idx] - bg_mean[parents]) / bg_std[parents]))
        strength = float(np.clip(0.05 + 0.20 * coupling + 0.03 * deviation, 0.05, 0.35))

        affected = np.asarray(descendants, dtype=int)
        removed = corrected[affected] * strength
        corrected[affected] -= removed

        parent_weights = np.abs(x_arr[parent_idx] - bg_mean[parents]) + bg_std[parents]
        parent_weights = parent_weights / parent_weights.sum()
        corrected[parent_idx] += float(np.sum(removed)) * parent_weights

    gap = original_total - float(np.sum(corrected))
    if gap != 0.0:
        corrected += gap / corrected.size
    return corrected


def _indexed_graph(dag: nx.DiGraph, feature_names: list[Hashable]) -> nx.DiGraph:
    node_to_idx = {node: i for i, node in enumerate(feature_names)}
    g = nx.DiGraph()
    g.add_nodes_from(range(len(feature_names)))
    for src, dst in dag.edges():
        if src in node_to_idx and dst in node_to_idx:
            g.add_edge(node_to_idx[src], node_to_idx[dst])
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("dag must be acyclic")
    return g


def _mean_abs_corr(background: np.ndarray, parents: list[int], collider: int) -> float:
    vals: list[float] = []
    c = background[:, collider]
    c_std = float(np.std(c))
    if c_std <= 1e-12:
        return 0.0
    for parent in parents:
        p = background[:, parent]
        if float(np.std(p)) <= 1e-12:
            continue
        corr = float(np.corrcoef(p, c)[0, 1])
        if np.isfinite(corr):
            vals.append(abs(corr))
    return float(np.mean(vals)) if vals else 0.0
