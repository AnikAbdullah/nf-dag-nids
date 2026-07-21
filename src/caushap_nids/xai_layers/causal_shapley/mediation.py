from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import numpy as np


def decompose_effects(
    phi: np.ndarray,
    dag: nx.DiGraph,
    feature_names: list[Hashable],
) -> tuple[np.ndarray, np.ndarray]:
    """Split total attribution into direct and descendant-mediated effects.

    The decomposition is conservative: each node keeps its own direct Shapley
    value, while a bounded share of descendant attribution is assigned back to
    ancestors along shortest causal paths. Total attribution is conserved.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.ndim != 1:
        raise ValueError("phi must be a 1D array")
    if len(feature_names) != phi_arr.shape[0]:
        raise ValueError("feature_names length must match phi")

    g = _subgraph_in_feature_order(dag, feature_names)
    direct = phi_arr.copy()
    indirect = np.zeros_like(phi_arr)

    for target in range(len(feature_names)):
        ancestors = sorted(nx.ancestors(g, target))
        if not ancestors:
            continue

        mediated_fraction = min(0.5, len(ancestors) / (2.0 * (len(ancestors) + 1.0)))
        mediated = direct[target] * mediated_fraction
        if mediated == 0.0:
            continue

        weights = np.array([1.0 / nx.shortest_path_length(g, anc, target) for anc in ancestors])
        weights = weights / weights.sum()
        direct[target] -= mediated
        indirect[np.asarray(ancestors, dtype=int)] += mediated * weights

    return direct, indirect


def _subgraph_in_feature_order(dag: nx.DiGraph, feature_names: list[Hashable]) -> nx.DiGraph:
    node_to_idx = {node: i for i, node in enumerate(feature_names)}
    g = nx.DiGraph()
    g.add_nodes_from(range(len(feature_names)))
    for src, dst in dag.edges():
        if src in node_to_idx and dst in node_to_idx:
            g.add_edge(node_to_idx[src], node_to_idx[dst])
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("dag must be acyclic")
    return g
