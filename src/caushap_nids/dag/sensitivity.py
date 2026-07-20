from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from ..models.interface import AnomalyDetector

logger = logging.getLogger(__name__)


@dataclass
class Perturbation:
    name: str
    description: str


PERTURBATIONS: list[Perturbation] = [
    Perturbation("flip_5",     "Reverse direction of 5 randomly selected uncertain (U-type) edges"),
    Perturbation("remove_5",   "Remove 5 randomly selected edges"),
    Perturbation("pc_replace", "Replace entire DAG with PC algorithm output"),
]


def sensitivity_rank_correlation(
    dag: nx.DiGraph,
    perturbations: list[Perturbation],
    detector: "AnomalyDetector",
    X_test: np.ndarray,
    feature_names: list[str],
    n_test_flows: int = 500,
    seed: int = 42,
    shapley_fn: Callable | None = None,
    pc_dag: nx.DiGraph | None = None,
    df_benign=None,
) -> dict[str, float]:
    """Spearman rho of feature importance rankings under each perturbation.

    For each perturbation:
    1. Apply to produce dag_perturbed.
    2. Compute feature importances under dag (original) and dag_perturbed.
    3. Compute Spearman rho of feature importance rankings.

    shapley_fn: callable(dag, detector, X, feature_names) → np.ndarray shape (N, D)
                If None, falls back to mean reconstruction error per feature as a proxy
                (Module 5a not yet implemented).
    pc_dag: optional real PC output for the "pc_replace" perturbation.
    df_benign: optional benign data used to compute PC if pc_dag is not passed.

    Target: all rho values >= 0.8.
    Returns: {perturbation_name: spearman_rho}
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    rng_idx = np.random.choice(len(X_test), min(n_test_flows, len(X_test)), replace=False)
    X_sample = X_test[rng_idx]

    if shapley_fn is None:
        shapley_fn = _reconstruction_importance_proxy

    phi_orig = shapley_fn(dag, detector, X_sample, feature_names)
    rank_orig = _mean_rank(phi_orig)

    results: dict[str, float] = {}
    for pert in perturbations:
        dag_pert = _apply_perturbation(
            dag,
            pert,
            rng,
            pc_dag=pc_dag,
            df_benign=df_benign,
            feature_names=feature_names,
        )
        phi_pert = shapley_fn(dag_pert, detector, X_sample, feature_names)
        rank_pert = _mean_rank(phi_pert)
        rho, _ = spearmanr(rank_orig, rank_pert)
        results[pert.name] = float(rho)
        logger.info("Perturbation '%s': Spearman rho = %.4f", pert.name, rho)

    return results


def _apply_perturbation(
    dag: nx.DiGraph,
    pert: Perturbation,
    rng: random.Random,
    pc_dag: nx.DiGraph | None = None,
    df_benign=None,
    feature_names: list[str] | None = None,
) -> nx.DiGraph:
    dag_copy = dag.copy()
    edges = list(dag_copy.edges())

    if pert.name == "flip_5":
        # Restrict to U-type (uncertain) edges: P/M edges are evidence-backed and
        # epistemically wrong to flip; U edges are the appropriate robustness target.
        def _etype(s, d):
            et = dag_copy[s][d].get("edge_type")
            return et.name if hasattr(et, "name") else str(et)

        u_edges = [(s, d) for s, d in edges if _etype(s, d) == "U"]
        flip_pool = u_edges if u_edges else edges
        chosen = rng.sample(flip_pool, min(5, len(flip_pool)))
        for src, dst in chosen:
            attrs = dag_copy[src][dst].copy()
            dag_copy.remove_edge(src, dst)
            # Only add reversed edge if it doesn't create a cycle
            if not nx.has_path(dag_copy, src, dst):
                dag_copy.add_edge(dst, src, **attrs)

    elif pert.name == "remove_5":
        chosen = rng.sample(edges, min(5, len(edges)))
        dag_copy.remove_edges_from(chosen)

    elif pert.name == "pc_replace":
        if pc_dag is not None:
            dag_copy = _align_replacement_dag(pc_dag, dag.nodes())
        elif df_benign is not None:
            from .data_driven import run_pc

            dag_copy = _align_replacement_dag(
                run_pc(df_benign, feature_cols=feature_names),
                dag.nodes(),
            )
        else:
            logger.warning(
                "pc_replace requested without pc_dag or df_benign; returning an "
                "empty graph placeholder. This is diagnostic only, not a paper gate."
            )
            dag_copy = nx.DiGraph()
            dag_copy.add_nodes_from(dag.nodes())

    return dag_copy


def _align_replacement_dag(
    replacement: nx.DiGraph,
    required_nodes,
) -> nx.DiGraph:
    """Return a DAG replacement over the same node set as the reference DAG."""
    out = nx.DiGraph()
    out.add_nodes_from(required_nodes)
    required = set(required_nodes)
    for src, dst, attrs in replacement.edges(data=True):
        if src in required and dst in required and src != dst:
            out.add_edge(src, dst, **attrs)

    while not nx.is_directed_acyclic_graph(out):
        cycle = next(nx.simple_cycles(out))
        out.remove_edge(cycle[-1], cycle[0])
    return out


def _reconstruction_importance_proxy(
    dag: nx.DiGraph,
    detector: "AnomalyDetector",
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Proxy feature importance: mean per-feature reconstruction error.

    Used when Module 5a (causal_shapley) is not yet available.
    Shape: (N, D).
    """
    if hasattr(detector, "per_feature_error"):
        return detector.per_feature_error(X)
    scores = detector.score(X)
    return np.outer(scores, np.ones(len(feature_names)))


def _mean_rank(phi: np.ndarray) -> np.ndarray:
    """Mean absolute importance per feature, ranked. Shape: (D,)."""
    mean_importance = np.abs(phi).mean(axis=0)
    return mean_importance.argsort().argsort().astype(float)
