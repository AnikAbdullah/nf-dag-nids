from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from caushap_nids.xai_layers.concept_abduction.concepts import ConceptSubgraph
    from caushap_nids.xai_layers.concept_abduction.abduce import AbductiveExplanation


def batch_concept_fidelity(
    detector,
    X: np.ndarray,
    explanations: list[AbductiveExplanation],
    concept_library: dict[str, ConceptSubgraph],
    feature_names: list[str],
    background_stats: dict,
    top_k: int = 2,
) -> np.ndarray:
    """
    Concept fidelity for each flow in X.
    Returns shape (N,) array of per-flow fidelity scores (target >= 0.8).
    """
    from caushap_nids.xai_layers.concept_abduction.fidelity import concept_fidelity
    return np.array([
        concept_fidelity(detector, X[i], explanations[i], concept_library, feature_names, background_stats, top_k=top_k)
        for i in range(len(X))
    ])


def batch_concept_rank_stability(
    X: np.ndarray,
    concept_library: dict[str, ConceptSubgraph],
    background_stats: dict,
    feature_names: list[str],
    dag: nx.DiGraph,
    n_perturbations: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Concept rank stability (Spearman rho) for each flow in X.
    Returns shape (N,) array of per-flow rho values (target >= 0.8).
    """
    from caushap_nids.xai_layers.concept_abduction.fidelity import concept_rank_stability
    return np.array([
        concept_rank_stability(X[i], concept_library, background_stats, feature_names, dag,
                               n_perturbations=n_perturbations, seed=seed)
        for i in range(len(X))
    ])
