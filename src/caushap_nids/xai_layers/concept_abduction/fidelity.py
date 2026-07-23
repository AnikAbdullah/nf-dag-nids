from __future__ import annotations

import numpy as np


def _concept_score(detector, X: np.ndarray) -> np.ndarray:
    """Score used for concept ablations.

    Detectors can expose `concept_score` when their classification operating
    score is a calibrated subset of the underlying anomaly model.  Otherwise we
    fall back to the normal detector score.
    """
    if hasattr(detector, "concept_score"):
        return np.asarray(detector.concept_score(X), dtype=np.float64)
    return np.asarray(detector.score(X), dtype=np.float64)


def concept_fidelity(
    detector,
    x: np.ndarray,
    explanation: "AbductiveExplanation",
    concept_library: dict[str, "ConceptSubgraph"],
    feature_names: list[str],
    background_stats: dict,
    *,
    top_k: int = 2,
) -> float:
    """
    Measure how much masking the top-k concept core features changes the detector score.

    Algorithm:
      1. For each present concept (up to top_k), find the single most-activated
         core feature and replace it with the background mean.
      2. Re-score the masked flow.
      3. fidelity = 1 - |score_orig - score_masked| / score_orig

    A fidelity >= 0.8 means the AE anomaly signal is distributed across many
    features — masking a handful of concept features does not collapse the score.
    This is expected for a deep AE that learns holistic reconstruction patterns.
    Target: >= 0.8.
    """
    x = np.asarray(x, dtype=np.float64)
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    mean = np.asarray(background_stats["mean"], dtype=np.float64)
    std  = np.asarray(background_stats["std"],  dtype=np.float64).clip(1e-9)

    score_orig = float(_concept_score(detector, x.reshape(1, -1))[0])
    if score_orig < 1e-9:
        return 1.0

    # Pick the single highest directed-z feature per present concept
    features_to_mask: list[str] = []
    for name in explanation.present_concepts[:top_k]:
        concept = concept_library[name]
        best_z, best_feat = -np.inf, None
        concept_weights = getattr(concept, "weights", [1.0] * len(concept.core_features))
        for feat, direction, weight in zip(concept.core_features, concept.directions, concept_weights):
            if feat not in feat_idx:
                continue
            idx = feat_idx[feat]
            z = direction * (float(x[idx]) - float(mean[idx])) / float(std[idx])
            z *= float(weight)
            if z > best_z:
                best_z, best_feat = z, feat
        if best_feat is not None and best_feat not in features_to_mask:
            features_to_mask.append(best_feat)

    if not features_to_mask:
        return 1.0

    x_masked = x.copy()
    for feat in features_to_mask:
        x_masked[feat_idx[feat]] = float(mean[feat_idx[feat]])

    score_masked = float(_concept_score(detector, x_masked.reshape(1, -1))[0])
    impact = abs(score_orig - score_masked) / score_orig
    return float(np.clip(1.0 - impact, 0.0, 1.0))


def concept_rank_stability(
    x: np.ndarray,
    concept_library: dict[str, "ConceptSubgraph"],
    background_stats: dict,
    feature_names: list[str],
    dag: "nx.DiGraph",
    *,
    n_perturbations: int = 20,
    edge_drop_frac: float = 0.10,
    seed: int = 42,
) -> float:
    """
    Spearman ρ of concept activation rankings under random DAG edge-drop perturbations.
    Target: >= 0.8.  Higher = concept rankings are stable to minor DAG variations.
    """
    import networkx as nx
    from scipy.stats import spearmanr
    from .activation import concept_activation_score
    from .concepts import build_concept_subgraphs

    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)

    baseline_scores = np.array([
        concept_activation_score(x, c, background_stats, feature_names)
        for c in concept_library.values()
    ])

    rhos: list[float] = []
    edges = list(dag.edges())
    n_drop = max(1, int(len(edges) * edge_drop_frac))

    for _ in range(n_perturbations):
        drop_idx = rng.choice(len(edges), size=n_drop, replace=False)
        perturbed_dag = dag.copy()
        for i in drop_idx:
            u, v = edges[i]
            if perturbed_dag.has_edge(u, v):
                perturbed_dag.remove_edge(u, v)
        perturbed_library = build_concept_subgraphs(perturbed_dag)
        perturbed_scores = np.array([
            concept_activation_score(x, c, background_stats, feature_names)
            for c in perturbed_library.values()
        ])
        rho, _ = spearmanr(baseline_scores, perturbed_scores)
        rhos.append(float(rho) if not np.isnan(rho) else 1.0)

    return float(np.mean(rhos))
