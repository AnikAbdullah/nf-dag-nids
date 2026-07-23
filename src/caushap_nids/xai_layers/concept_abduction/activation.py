from __future__ import annotations

import numpy as np
from scipy.special import expit   # numerically stable sigmoid


def compute_background_stats(background: np.ndarray) -> dict:
    """
    Compute mean and std of background (benign) data for z-score normalisation.
    Returns {"mean": ndarray, "std": ndarray}; std is clipped to 1e-9 to avoid /0.
    """
    mean = background.mean(axis=0)
    std  = background.std(axis=0).clip(1e-9)
    return {"mean": mean, "std": std}


def concept_activation_score(
    x: np.ndarray,
    concept: "ConceptSubgraph",
    background_stats: dict,
    feature_names: list[str],
) -> float:
    """
    Weighted aggregate of directed z-scores for the concept's core features.

    For each core feature:
      z = direction × clip((x[i] - mean[i]) / std[i], -5, 5)

    The mean z-score is then passed through a sigmoid to produce a value in [0, 1].
    A score > calibrated threshold indicates the concept is 'activated'.
    """
    x = np.asarray(x, dtype=np.float64)
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    mean: np.ndarray = np.asarray(background_stats["mean"], dtype=np.float64)
    std:  np.ndarray = np.asarray(background_stats["std"],  dtype=np.float64).clip(1e-9)

    directed_zs: list[float] = []
    weights: list[float] = []
    concept_weights = getattr(concept, "weights", [1.0] * len(concept.core_features))
    for feat, direction, weight in zip(concept.core_features, concept.directions, concept_weights):
        if feat not in feat_idx:
            continue
        idx = feat_idx[feat]
        z = (float(x[idx]) - float(mean[idx])) / float(std[idx])
        directed_zs.append(direction * float(np.clip(z, -5.0, 5.0)))
        weights.append(float(weight))

    if not directed_zs:
        return 0.0
    return float(expit(float(np.average(directed_zs, weights=weights))))


def calibrate_thresholds(
    concept_library: dict[str, "ConceptSubgraph"],
    background: np.ndarray,
    feature_names: list[str],
    background_stats: dict,
    *,
    target_fpr: float = 0.05,
) -> dict[str, float]:
    """
    Calibrate per-concept activation thresholds on benign (background) data.

    For each concept, scores all background rows and returns the
    (1 - target_fpr) percentile as the threshold. At most target_fpr fraction
    of benign flows will exceed this threshold.
    """
    thresholds: dict[str, float] = {}
    for name, concept in concept_library.items():
        scores = np.array([
            concept_activation_score(x, concept, background_stats, feature_names)
            for x in background
        ])
        thresholds[name] = float(np.percentile(scores, (1.0 - target_fpr) * 100.0))
    return thresholds
