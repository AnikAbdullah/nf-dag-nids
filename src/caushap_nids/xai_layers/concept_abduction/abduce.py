from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activation import concept_activation_score


@dataclass
class AbductiveExplanation:
    present_concepts: list[str]
    absent_concepts: list[str]
    activation_scores: dict[str, float]
    natural_language: str
    top_features: dict[str, list[str]] = field(default_factory=dict)


def abductive_explanation(
    x: np.ndarray,
    concept_library: dict[str, "ConceptSubgraph"],
    thresholds: dict[str, float],
    background_stats: dict,
    feature_names: list[str],
    *,
    top_k: int = 3,
) -> AbductiveExplanation:
    """
    Main entry point for Layer C.

    Classifies each concept as 'present' (score > threshold) or 'absent'.
    Generates a natural-language explanation:
      "Attack predicted because [ConceptA] present AND [ConceptB] absent."

    top_k caps the number of absent concepts reported (ordered most-absent first).
    For present concepts, the top contributing feature per concept is also listed.
    """
    x = np.asarray(x, dtype=np.float64)
    scores: dict[str, float] = {
        name: concept_activation_score(x, concept, background_stats, feature_names)
        for name, concept in concept_library.items()
    }

    present: list[str] = [n for n, s in scores.items() if s > thresholds.get(n, 0.5)]
    absent:  list[str] = sorted(
        [n for n in scores if n not in present],
        key=lambda n: scores[n]   # lowest score = most absent, listed first
    )[:top_k]

    # Top contributing feature per present concept (highest |directed z-score|)
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    mean = np.asarray(background_stats["mean"], dtype=np.float64)
    std  = np.asarray(background_stats["std"],  dtype=np.float64).clip(1e-9)
    top_features: dict[str, list[str]] = {}
    for name in present:
        concept = concept_library[name]
        ranked: list[tuple[float, str]] = []
        concept_weights = getattr(concept, "weights", [1.0] * len(concept.core_features))
        for feat, direction, weight in zip(concept.core_features, concept.directions, concept_weights):
            if feat in feat_idx:
                idx = feat_idx[feat]
                z = direction * (float(x[idx]) - float(mean[idx])) / float(std[idx])
                ranked.append((z * float(weight), feat))
        ranked.sort(reverse=True)
        top_features[name] = [f for _, f in ranked[:2]]

    parts: list[str] = []
    if present:
        parts.append("present — " + ", ".join(
            f"{n}({', '.join(top_features.get(n, []))})" for n in present
        ))
    if absent:
        parts.append("absent — " + ", ".join(absent))
    nl = ("Attack predicted because " + "; ".join(parts)) if parts else "No concept pattern identified."

    return AbductiveExplanation(
        present_concepts=present,
        absent_concepts=absent,
        activation_scores=scores,
        natural_language=nl,
        top_features=top_features,
    )
