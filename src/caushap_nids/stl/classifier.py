"""Window-level MITRE technique classifier.

Given a window DataFrame (raw NF-v2 flows), evaluates all registered
MitreTechnique formulae and returns:
  - the full list of STLResult objects (one per technique)
  - the single best prediction (highest positive robustness, or None)
"""

from __future__ import annotations

import polars as pl

from .formulae import MitreTechnique, STLResult, ALL_TECHNIQUES


def evaluate_all(
    window: pl.DataFrame,
    techniques: tuple[MitreTechnique, ...] = ALL_TECHNIQUES,
) -> list[STLResult]:
    """Evaluate every technique formula against the window.

    Returns one STLResult per technique, in the same order as *techniques*.
    """
    return [t.evaluate(window) for t in techniques]


def predict(
    window: pl.DataFrame,
    techniques: tuple[MitreTechnique, ...] = ALL_TECHNIQUES,
) -> str | None:
    """Return the MITRE ID of the satisfied formula with the highest robustness.

    If multiple formulae are satisfied (overlapping patterns), the most
    confident one (largest ρ) wins.  Returns None if no formula fires.
    """
    results = evaluate_all(window, techniques)
    satisfied = [r for r in results if r.satisfied]
    if not satisfied:
        return None
    return max(satisfied, key=lambda r: r.robustness).mitre_id


def window_ground_truth(
    window: pl.DataFrame,
    attack_family_col: str = "attack_family",
    majority_threshold: float = 0.5,
) -> str | None:
    """Derive the ground-truth label for a window by majority vote.

    Returns:
        The attack_family string if one family accounts for ≥ majority_threshold
        of the window's flows, otherwise None (ambiguous / excluded from gate).

    The 'Benign' family is treated identically to attack families — a
    benign-dominated window returns "Benign".
    """
    if attack_family_col not in window.columns or len(window) == 0:
        return None

    counts = (
        window[attack_family_col]
        .value_counts(sort=True)
    )
    if len(counts) == 0:
        return None

    # Use column-then-index access to get plain Python scalars (Polars ≥ 1.0)
    top_family = counts[attack_family_col][0]
    top_count  = int(counts["count"][0])
    frac = top_count / len(window)

    if frac >= majority_threshold:
        return str(top_family)
    return None
