from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon as _wilcoxon


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for a paired comparison (a - b)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    std = diff.std(ddof=1)
    if std == 0.0:
        return 0.0
    return float(diff.mean() / std)


def wilcoxon_bonferroni(
    a: np.ndarray,
    b: np.ndarray,
    n_comparisons: int = 6,
    alpha: float = 0.01,
) -> dict:
    """
    Wilcoxon signed-rank test with Bonferroni correction across n_comparisons.
    Effective alpha = 0.01 / 6 ≈ 0.0017 for the full ablation table.

    Returns: stat, p_value, corrected_alpha, significant, cohens_d.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    corrected_alpha = alpha / n_comparisons
    try:
        stat, p_value = _wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        stat, p_value = 0.0, 1.0
    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "corrected_alpha": corrected_alpha,
        "significant": bool(p_value < corrected_alpha),
        "cohens_d": cohens_d(a, b),
    }
