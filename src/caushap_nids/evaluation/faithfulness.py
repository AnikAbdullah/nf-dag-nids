from __future__ import annotations

from typing import Callable

import numpy as np


def _top_k_idx(phi: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(phi))[::-1][:k]


def sufficiency(
    detector,
    x: np.ndarray,
    phi: np.ndarray,
    top_k: int = 5,
    background: np.ndarray | None = None,
) -> float:
    """
    ERASER sufficiency (DeYoung 2020).
    Top-k features alone are used; all others replaced with background mean.
    sufficiency = 1 - |score(x) - score(x_top_k_only)| / (|score(x)| + eps)
    Higher is better.
    """
    bg_mean = background.mean(axis=0) if background is not None else np.zeros_like(x)
    x_masked = bg_mean.copy()
    x_masked[_top_k_idx(phi, top_k)] = x[_top_k_idx(phi, top_k)]
    score_full = float(detector.score(x.reshape(1, -1))[0])
    score_masked = float(detector.score(x_masked.reshape(1, -1))[0])
    value = 1.0 - abs(score_full - score_masked) / (abs(score_full) + 1e-8)
    return float(np.clip(value, 0.0, 1.0))


def comprehensiveness(
    detector,
    x: np.ndarray,
    phi: np.ndarray,
    top_k: int = 5,
    background: np.ndarray | None = None,
) -> float:
    """
    ERASER comprehensiveness (DeYoung 2020).
    Top-k features removed (replaced with background mean).
    comprehensiveness = |score(x) - score(x_without_top_k)| / (|score(x)| + eps)
    Higher is better.
    """
    bg_mean = background.mean(axis=0) if background is not None else np.zeros_like(x)
    x_ablated = x.copy()
    x_ablated[_top_k_idx(phi, top_k)] = bg_mean[_top_k_idx(phi, top_k)]
    score_full = float(detector.score(x.reshape(1, -1))[0])
    score_ablated = float(detector.score(x_ablated.reshape(1, -1))[0])
    value = abs(score_full - score_ablated) / (abs(score_full) + 1e-8)
    return float(np.clip(value, 0.0, 1.0))


def lipschitz_stability(
    explain_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    n_perturbations: int = 50,
    epsilon: float = 0.05,
    seed: int = 42,
    quantile: float | None = None,
) -> float:
    """
    Alvarez-Melis & Jaakkola (2018) empirical Lipschitz constant.

    L = max_i ||phi(x + delta_i) - phi(x)||_2 / ||delta_i||_2  (default: quantile=None → max)
    L = quantile_q of the per-perturbation ratios                (when quantile in (0, 1] given)

    Lower = more stable. The quantile mode is a robust variant (Friedman 2001 §10
    "robust loss"-style trimming) — a single outlier perturbation can swing the
    max wildly, so reporting the 95th percentile of ratios is a more stable
    empirical estimator at the same sample size. The original max behaviour is
    preserved as the default (quantile=None) for backwards compatibility.
    """
    rng = np.random.default_rng(seed)
    phi_x = explain_fn(x)
    ratios = []
    for _ in range(n_perturbations):
        delta = rng.normal(0.0, epsilon, size=x.shape)
        phi_perturbed = explain_fn(x + delta)
        ratio = float(np.linalg.norm(phi_perturbed - phi_x)) / (float(np.linalg.norm(delta)) + 1e-12)
        ratios.append(ratio)
    if quantile is None:
        return float(max(ratios)) if ratios else 0.0
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    return float(np.quantile(np.asarray(ratios), quantile))
