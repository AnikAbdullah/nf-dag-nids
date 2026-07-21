from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from math import comb

import networkx as nx
import numpy as np

from .cache import background_digest, cached_subset_eval
from .cc_shapley import cc_shapley_correction
from .interventional import (
    graph_fingerprint,
    indexed_dag,
    infer_feature_names,
    interventional_expectation,
)
from .mediation import decompose_effects


@dataclass
class CausalShapleyExplanation:
    phi: np.ndarray
    direct_effects: np.ndarray
    indirect_effects: np.ndarray
    feature_names: list[str]
    flow_index: int | None = None


def causal_shapley(
    detector: object,
    dag: nx.DiGraph,
    x: np.ndarray,
    background: np.ndarray,
    *,
    n_samples: int = 200,
    causal_method: str = "interventional",
    stability_smoothing: float = 0.0,
) -> CausalShapleyExplanation:
    """Compute causal Shapley values for one flow.

    The value function is the detector anomaly score. Coalitions are evaluated
    using interventional expectations over the DAG and solved with a weighted
    KernelSHAP regression constrained to satisfy efficiency.
    """
    if causal_method not in {"interventional", "asymmetric", "cc-shapley"}:
        raise ValueError(
            "causal_method must be one of: 'interventional', 'asymmetric', 'cc-shapley'"
        )
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 <= stability_smoothing <= 1.0:
        raise ValueError("stability_smoothing must be between 0 and 1")

    x_arr = np.asarray(x, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array")
    if bg.ndim != 2:
        raise ValueError("background must be a 2D array")
    if bg.shape[1] != x_arr.shape[0]:
        raise ValueError("background feature count must match x")

    n_features = x_arr.shape[0]
    feature_names_raw = infer_feature_names(dag, n_features)
    feature_names = [str(name) for name in feature_names_raw]
    score_fn = _score_fn(detector)

    g_idx = indexed_dag(dag, feature_names_raw)
    rng = np.random.default_rng(_seed_from_x(x_arr, background_digest(bg), causal_method))
    masks = _coalition_masks(n_features, n_samples, rng, g_idx, causal_method)

    x_tuple = tuple(float(v) for v in np.round(x_arr, 12))
    eval_hash = (
        f"{background_digest(bg)}:{graph_fingerprint(dag, feature_names_raw)}:"
        f"{causal_method}:{n_samples}"
    )

    def eval_subset(subset: frozenset[int]) -> float:
        return interventional_expectation(
            x_arr,
            subset,
            dag,
            bg,
            score_fn,
            n_samples=n_samples,
        )

    empty = cached_subset_eval(frozenset(), x_tuple, eval_hash, eval_subset)
    full_subset = frozenset(range(n_features))
    full = cached_subset_eval(full_subset, x_tuple, eval_hash, eval_subset)

    values = np.array(
        [cached_subset_eval(frozenset(np.flatnonzero(mask)), x_tuple, eval_hash, eval_subset)
         for mask in masks],
        dtype=np.float64,
    )
    phi = _weighted_kernel_solution(masks, values, empty, full)

    if causal_method == "cc-shapley":
        phi = cc_shapley_correction(
            phi,
            dag,
            x_arr,
            bg,
            score_fn,
            feature_names_raw,
            n_samples=n_samples,
        )
        phi = _enforce_efficiency(phi, full - empty)

    if stability_smoothing > 0.0:
        phi = _graph_smooth_attribution(phi, dag, feature_names_raw, stability_smoothing)
        phi = _enforce_efficiency(phi, full - empty)

    direct, indirect = decompose_effects(phi, dag, feature_names_raw)
    return CausalShapleyExplanation(
        phi=phi,
        direct_effects=direct,
        indirect_effects=indirect,
        feature_names=feature_names,
    )


def vanilla_kernel_shap(
    detector: object,
    x: np.ndarray,
    background: np.ndarray,
    *,
    n_samples: int = 200,
    feature_names: list[str] | None = None,
) -> CausalShapleyExplanation:
    """Compute independent KernelSHAP values for one flow.

    This is the vanilla baseline used by A0/A1.  Coalitions are evaluated by
    replacing absent features with benign background rows independently of any
    DAG, then solving the same constrained KernelSHAP regression as the causal
    implementation.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    x_arr = np.asarray(x, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array")
    if bg.ndim != 2:
        raise ValueError("background must be a 2D array")
    if bg.shape[1] != x_arr.shape[0]:
        raise ValueError("background feature count must match x")

    n_features = x_arr.shape[0]
    names = feature_names or [f"f{i}" for i in range(n_features)]
    rng = np.random.default_rng(_seed_from_x(x_arr, background_digest(bg), "vanilla"))
    masks = _independent_coalition_masks(n_features, n_samples, rng)
    score_fn = _score_fn(detector)

    empty = float(np.mean(score_fn(bg)))
    full = float(score_fn(x_arr.reshape(1, -1))[0])

    values = np.array([
        _independent_subset_value(x_arr, bg, np.flatnonzero(mask), score_fn)
        for mask in masks
    ], dtype=np.float64)
    phi = _weighted_kernel_solution(masks, values, empty, full)
    return CausalShapleyExplanation(
        phi=phi,
        direct_effects=phi.copy(),
        indirect_effects=np.zeros_like(phi),
        feature_names=[str(name) for name in names],
    )


def _score_fn(detector: object) -> Callable[[np.ndarray], np.ndarray]:
    if hasattr(detector, "score"):
        return getattr(detector, "score")
    if callable(detector):
        return detector
    raise TypeError("detector must implement score(X) or be callable")


def _independent_subset_value(
    x: np.ndarray,
    background: np.ndarray,
    subset_idx: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    samples = background.copy()
    if len(subset_idx) > 0:
        samples[:, subset_idx] = x[subset_idx]
    return float(np.mean(score_fn(samples)))


def _independent_coalition_masks(
    n_features: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_features == 1:
        return np.ones((1, 1), dtype=bool)

    max_exact = 2**n_features - 2
    if n_features <= 10 and n_samples >= max_exact:
        return np.array([
            [(raw >> i) & 1 for i in range(n_features)]
            for raw in range(1, 2**n_features - 1)
        ], dtype=bool)

    target = max(n_samples, n_features + 2)
    masks: list[np.ndarray] = []
    for idx in range(n_features):
        singleton = np.zeros(n_features, dtype=bool)
        singleton[idx] = True
        masks.append(singleton)

        leave_one_out = np.ones(n_features, dtype=bool)
        leave_one_out[idx] = False
        masks.append(leave_one_out)

    sizes = np.arange(1, n_features)
    size_probs = 1.0 / (sizes * (n_features - sizes))
    size_probs = size_probs / size_probs.sum()
    attempts = 0
    while len(masks) < target and attempts < target * 20:
        attempts += 1
        k = int(rng.choice(sizes, p=size_probs))
        chosen = rng.choice(n_features, size=k, replace=False)
        mask = np.zeros(n_features, dtype=bool)
        mask[chosen] = True
        masks.append(mask)

    return _unique_masks(masks, n_features)


def _coalition_masks(
    n_features: int,
    n_samples: int,
    rng: np.random.Generator,
    dag_idx: nx.DiGraph,
    causal_method: str,
) -> np.ndarray:
    if n_features == 1:
        return np.ones((1, 1), dtype=bool)

    max_exact = 2**n_features - 2
    if n_features <= 10 and n_samples >= max_exact:
        masks = []
        for raw in range(1, 2**n_features - 1):
            mask = np.array([(raw >> i) & 1 for i in range(n_features)], dtype=bool)
            masks.append(_dag_mask(mask, dag_idx, causal_method))
        return _unique_masks(masks, n_features)

    target = max(n_samples, n_features + 2)
    masks: list[np.ndarray] = []

    for idx in range(n_features):
        singleton = np.zeros(n_features, dtype=bool)
        singleton[idx] = True
        masks.append(_dag_mask(singleton, dag_idx, causal_method))

        leave_one_out = np.ones(n_features, dtype=bool)
        leave_one_out[idx] = False
        masks.append(_dag_mask(leave_one_out, dag_idx, causal_method))

    sizes = np.arange(1, n_features)
    size_probs = 1.0 / (sizes * (n_features - sizes))
    size_probs = size_probs / size_probs.sum()
    attempts = 0
    while len(masks) < target and attempts < target * 20:
        attempts += 1
        k = int(rng.choice(sizes, p=size_probs))
        chosen = rng.choice(n_features, size=k, replace=False)
        mask = np.zeros(n_features, dtype=bool)
        mask[chosen] = True
        masks.append(_dag_mask(mask, dag_idx, causal_method))

    return _unique_masks(masks, n_features)


def _dag_mask(mask: np.ndarray, dag_idx: nx.DiGraph, causal_method: str) -> np.ndarray:
    if causal_method != "asymmetric":
        return mask

    closed = mask.copy()
    for node in np.flatnonzero(mask):
        for ancestor in nx.ancestors(dag_idx, int(node)):
            closed[int(ancestor)] = True
    return closed


def _unique_masks(masks: list[np.ndarray], n_features: int) -> np.ndarray:
    seen: set[tuple[bool, ...]] = set()
    unique: list[np.ndarray] = []
    for mask in masks:
        key = tuple(bool(v) for v in mask)
        if any(key) and not all(key) and key not in seen:
            seen.add(key)
            unique.append(mask)
    if not unique:
        return np.ones((1, n_features), dtype=bool)
    return np.vstack(unique)


def _weighted_kernel_solution(
    masks: np.ndarray,
    values: np.ndarray,
    empty_value: float,
    full_value: float,
) -> np.ndarray:
    n_features = masks.shape[1]
    if n_features == 1:
        return np.array([full_value - empty_value], dtype=np.float64)

    z = masks.astype(np.float64)
    y = values - empty_value
    weights = np.array([_kernel_weight(int(row.sum()), n_features) for row in masks])

    full_row = np.ones((1, n_features), dtype=np.float64)
    z = np.vstack([z, full_row])
    y = np.concatenate([y, np.array([full_value - empty_value], dtype=np.float64)])
    weights = np.concatenate([weights, np.array([max(weights.max(), 1.0) * 1e6])])

    sqrt_w = np.sqrt(weights)
    design = z * sqrt_w[:, None]
    target = y * sqrt_w
    lhs = design.T @ design
    rhs = design.T @ target
    try:
        phi = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        ridge = np.eye(n_features, dtype=np.float64) * (
            1e-14 * max(float(np.trace(lhs)) / n_features, 1.0)
        )
        try:
            phi = np.linalg.solve(lhs + ridge, rhs)
        except np.linalg.LinAlgError:
            phi = np.linalg.lstsq(lhs + ridge, rhs, rcond=None)[0]

    return _enforce_efficiency(phi, full_value - empty_value)


def _kernel_weight(coalition_size: int, n_features: int) -> float:
    if coalition_size <= 0 or coalition_size >= n_features:
        return 1e6
    return (n_features - 1.0) / (
        comb(n_features, coalition_size) * coalition_size * (n_features - coalition_size)
    )


def _enforce_efficiency(phi: np.ndarray, expected_sum: float) -> np.ndarray:
    out = np.asarray(phi, dtype=np.float64).copy()
    gap = float(expected_sum - np.sum(out))
    if abs(gap) <= 1e-10:
        return out
    weights = np.abs(out)
    if float(weights.sum()) <= 1e-12:
        out += gap / out.size
    else:
        out += gap * (weights / weights.sum())
    return out


def _graph_smooth_attribution(
    phi: np.ndarray,
    dag: nx.DiGraph,
    feature_names: list[Hashable],
    smoothing: float,
) -> np.ndarray:
    """Diffuse attribution over DAG Markov neighborhoods while preserving sum."""
    if dag.number_of_edges() == 0 or smoothing <= 0.0:
        return np.asarray(phi, dtype=np.float64).copy()

    n_features = len(feature_names)
    node_to_idx = {node: i for i, node in enumerate(feature_names)}
    adjacency = np.eye(n_features, dtype=np.float64)
    for src, dst in dag.edges():
        if src not in node_to_idx or dst not in node_to_idx:
            continue
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        adjacency[src_idx, dst_idx] = 1.0
        adjacency[dst_idx, src_idx] = 1.0

    col_sums = adjacency.sum(axis=0, keepdims=True)
    transition = adjacency / np.maximum(col_sums, 1.0)
    phi_arr = np.asarray(phi, dtype=np.float64)
    return ((1.0 - smoothing) * phi_arr) + (smoothing * (transition @ phi_arr))


def _seed_from_x(x: np.ndarray, bg_hash: str, causal_method: str) -> int:
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    h.update(str(np.asarray(x).shape[0]).encode("utf-8"))
    h.update(bg_hash.encode("utf-8"))
    h.update(causal_method.encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)
