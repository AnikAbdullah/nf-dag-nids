from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from hashlib import blake2b

import networkx as nx
import numpy as np

from .cache import background_digest

_SEM_CACHE: OrderedDict[tuple[int, tuple[int, ...], str, tuple[Hashable, ...]], "_LinearSEM"] = (
    OrderedDict()
)
_MAX_SEM_CACHE_SIZE = 16


@dataclass
class _LinearSEM:
    feature_names: list[Hashable]
    topo_indices: list[int]
    parents: list[np.ndarray]
    intercepts: np.ndarray
    coefs: list[np.ndarray]
    residuals: list[np.ndarray]
    low: np.ndarray
    high: np.ndarray


def interventional_expectation(
    x: np.ndarray,
    S: frozenset[int],
    dag: nx.DiGraph,
    background: np.ndarray,
    detector_score_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 200,
) -> float:
    """Approximate E[score(X) | do(X_S = x_S)] under a DAG.

    The sampler fits a cached linear structural equation model (SEM) on the
    benign background. For a coalition S, it draws common exogenous background
    rows, clamps X_S to x_S, then recomputes descendants through their parent
    structural equations. This gives deterministic, fast Monte Carlo values
    suitable for repeated KernelSHAP coalition evaluation.
    """
    x_arr = _as_1d_float(x)
    bg = _as_2d_float(background)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if x_arr.shape[0] != bg.shape[1]:
        raise ValueError(
            f"x has {x_arr.shape[0]} features but background has {bg.shape[1]} columns"
        )

    d = x_arr.shape[0]
    subset = frozenset(int(i) for i in S)
    _validate_subset(subset, d)

    if len(subset) == d:
        return _score_mean(detector_score_fn, x_arr.reshape(1, -1))

    sem = _get_sem(dag, bg, d)
    sample_count = min(int(n_samples), max(1, len(bg)))
    rng = _common_rng(x_arr, bg, sample_count)
    row_idx = rng.integers(0, len(bg), size=sample_count)
    samples = bg[row_idx].copy()

    if not subset:
        return _score_mean(detector_score_fn, samples)

    residual_idx = rng.integers(0, len(bg), size=(sample_count, d))
    changed = np.zeros(d, dtype=bool)
    for idx in subset:
        samples[:, idx] = x_arr[idx]
        changed[idx] = True

    for node_idx in sem.topo_indices:
        if node_idx in subset:
            continue
        parent_idx = sem.parents[node_idx]
        if parent_idx.size == 0 or not changed[parent_idx].any():
            continue

        predicted = sem.intercepts[node_idx] + samples[:, parent_idx] @ sem.coefs[node_idx]
        noise = sem.residuals[node_idx][residual_idx[:, node_idx]]
        samples[:, node_idx] = np.clip(predicted + noise, sem.low[node_idx], sem.high[node_idx])
        changed[node_idx] = True

    return _score_mean(detector_score_fn, samples)


def infer_feature_names(dag: nx.DiGraph, n_features: int) -> list[Hashable]:
    """Infer the matrix column order represented by a DAG."""
    nodes = list(dag.nodes())
    if len(nodes) != n_features:
        raise ValueError(f"dag has {len(nodes)} nodes but data has {n_features} features")

    int_nodes = set(range(n_features))
    if set(nodes) == int_nodes:
        return list(range(n_features))

    try:
        from caushap_nids.dag.constants import NF_V2_FEATURES
    except Exception:  # pragma: no cover - defensive for standalone reuse
        NF_V2_FEATURES = []

    if len(NF_V2_FEATURES) == n_features and set(NF_V2_FEATURES) == set(nodes):
        return list(NF_V2_FEATURES)

    return nodes


def graph_fingerprint(dag: nx.DiGraph, feature_names: list[Hashable]) -> str:
    """Hash graph structure in the resolved feature order."""
    node_to_idx = {node: i for i, node in enumerate(feature_names)}
    h = blake2b(digest_size=12)
    for src, dst in sorted(dag.edges()):
        if src in node_to_idx and dst in node_to_idx:
            h.update(f"{node_to_idx[src]}->{node_to_idx[dst]}|".encode("utf-8"))
    return h.hexdigest()


def indexed_dag(dag: nx.DiGraph, feature_names: list[Hashable]) -> nx.DiGraph:
    """Relabel a feature-name DAG to integer column indices."""
    node_to_idx = {node: i for i, node in enumerate(feature_names)}
    missing = [node for node in dag.nodes() if node not in node_to_idx]
    if missing:
        raise ValueError(f"dag contains nodes not present in feature_names: {missing[:3]}")

    out = nx.DiGraph()
    out.add_nodes_from(range(len(feature_names)))
    out.add_edges_from((node_to_idx[src], node_to_idx[dst]) for src, dst in dag.edges())
    if not nx.is_directed_acyclic_graph(out):
        raise ValueError("dag must be acyclic")
    return out


def _get_sem(dag: nx.DiGraph, background: np.ndarray, n_features: int) -> _LinearSEM:
    feature_names = infer_feature_names(dag, n_features)
    fp = graph_fingerprint(dag, feature_names)
    key = (id(background), background.shape, background_digest(background), tuple(feature_names))
    key = (key[0], key[1], f"{key[2]}:{fp}", key[3])

    cached = _SEM_CACHE.get(key)
    if cached is not None:
        _SEM_CACHE.move_to_end(key)
        return cached

    sem = _fit_linear_sem(dag, background, feature_names)
    _SEM_CACHE[key] = sem
    if len(_SEM_CACHE) > _MAX_SEM_CACHE_SIZE:
        _SEM_CACHE.popitem(last=False)
    return sem


def _fit_linear_sem(
    dag: nx.DiGraph,
    background: np.ndarray,
    feature_names: list[Hashable],
) -> _LinearSEM:
    g_idx = indexed_dag(dag, feature_names)
    d = len(feature_names)
    parents: list[np.ndarray] = []
    intercepts = np.zeros(d, dtype=np.float64)
    coefs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    low = np.nanmin(background, axis=0).astype(np.float64)
    high = np.nanmax(background, axis=0).astype(np.float64)

    for node_idx in range(d):
        parent_idx = np.fromiter(sorted(g_idx.predecessors(node_idx)), dtype=int)
        parents.append(parent_idx)
        y = background[:, node_idx]

        if parent_idx.size == 0:
            mu = float(np.mean(y))
            intercepts[node_idx] = mu
            coefs.append(np.empty(0, dtype=np.float64))
            residuals.append((y - mu).astype(np.float64))
            continue

        x_parent = background[:, parent_idx]
        design = np.column_stack([np.ones(len(background), dtype=np.float64), x_parent])
        beta = _ridge_solve(design, y)
        intercepts[node_idx] = float(beta[0])
        coefs.append(beta[1:].astype(np.float64))
        residuals.append((y - design @ beta).astype(np.float64))

    return _LinearSEM(
        feature_names=list(feature_names),
        topo_indices=list(nx.topological_sort(g_idx)),
        parents=parents,
        intercepts=intercepts,
        coefs=coefs,
        residuals=residuals,
        low=low,
        high=high,
    )


def _ridge_solve(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    gram = design.T @ design
    rhs = design.T @ y
    scale = max(float(np.trace(gram)) / max(gram.shape[0], 1), 1.0)
    gram = gram + np.eye(gram.shape[0], dtype=np.float64) * (1e-8 * scale)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _common_rng(x: np.ndarray, background: np.ndarray, sample_count: int) -> np.random.Generator:
    del x
    h = blake2b(digest_size=8)
    h.update(background_digest(background).encode("utf-8"))
    h.update(str(sample_count).encode("utf-8"))
    seed = int.from_bytes(h.digest(), "little", signed=False)
    return np.random.default_rng(seed)


def _score_mean(
    detector_score_fn: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
) -> float:
    scores = np.asarray(detector_score_fn(samples), dtype=np.float64)
    if scores.ndim == 0:
        return float(scores)
    scores = scores.reshape(-1)
    if scores.size == 0:
        raise ValueError("detector_score_fn returned no scores")
    return float(np.nanmean(scores))


def _as_1d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be a 1D array")
    if not np.isfinite(arr).all():
        raise ValueError("x contains NaN or infinite values")
    return arr


def _as_2d_float(background: np.ndarray) -> np.ndarray:
    arr = np.asarray(background, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("background must be a 2D array")
    if arr.shape[0] == 0:
        raise ValueError("background must contain at least one row")
    if not np.isfinite(arr).all():
        raise ValueError("background contains NaN or infinite values")
    return arr


def _validate_subset(subset: frozenset[int], n_features: int) -> None:
    bad = [idx for idx in subset if idx < 0 or idx >= n_features]
    if bad:
        raise ValueError(f"subset contains invalid feature indices: {bad[:3]}")
