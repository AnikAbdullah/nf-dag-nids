import logging
import warnings

import networkx as nx
import numpy as np

from .constants import NF_V2_FEATURES
from .edge_types import EdgeType

logger = logging.getLogger(__name__)


def run_pc(
    df_benign,
    alpha: float = 0.05,
    feature_cols: list[str] | None = None,
) -> nx.DiGraph:
    """Run PC algorithm (causal-learn) on benign flows.

    Returns a directed graph; undirected edges are kept as EdgeType.D.
    Falls back to an empty graph with a warning if causal-learn is unavailable.

    df_benign: polars or pandas DataFrame with benign-only flows.
    """
    if feature_cols is None:
        feature_cols = [c for c in NF_V2_FEATURES if c in _columns(df_benign)]

    X = _to_numpy(df_benign, feature_cols)

    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
    except ImportError:
        warnings.warn(
            "causal-learn not installed — PC algorithm unavailable. "
            "Install with: pip install causal-learn",
            stacklevel=2,
        )
        return _empty_dag(feature_cols)

    # Drop near-zero-variance columns — FisherZ needs an invertible correlation matrix
    col_var = np.var(X, axis=0)
    active_mask = col_var > 1e-10
    active_cols = [c for c, keep in zip(feature_cols, active_mask) if keep]
    n_dropped = len(feature_cols) - len(active_cols)
    if n_dropped:
        dropped = [c for c, keep in zip(feature_cols, active_mask) if not keep]
        warnings.warn(
            f"PC: dropped {n_dropped} near-zero-variance feature(s) before FisherZ test: {dropped}",
            stacklevel=2,
        )
        X = X[:, active_mask]

    _patch_fisherz_robust()
    logger.info("Running PC algorithm on %d benign samples × %d features, alpha=%s",
                len(X), len(active_cols), alpha)
    cg = pc(X, alpha=alpha, indep_test=fisherz)

    dag: nx.DiGraph = nx.DiGraph()
    dag.add_nodes_from(feature_cols)
    for i, j in zip(*cg.G.graph.nonzero()):
        src = active_cols[i]
        dst = active_cols[j]
        if dag.has_edge(dst, src):
            continue
        dag.add_edge(src, dst, edge_type=EdgeType.D, justification="PC algorithm (benign data)")

    return dag


def run_notears(
    df_benign,
    lambda1: float = 0.1,
    weight_threshold: float = 0.3,
    feature_cols: list[str] | None = None,
) -> nx.DiGraph:
    """Run NOTEARS continuous structure learning on benign flows.

    Weights are thresholded at weight_threshold to produce binary edges.
    Falls back to an empty graph with a warning if notears is unavailable.
    """
    if feature_cols is None:
        feature_cols = [c for c in NF_V2_FEATURES if c in _columns(df_benign)]

    X = _to_numpy(df_benign, feature_cols)

    try:
        from notears.linear import notears_linear
    except ImportError:
        warnings.warn(
            "notears not installed — NOTEARS algorithm unavailable. "
            "Install with: pip install git+https://github.com/xunzheng/notears",
            stacklevel=2,
        )
        return _empty_dag(feature_cols)

    logger.info("Running NOTEARS on %d benign samples, lambda1=%s", len(X), lambda1)
    W_est = notears_linear(X, lambda1=lambda1, loss_type="l2")

    dag: nx.DiGraph = nx.DiGraph()
    dag.add_nodes_from(feature_cols)
    n = len(feature_cols)
    for i in range(n):
        for j in range(n):
            if abs(W_est[i, j]) > weight_threshold:
                dag.add_edge(
                    feature_cols[i], feature_cols[j],
                    edge_type=EdgeType.D,
                    justification=f"NOTEARS weight={W_est[i,j]:.3f}",
                    weight=float(W_est[i, j]),
                )

    # NOTEARS can produce cycles in edge selection; remove back-edges
    while not nx.is_directed_acyclic_graph(dag):
        cycle = next(nx.simple_cycles(dag))
        dag.remove_edge(cycle[-1], cycle[0])
        logger.debug("Removed back-edge %s→%s to break cycle", cycle[-1], cycle[0])

    return dag


# ── helpers ────────────────────────────────────────────────────────────────

def _patch_fisherz_robust() -> None:
    """Monkey-patch causallearn FisherZ to survive singular / NaN sub-correlation matrices.

    np.corrcoef produces NaN for zero-variance columns; those NaNs propagate into
    every sub-matrix FisherZ builds during skeleton discovery.  np.linalg.inv on a
    NaN matrix does NOT raise LinAlgError — it returns a NaN matrix — so the
    existing try/except in causallearn never fires; instead the downstream arithmetic
    produces a NaN r-value which eventually surfaces as the ValueError the caller sees.

    This patch: (1) replaces NaN entries with 0 correlation + identity diagonal,
    (2) falls back to pinv when inv still fails, (3) returns p=1.0 (treat as
    independent) if the result is still non-finite after pinv.
    """
    try:
        from causallearn.utils.cit import FisherZ
    except ImportError:
        return  # causal-learn not installed; run_pc will warn and return early anyway

    from math import sqrt, log
    from scipy.stats import norm as _norm
    import numpy as _np

    if getattr(FisherZ, '_robust_patched', False):
        return  # already patched in this process

    _orig_call = FisherZ.__call__

    def _robust_call(self, X, Y, condition_set=None):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]

        var = Xs + Ys + condition_set
        sub_corr = self.correlation_matrix[_np.ix_(var, var)].copy()

        if _np.isnan(sub_corr).any():
            sub_corr = _np.nan_to_num(sub_corr, nan=0.0)
            _np.fill_diagonal(sub_corr, 1.0)

        try:
            inv = _np.linalg.inv(sub_corr)
        except _np.linalg.LinAlgError:
            inv = _np.linalg.pinv(sub_corr)

        denom = sqrt(abs(inv[0, 0] * inv[1, 1]))
        if denom < 1e-15 or not _np.isfinite(inv[0, 1]):
            p = 1.0
            self.pvalue_cache[cache_key] = p
            return p

        r = -inv[0, 1] / denom
        r = max(-(1.0 - _np.finfo(float).eps), min(1.0 - _np.finfo(float).eps, r))
        Z = 0.5 * log((1 + r) / (1 - r))
        df = max(self.sample_size - len(condition_set) - 3, 1)
        stat = sqrt(df) * abs(Z)
        p = 2.0 * (1.0 - _norm.cdf(abs(stat)))
        self.pvalue_cache[cache_key] = p
        return p

    FisherZ.__call__ = _robust_call
    FisherZ._robust_patched = True

def _columns(df) -> list[str]:
    try:
        return list(df.columns)
    except AttributeError:
        return list(df.keys())


def _to_numpy(df, cols: list[str]) -> np.ndarray:
    try:
        return df.select(cols).to_numpy().astype(np.float64)
    except AttributeError:
        return df[cols].to_numpy(dtype=np.float64)


def _empty_dag(feature_cols: list[str]) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(feature_cols)
    return dag
