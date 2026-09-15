"""Structural-equation feasibility for DAG-constrained counterfactuals.

The original co-change criterion (``objectives.feasibility``) scores an edge as
satisfied when both endpoints change or neither does.  That criterion is
degenerate: it is maximised both by changing nothing and by changing every
feature, and is *worst* in the 5-20 feature range where actionable recourse
lives.  A search told to maximise it therefore drives sparsity to the feature
count rather than down.

This module replaces it with the standard causal-recourse criterion (Mahajan
et al., 2019; Karimi et al., 2020): a counterfactual is feasible when every
non-root feature is consistent with what its parents predict under structural
equations fitted on benign traffic.  Changing a parent and letting children
follow is feasible; a child moving with no parental support is not.  Unlike
co-change the criterion is asymmetric, so it cannot be satisfied by perturbing
everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

_MIN_RESID_STD = 1e-6


@dataclass
class StructuralEquations:
    """Per-node linear structural equations f_v(parents(v)) fitted on benign rows.

    Roots (no parents in the DAG) carry no equation: nothing predicts them, so
    intervening on one is always structurally feasible.
    """

    parents: dict[str, list[str]]
    coef: dict[str, np.ndarray]
    intercept: dict[str, float]
    resid_std: dict[str, float]
    feature_names: list[str]

    @property
    def non_root_nodes(self) -> list[str]:
        return [v for v, p in self.parents.items() if p]

    def predict(self, x: np.ndarray, node: str) -> float:
        """Predicted value of ``node`` given the parent values in ``x``."""
        idx = {f: i for i, f in enumerate(self.feature_names)}
        pa = self.parents[node]
        if not pa:
            raise ValueError(f"{node!r} is a root: no structural equation")
        pa_vals = np.array([x[idx[p]] for p in pa], dtype=np.float64)
        return float(self.coef[node] @ pa_vals + self.intercept[node])


def fit_structural_equations(
    X_benign: np.ndarray,
    dag: nx.DiGraph,
    feature_names: list[str],
    *,
    ridge_alpha: float = 1.0,
    max_rows: int = 200_000,
    seed: int = 42,
) -> StructuralEquations:
    """Fit one ridge equation per non-root node on benign traffic.

    Benign-only, matching the no-leakage rule used for detector training: the
    equations describe how features relate under normal protocol behaviour, and
    a counterfactual is judged by whether it stays on that manifold.
    """
    X_benign = np.asarray(X_benign, dtype=np.float64)
    if X_benign.ndim != 2:
        raise ValueError("X_benign must be 2-D (n_rows, n_features)")

    if len(X_benign) > max_rows:
        rng = np.random.default_rng(seed)
        X_benign = X_benign[rng.choice(len(X_benign), size=max_rows, replace=False)]

    idx = {f: i for i, f in enumerate(feature_names)}
    parents: dict[str, list[str]] = {}
    coef: dict[str, np.ndarray] = {}
    intercept: dict[str, float] = {}
    resid_std: dict[str, float] = {}

    for node in dag.nodes():
        if node not in idx:
            continue
        pa = [p for p in dag.predecessors(node) if p in idx]
        parents[node] = pa
        if not pa:
            continue

        A = X_benign[:, [idx[p] for p in pa]]
        y = X_benign[:, idx[node]]

        # Ridge via the normal equations, intercept handled by centring.
        A_mean, y_mean = A.mean(axis=0), float(y.mean())
        Ac, yc = A - A_mean, y - y_mean
        gram = Ac.T @ Ac + ridge_alpha * np.eye(Ac.shape[1])
        w = np.linalg.solve(gram, Ac.T @ yc)

        coef[node] = w
        intercept[node] = y_mean - float(w @ A_mean)
        resid_std[node] = max(float(np.std(yc - Ac @ w)), _MIN_RESID_STD)

    return StructuralEquations(
        parents=parents,
        coef=coef,
        intercept=intercept,
        resid_std=resid_std,
        feature_names=list(feature_names),
    )


def structural_violation(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    sem: StructuralEquations,
    *,
    k_sigma: float = 2.0,
) -> float:
    """Fraction of non-root features left unexplained by their parents.

    A node counts as violated when the counterfactual moves it further than
    ``k_sigma`` residual deviations from its parent-predicted value *and* that
    is worse than the anchor flow already was — so a counterfactual is not
    penalised for inconsistency it inherited rather than introduced.

    0.0 = fully feasible, 1.0 = every non-root feature unexplained.
    Objective: minimise.  Mirrors ``objectives.feasibility``'s sign convention.
    """
    nodes = sem.non_root_nodes
    if not nodes:
        return 0.0

    x_orig = np.asarray(x_orig, dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)
    idx = {f: i for i, f in enumerate(sem.feature_names)}

    violations = 0
    for node in nodes:
        i = idx[node]
        tol = k_sigma * sem.resid_std[node]
        dev_cf = abs(x_cf[i] - sem.predict(x_cf, node))
        if dev_cf <= tol:
            continue
        # Only count inconsistency the counterfactual actually introduced.
        dev_orig = abs(x_orig[i] - sem.predict(x_orig, node))
        if dev_cf > max(tol, dev_orig):
            violations += 1

    return violations / len(nodes)


def structural_feasibility_rate(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    sem: StructuralEquations,
    *,
    k_sigma: float = 2.0,
) -> float:
    """1 - structural_violation. 1.0 = fully feasible. Higher is better."""
    return 1.0 - structural_violation(x_orig, x_cf, sem, k_sigma=k_sigma)


def per_node_report(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    sem: StructuralEquations,
    *,
    k_sigma: float = 2.0,
) -> dict[str, dict[str, float | bool]]:
    """Per-node diagnostic: predicted value, deviation, tolerance, verdict.

    Used to explain *why* a counterfactual is infeasible rather than only that
    it is — an analyst-facing view of which feature moved without support.
    """
    x_orig = np.asarray(x_orig, dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)
    idx = {f: i for i, f in enumerate(sem.feature_names)}

    report: dict[str, dict[str, float | bool]] = {}
    for node in sem.non_root_nodes:
        i = idx[node]
        pred = sem.predict(x_cf, node)
        tol = k_sigma * sem.resid_std[node]
        dev_cf = abs(x_cf[i] - pred)
        dev_orig = abs(x_orig[i] - sem.predict(x_orig, node))
        report[node] = {
            "value": float(x_cf[i]),
            "predicted": float(pred),
            "deviation": float(dev_cf),
            "tolerance": float(tol),
            "violated": bool(dev_cf > tol and dev_cf > max(tol, dev_orig)),
        }
    return report
