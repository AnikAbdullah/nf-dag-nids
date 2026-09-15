"""Tests for structural-equation feasibility (Layer B).

The properties pinned here are the ones the co-change criterion in
``objectives.feasibility`` fails: most importantly that perturbing every
feature at once must NOT score as feasible.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from caushap_nids.xai_layers.multi_obj_cf.objectives import feasibility as co_change_violation
from caushap_nids.xai_layers.multi_obj_cf.structural import (
    fit_structural_equations,
    per_node_report,
    structural_feasibility_rate,
    structural_violation,
)

NAMES = ["a", "b", "c", "d"]


def _chain_dag() -> nx.DiGraph:
    """a -> b -> c, with d an isolated root."""
    g = nx.DiGraph()
    g.add_nodes_from(NAMES)
    g.add_edges_from([("a", "b"), ("b", "c")])
    return g


def _scm_sample(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Draw from a known linear SCM: b = 2a + eps, c = -1.5b + eps, d independent."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(scale=0.05, size=n)
    c = -1.5 * b + rng.normal(scale=0.05, size=n)
    d = rng.normal(size=n)
    return np.column_stack([a, b, c, d])


@pytest.fixture(scope="module")
def sem():
    return fit_structural_equations(_scm_sample(), _chain_dag(), NAMES)


def test_fit_recovers_known_coefficients(sem):
    assert sem.coef["b"][0] == pytest.approx(2.0, abs=0.05)
    assert sem.coef["c"][0] == pytest.approx(-1.5, abs=0.05)


def test_roots_carry_no_equation(sem):
    assert sem.parents["a"] == []
    assert sem.parents["d"] == []
    assert set(sem.non_root_nodes) == {"b", "c"}


def test_no_change_is_feasible(sem):
    x = np.array([1.0, 2.0, -3.0, 0.5])
    assert structural_feasibility_rate(x, x.copy(), sem) == pytest.approx(1.0)


def test_changing_everything_is_infeasible(sem):
    """The regression test: co-change calls this perfectly feasible, structural does not."""
    x = np.array([1.0, 2.0, -3.0, 0.5])
    x_cf = x + 5.0  # every feature moved by the same arbitrary amount

    assert 1.0 - co_change_violation(x, x_cf, _chain_dag(), NAMES) == pytest.approx(1.0)
    assert structural_feasibility_rate(x, x_cf, sem) < 0.5


def test_child_moved_without_parent_is_violated(sem):
    x = np.array([1.0, 2.0, -3.0, 0.5])
    x_cf = x.copy()
    x_cf[NAMES.index("c")] += 4.0      # c moves; its parent b does not
    assert structural_violation(x, x_cf, sem) > 0.0
    assert per_node_report(x, x_cf, sem)["c"]["violated"] is True


def test_consistent_propagation_is_feasible(sem):
    """Move a parent and let the fitted equations carry the children: feasible."""
    x = np.array([1.0, 2.0, -3.0, 0.5])
    x_cf = x.copy()
    x_cf[NAMES.index("a")] += 1.0
    x_cf[NAMES.index("b")] = sem.predict(x_cf, "b")
    x_cf[NAMES.index("c")] = sem.predict(x_cf, "c")
    assert structural_feasibility_rate(x, x_cf, sem) == pytest.approx(1.0)


def test_root_only_intervention_is_feasible(sem):
    """d has no parents and no children: intervening on it is always feasible."""
    x = np.array([1.0, 2.0, -3.0, 0.5])
    x_cf = x.copy()
    x_cf[NAMES.index("d")] += 3.0
    assert structural_feasibility_rate(x, x_cf, sem) == pytest.approx(1.0)


def test_parent_moved_alone_is_violated(sem):
    """Moving a parent without updating a strongly-dependent child is unsupported."""
    x = np.array([1.0, 2.0, -3.0, 0.5])
    x_cf = x.copy()
    x_cf[NAMES.index("a")] += 3.0      # b should have followed, but does not
    assert per_node_report(x, x_cf, sem)["b"]["violated"] is True


def test_inherited_inconsistency_not_penalised(sem):
    """An anchor already off-manifold is not counted against the counterfactual."""
    x = np.array([1.0, 50.0, -3.0, 0.5])   # b wildly inconsistent with a already
    x_cf = x.copy()
    x_cf[NAMES.index("d")] += 1.0          # unrelated root edit
    report = per_node_report(x, x_cf, sem)
    assert report["b"]["violated"] is False


def test_feasibility_is_monotone_in_arbitrary_change(sem):
    """More arbitrary perturbation must not score as more feasible (no second peak)."""
    rng = np.random.default_rng(1)
    x = np.array([1.0, 2.0, -3.0, 0.5])
    rates = []
    for k in (0, 1, 2, 4):
        vals = []
        for _ in range(30):
            x_cf = x.copy()
            if k:
                idx = rng.choice(len(NAMES), size=k, replace=False)
                x_cf[idx] += rng.normal(scale=3.0, size=k)
            vals.append(structural_feasibility_rate(x, x_cf, sem))
        rates.append(float(np.mean(vals)))
    assert rates == sorted(rates, reverse=True), f"not monotone: {rates}"


def test_empty_dag_is_feasible():
    g = nx.DiGraph()
    g.add_nodes_from(NAMES)
    sem_empty = fit_structural_equations(_scm_sample(), g, NAMES)
    x = np.array([1.0, 2.0, -3.0, 0.5])
    assert structural_violation(x, x + 1.0, sem_empty) == pytest.approx(0.0)


def test_features_outside_dag_are_ignored(sem):
    x = np.array([1.0, 2.0, -3.0, 0.5])
    sem_sub = fit_structural_equations(_scm_sample(), _chain_dag(), NAMES)
    assert "d" not in sem_sub.non_root_nodes
    assert structural_feasibility_rate(x, x.copy(), sem_sub) == pytest.approx(1.0)
