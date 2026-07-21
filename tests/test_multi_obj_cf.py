"""Unit tests for Module 5b — Multi-Objective Causally-Constrained Counterfactuals."""
import numpy as np
import networkx as nx
import pytest

from caushap_nids.xai_layers.multi_obj_cf.objectives import (
    validity,
    proximity,
    sparsity,
    feasibility as feasibility_obj,
)
from caushap_nids.xai_layers.multi_obj_cf.feasibility import (
    dag_respecting_mutation,
    check_feasibility,
)
from caushap_nids.xai_layers.multi_obj_cf.nsga import (
    CounterfactualExplanation,
    _sparsify_valid_cf,
    generate_cf_pareto_front,
)
from caushap_nids.xai_layers.multi_obj_cf.pareto import dominates, hypervolume


# ── Shared fixtures ────────────────────────────────────────────────────────────

class _SimpleDetector:
    """score(X) = X[:,0] + 0.5*X[:,1]. Threshold=1.0: benign if score < 1.0."""

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X[:, 0] + 0.5 * X[:, 1]


THRESHOLD = 1.0


def _chain_dag(n: int = 4):
    """Linear chain DAG: f0 -> f1 -> f2 -> ... -> f(n-1)."""
    dag = nx.DiGraph()
    nodes = [f"f{i}" for i in range(n)]
    dag.add_nodes_from(nodes)
    for i in range(n - 1):
        dag.add_edge(nodes[i], nodes[i + 1])
    return dag, nodes


def _make_cf(
    validity_: float,
    proximity_: float,
    sparsity_: int,
    feasibility_rate: float,
) -> CounterfactualExplanation:
    return CounterfactualExplanation(
        x_orig=np.zeros(2),
        x_cf=np.zeros(2),
        validity=validity_,
        proximity=proximity_,
        sparsity=sparsity_,
        feasibility_rate=feasibility_rate,
    )


# ── objectives.py ──────────────────────────────────────────────────────────────

def test_validity_benign():
    det = _SimpleDetector()
    x_cf = np.array([0.0, 0.0, 0.0, 0.0])
    assert validity(x_cf, det, THRESHOLD) == 1.0


def test_validity_attack():
    det = _SimpleDetector()
    x_cf = np.array([2.0, 0.0, 0.0, 0.0])
    assert validity(x_cf, det, THRESHOLD) == 0.0


def test_proximity_l1():
    x = np.array([1.0, 2.0, 3.0])
    x_cf = np.array([2.0, 2.0, 1.0])
    assert proximity(x, x_cf) == pytest.approx(3.0)


def test_proximity_identical():
    x = np.array([1.0, 2.0])
    assert proximity(x, x) == pytest.approx(0.0)


def test_sparsity_count():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    x_cf = np.array([1.0, 3.0, 3.0, 5.0])
    assert sparsity(x, x_cf) == pytest.approx(2.0)


def test_sparsity_identical():
    x = np.array([1.0, 2.0])
    assert sparsity(x, x) == pytest.approx(0.0)


def test_feasibility_no_violation():
    dag, names = _chain_dag(3)
    x_orig = np.array([0.0, 0.0, 0.0])
    x_cf = np.array([1.0, 1.0, 1.0])   # every endpoint in the causal chain updated
    assert feasibility_obj(x_orig, x_cf, dag, names) == pytest.approx(0.0)


def test_feasibility_child_changed_without_parent():
    dag, names = _chain_dag(3)
    x_orig = np.array([0.0, 0.0, 0.0])
    x_cf = np.array([0.0, 1.0, 0.0])   # f1 changed, f0 (parent) unchanged → violation
    feas = feasibility_obj(x_orig, x_cf, dag, names)
    assert feas > 0.0


def test_feasibility_empty_dag():
    dag = nx.DiGraph()
    dag.add_nodes_from(["a", "b"])
    x = np.array([1.0, 2.0])
    assert feasibility_obj(x, x + 1.0, dag, ["a", "b"]) == pytest.approx(0.0)


def test_feasibility_full_violation():
    dag, names = _chain_dag(2)   # single edge f0->f1
    x_orig = np.zeros(2)
    x_cf = np.array([0.0, 1.0])  # only child changed → 1 of 1 edges violated
    assert feasibility_obj(x_orig, x_cf, dag, names) == pytest.approx(1.0)


def test_feasibility_unknown_features_skipped():
    dag = nx.DiGraph()
    dag.add_edge("a", "b")
    # feature_names doesn't contain 'a' or 'b' — no index found, defaults feasible
    x = np.array([0.0, 0.0])
    assert feasibility_obj(x, x + 1.0, dag, ["x", "y"]) == pytest.approx(0.0)


# ── feasibility.py ─────────────────────────────────────────────────────────────

def test_dag_mutation_returns_copy():
    dag, names = _chain_dag(4)
    x = np.zeros(4)
    x_mut = dag_respecting_mutation(x, dag, names, mutation_rate=1.0)
    assert x_mut is not x


def test_dag_mutation_ancestor_propagation():
    """Mutating f3 (leaf) must also touch at least one of its ancestors (f0, f1, f2)."""
    dag, names = _chain_dag(4)
    x = np.zeros(4)
    found = False
    for seed in range(30):
        rng = np.random.default_rng(seed)
        x_mut = dag_respecting_mutation(x, dag, names, mutation_rate=1.0, rng=rng)
        if abs(x_mut[3]) > 1e-9 and abs(x_mut[0]) > 1e-9:
            found = True
            break
    assert found, "Ancestor propagation never triggered across 30 random seeds"


def test_dag_mutation_zero_rate_unchanged():
    dag, names = _chain_dag(4)
    x = np.ones(4)
    x_mut = dag_respecting_mutation(x, dag, names, mutation_rate=0.0)
    np.testing.assert_array_equal(x_mut, x)


def test_check_feasibility_all_satisfied():
    dag, names = _chain_dag(3)
    x_orig = np.zeros(3)
    x_cf = np.array([1.0, 1.0, 1.0])
    result = check_feasibility(x_orig, x_cf, dag, names)
    assert all(result.values())


def test_check_feasibility_violation_detected():
    dag, names = _chain_dag(3)
    x_orig = np.zeros(3)
    x_cf = np.array([0.0, 1.0, 0.0])   # f1 changed, f0 not
    result = check_feasibility(x_orig, x_cf, dag, names)
    assert not result["f0->f1"]


def test_check_feasibility_parent_only_violation_detected():
    dag, names = _chain_dag(3)
    x_orig = np.zeros(3)
    x_cf = np.array([1.0, 0.0, 0.0])   # f0 changed, f1 not updated
    result = check_feasibility(x_orig, x_cf, dag, names)
    assert not result["f0->f1"]


def test_check_feasibility_returns_all_edges():
    dag, names = _chain_dag(4)
    x = np.zeros(4)
    result = check_feasibility(x, x, dag, names)
    assert len(result) == dag.number_of_edges()


# ── nsga.py ────────────────────────────────────────────────────────────────────

def test_generate_cf_pareto_front_returns_list():
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    x_attack = np.array([3.0, 2.0, 0.0, 0.0])
    cfs = generate_cf_pareto_front(
        det, dag, x_attack, names,
        population_size=20, n_generations=10,
        seed=42, bounds=(-1.0, 5.0), threshold=THRESHOLD,
    )
    assert isinstance(cfs, list)
    assert all(isinstance(cf, CounterfactualExplanation) for cf in cfs)


def test_generate_cf_pareto_front_has_valid_cfs():
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    x_attack = np.array([3.0, 2.0, 0.0, 0.0])
    cfs = generate_cf_pareto_front(
        det, dag, x_attack, names,
        population_size=30, n_generations=20,
        seed=42, bounds=(-1.0, 5.0), threshold=THRESHOLD,
    )
    valid_cfs = [cf for cf in cfs if cf.validity == 1.0]
    assert len(valid_cfs) > 0, "NSGA-II should find at least one valid (benign) CF"


def test_generate_cf_raises_without_threshold():
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    with pytest.raises(ValueError, match="threshold"):
        generate_cf_pareto_front(det, dag, np.array([2.0, 2.0, 0.0, 0.0]), names)


def test_sparsify_valid_cf_removes_unneeded_changes():
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    x_attack = np.array([3.0, 2.0, 0.0, 0.0])
    x_cf = np.array([-1.0, -1.0, 4.0, 4.0])

    sparse = _sparsify_valid_cf(x_attack, x_cf, det, dag, names, THRESHOLD)

    assert validity(sparse, det, THRESHOLD) == 1.0
    assert feasibility_obj(x_attack, sparse, dag, names) == pytest.approx(0.0)
    assert proximity(x_attack, sparse) < proximity(x_attack, x_cf)


def test_generate_cf_fields_populated():
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    cfs = generate_cf_pareto_front(
        det, dag, np.array([3.0, 2.0, 0.0, 0.0]), names,
        population_size=20, n_generations=10,
        seed=0, bounds=(-1.0, 5.0), threshold=THRESHOLD,
    )
    for cf in cfs:
        assert 0.0 <= cf.validity <= 1.0
        assert cf.proximity >= 0.0
        assert cf.sparsity >= 0
        assert 0.0 <= cf.feasibility_rate <= 1.0
        assert isinstance(cf.changed_features, list)


def test_generate_cf_respects_n_cfs_limit():
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    cfs = generate_cf_pareto_front(
        det, dag, np.array([3.0, 2.0, 0.0, 0.0]), names,
        population_size=30, n_generations=15,
        seed=0, bounds=(-1.0, 5.0), threshold=THRESHOLD,
        return_valid_only=True, n_cfs=2,
    )
    assert 0 < len(cfs) <= 2
    assert all(cf.validity == 1.0 for cf in cfs)


def test_pareto_front_non_dominated():
    """No CF in the returned Pareto front should be dominated by another CF in the same front."""
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    cfs = generate_cf_pareto_front(
        det, dag, np.array([3.0, 2.0, 0.0, 0.0]), names,
        population_size=30, n_generations=15,
        seed=0, bounds=(-1.0, 5.0), threshold=THRESHOLD,
    )
    for i, a in enumerate(cfs):
        for j, b in enumerate(cfs):
            if i != j:
                assert not dominates(b, a), (
                    f"CF {j} dominates CF {i} — Pareto front returned by NSGA-II is not non-dominated"
                )


# ── pareto.py ──────────────────────────────────────────────────────────────────

def test_dominates_clearly_better():
    a = _make_cf(1.0, 1.0, 2, 0.9)
    b = _make_cf(0.5, 2.0, 3, 0.5)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_incomparable():
    a = _make_cf(1.0, 1.0, 3, 0.5)
    b = _make_cf(0.9, 0.5, 2, 0.9)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_identical_is_false():
    a = _make_cf(1.0, 1.0, 2, 1.0)
    assert not dominates(a, a)


def test_hypervolume_positive_for_good_front():
    cfs = [
        _make_cf(1.0, 0.5, 2, 1.0),
        _make_cf(0.8, 0.3, 1, 0.9),
    ]
    hv = hypervolume(cfs)
    assert hv > 0.0


def test_hypervolume_empty():
    assert hypervolume([]) == 0.0


def test_hypervolume_dominated_front_smaller():
    good = [_make_cf(1.0, 0.5, 2, 1.0)]
    bad = [_make_cf(0.0, 5.0, 10, 0.0)]
    ref = np.array([2.0, 10.0, 15.0, 2.0])
    assert hypervolume(good, ref) > hypervolume(bad, ref)


# ── baselines/dice_baseline.py ─────────────────────────────────────────────────

from caushap_nids.xai_layers.multi_obj_cf.baselines.dice_baseline import (
    _SklearnDetectorWrapper,
    generate_dice_cfs,
    generate_scalarized_dice_cfs,
)


def _dice_background(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    benign = rng.normal(loc=0.0, scale=0.2, size=(n // 2, 4))
    attack = rng.normal(loc=2.0, scale=0.2, size=(n - n // 2, 4))
    return np.vstack([benign, attack]), rng.normal(loc=2.5, scale=0.1, size=4)


def test_scalarized_dice_returns_nonempty_cfs():
    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    bg, x_attack = _dice_background(64, seed=1)
    cfs = generate_scalarized_dice_cfs(
        det, bg, x_attack, names, dag, THRESHOLD,
        total_cfs=5, seed=42,
    )
    assert len(cfs) >= 1
    assert all(cf.validity in (0.0, 1.0) for cf in cfs)


def test_scalarized_dice_falls_back_to_random_method(monkeypatch):
    """When method='genetic' fails, scalarized DiCE must retry with method='random'."""
    import warnings as _warnings
    import caushap_nids.xai_layers.multi_obj_cf.baselines.dice_baseline as db

    dag, names = _chain_dag(4)
    det = _SimpleDetector()
    bg, x_attack = _dice_background(64, seed=2)

    real = db.generate_dice_cfs
    call_log: list[str] = []

    def _spy(detector, background, x_orig, feature_names, dag_, threshold, *,
             total_cfs, method, seed, allow_fallback):
        call_log.append(method)
        if method == "genetic":
            # Simulate the production warning that the real generate_dice_cfs
            # emits when DiCE returns 0 CFs.
            _warnings.warn(
                f"DiCE method={method!r} produced 0 CFs (empty cfs_df).",
                stacklevel=2,
            )
            return []
        return real(detector, background, x_orig, feature_names, dag_, threshold,
                    total_cfs=total_cfs, method=method, seed=seed,
                    allow_fallback=allow_fallback)

    monkeypatch.setattr(db, "generate_dice_cfs", _spy)

    with pytest.warns(UserWarning, match=r"method='genetic'"):
        cfs = db.generate_scalarized_dice_cfs(
            det, bg, x_attack, names, dag, THRESHOLD,
            total_cfs=5, seed=42,
        )

    assert call_log[0] == "genetic"
    assert "random" in call_log
    assert len(cfs) >= 1


def test_scalarized_dice_proba_has_gradient():
    """predict_proba must give a usable spread of probabilities — not flat ~0.5."""
    det = _SimpleDetector()
    wrapper = _SklearnDetectorWrapper(det, threshold=THRESHOLD)
    rng = np.random.default_rng(7)
    X = rng.normal(loc=1.0, scale=1.0, size=(64, 4))
    proba = wrapper.predict_proba(X)

    assert proba.shape == (64, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[:, 1].std() > 0.05, (
        f"predict_proba is too flat (std={proba[:, 1].std():.4f}); "
        "DiCE genetic optimizer needs gradient signal"
    )
