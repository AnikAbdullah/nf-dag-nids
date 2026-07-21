import networkx as nx
import numpy as np
import pytest

from caushap_nids.xai_layers.causal_shapley import (
    causal_shapley,
    cc_shapley_correction,
    decompose_effects,
    interventional_expectation,
    vanilla_kernel_shap,
)
from caushap_nids.xai_layers.causal_shapley.cache import (
    cached_subset_eval,
    clear_subset_cache,
    subset_cache_info,
)


class LinearDetector:
    def __init__(self, weights, intercept=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)

    def score(self, X):
        return X @ self.weights + self.intercept


def _empty_dag(n_features):
    dag = nx.DiGraph()
    dag.add_nodes_from([f"f{i}" for i in range(n_features)])
    return dag


def test_interventional_expectation_full_subset_is_detector_score():
    detector = LinearDetector([1.0, -2.0, 0.5])
    dag = _empty_dag(3)
    background = np.zeros((16, 3))
    x = np.array([2.0, 3.0, 4.0])

    value = interventional_expectation(
        x,
        frozenset({0, 1, 2}),
        dag,
        background,
        detector.score,
        n_samples=8,
    )

    assert value == pytest.approx(detector.score(x.reshape(1, -1))[0])


def test_causal_shapley_satisfies_efficiency_for_additive_score():
    detector = LinearDetector([1.0, -2.0, 0.5, 3.0], intercept=7.0)
    dag = _empty_dag(4)
    background = np.zeros((32, 4))
    x = np.array([2.0, 3.0, 4.0, 1.0])

    explanation = causal_shapley(detector, dag, x, background, n_samples=32)

    expected_phi = detector.weights * x
    expected_gap = detector.score(x.reshape(1, -1))[0] - detector.score(background).mean()
    assert explanation.phi == pytest.approx(expected_phi, abs=1e-6)
    assert explanation.phi.sum() == pytest.approx(expected_gap, abs=1e-6)
    assert explanation.direct_effects.shape == explanation.phi.shape
    assert explanation.indirect_effects.shape == explanation.phi.shape


def test_vanilla_kernel_shap_satisfies_efficiency_for_additive_score():
    detector = LinearDetector([1.0, -2.0, 0.5], intercept=3.0)
    background = np.zeros((32, 3))
    x = np.array([2.0, 3.0, 4.0])

    explanation = vanilla_kernel_shap(detector, x, background, n_samples=32)

    expected_gap = detector.score(x.reshape(1, -1))[0] - detector.score(background).mean()
    assert explanation.phi.sum() == pytest.approx(expected_gap, abs=1e-6)
    assert explanation.indirect_effects.sum() == pytest.approx(0.0, abs=1e-12)


def test_stability_smoothing_preserves_efficiency_and_changes_nonempty_dag():
    detector = LinearDetector([1.0, -2.0, 0.5, 3.0], intercept=7.0)
    dag = nx.DiGraph()
    dag.add_edges_from([("f0", "f1"), ("f1", "f2")])
    dag.add_node("f3")
    background = np.zeros((32, 4))
    x = np.array([2.0, 3.0, 4.0, 1.0])

    plain = causal_shapley(detector, dag, x, background, n_samples=32)
    smoothed = causal_shapley(
        detector,
        dag,
        x,
        background,
        n_samples=32,
        stability_smoothing=0.3,
    )

    expected_gap = detector.score(x.reshape(1, -1))[0] - detector.score(background).mean()
    assert smoothed.phi.sum() == pytest.approx(expected_gap, abs=1e-6)
    assert not np.allclose(smoothed.phi, plain.phi)


def test_asymmetric_method_closes_coalitions_over_ancestors():
    class DownstreamDetector:
        def score(self, X):
            return X[:, 2]

    dag = nx.DiGraph()
    dag.add_edges_from([("f0", "f1"), ("f1", "f2")])
    dag.add_nodes_from(["f0", "f1", "f2"])
    root = np.random.default_rng(7).normal(size=(64, 1))
    background = np.hstack([root, root, root])
    x = np.array([2.0, 2.0, 2.0])

    interventional = causal_shapley(
        DownstreamDetector(),
        dag,
        x,
        background,
        n_samples=16,
        causal_method="interventional",
    )
    asymmetric = causal_shapley(
        DownstreamDetector(),
        dag,
        x,
        background,
        n_samples=16,
        causal_method="asymmetric",
    )

    assert interventional.phi.sum() == pytest.approx(asymmetric.phi.sum(), abs=1e-6)
    assert not np.allclose(interventional.phi, asymmetric.phi)


def test_cc_shapley_correction_changes_collider_descendants_and_preserves_sum():
    dag = nx.DiGraph()
    dag.add_edges_from([("p0", "c"), ("p1", "c"), ("c", "d")])
    dag.add_nodes_from(["p0", "p1", "c", "d"])
    rng = np.random.default_rng(42)
    parents = rng.normal(size=(128, 2))
    collider = parents.sum(axis=1, keepdims=True) + rng.normal(scale=0.01, size=(128, 1))
    descendant = collider + rng.normal(scale=0.01, size=(128, 1))
    background = np.hstack([parents, collider, descendant])
    x = np.array([2.0, -1.0, 1.0, 1.5])
    phi = np.array([0.2, 0.1, 0.3, 1.0])

    corrected = cc_shapley_correction(
        phi,
        dag,
        x,
        background,
        lambda X: X.sum(axis=1),
        ["p0", "p1", "c", "d"],
    )

    assert corrected.sum() == pytest.approx(phi.sum(), abs=1e-12)
    assert abs(corrected[3] - phi[3]) / abs(phi[3]) >= 0.05


def test_decompose_effects_conserves_total_attribution():
    dag = nx.DiGraph()
    dag.add_edges_from([("a", "b"), ("b", "c")])
    dag.add_nodes_from(["a", "b", "c"])
    phi = np.array([1.0, 2.0, 3.0])

    direct, indirect = decompose_effects(phi, dag, ["a", "b", "c"])

    assert direct.shape == phi.shape
    assert indirect.shape == phi.shape
    assert direct.sum() + indirect.sum() == pytest.approx(phi.sum(), abs=1e-12)
    assert indirect[0] > 0


def test_cached_subset_eval_reuses_value():
    clear_subset_cache()
    calls = {"n": 0}

    def eval_fn(subset):
        calls["n"] += 1
        return float(len(subset))

    subset = frozenset({1, 2})
    assert cached_subset_eval(subset, (1.0, 2.0), "hash", eval_fn) == 2.0
    assert cached_subset_eval(subset, (1.0, 2.0), "hash", eval_fn) == 2.0
    assert calls["n"] == 1
    assert subset_cache_info()["subset_entries"] == 1


def test_invalid_causal_method_raises():
    with pytest.raises(ValueError, match="causal_method"):
        causal_shapley(
            LinearDetector([1.0]),
            _empty_dag(1),
            np.array([1.0]),
            np.zeros((4, 1)),
            causal_method="bad",
        )


def test_invalid_stability_smoothing_raises():
    with pytest.raises(ValueError, match="stability_smoothing"):
        causal_shapley(
            LinearDetector([1.0]),
            _empty_dag(1),
            np.array([1.0]),
            np.zeros((4, 1)),
            stability_smoothing=1.1,
        )
