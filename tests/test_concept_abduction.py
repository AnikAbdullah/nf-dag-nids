"""Tests for Module 5c — Concept-Based Abductive Explanations (Layer C)."""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from caushap_nids.xai_layers.concept_abduction import (
    ConceptSubgraph,
    build_concept_subgraphs,
    compute_background_stats,
    concept_activation_score,
    calibrate_thresholds,
    AbductiveExplanation,
    abductive_explanation,
    concept_fidelity,
    concept_rank_stability,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "FTP_COMMAND_RET_CODE",
]
N_FEATURES = len(FEATURE_NAMES)
_FI = {f: i for i, f in enumerate(FEATURE_NAMES)}

# Minimal DAG: just enough edges for concept subgraph extraction
@pytest.fixture
def dag():
    g = nx.DiGraph()
    g.add_nodes_from(FEATURE_NAMES)
    g.add_edge("PROTOCOL", "L4_DST_PORT")
    g.add_edge("L7_PROTO", "FTP_COMMAND_RET_CODE")
    g.add_edge("PROTOCOL", "IN_BYTES")
    return g


@pytest.fixture
def concept_library(dag):
    return build_concept_subgraphs(dag)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def benign_background(rng):
    """Synthetic benign flows: moderate values, low variance."""
    bg = rng.normal(loc=0.5, scale=0.1, size=(200, N_FEATURES))
    bg = np.clip(bg, 0.0, 1.0)
    # Benign: low FTP codes (around 220-230 mapped low), short out_bytes, normal duration
    bg[:, _FI["FTP_COMMAND_RET_CODE"]] = rng.normal(0.1, 0.05, 200).clip(0, 1)
    bg[:, _FI["OUT_BYTES"]] = rng.normal(0.1, 0.05, 200).clip(0, 1)
    return bg


@pytest.fixture
def bg_stats(benign_background):
    return compute_background_stats(benign_background)


@pytest.fixture
def thresholds(concept_library, benign_background, bg_stats):
    return calibrate_thresholds(concept_library, benign_background, FEATURE_NAMES, bg_stats, target_fpr=0.05)


# Mock detector for fidelity tests
class _MockDetector:
    """Returns L2 distance from origin as 'anomaly score'."""
    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.linalg.norm(X, axis=1)


# ── concepts.py tests ─────────────────────────────────────────────────────────

class TestBuildConceptSubgraphs:
    def test_returns_three_concepts(self, concept_library):
        assert set(concept_library.keys()) == {"ScanBehaviour", "BruteForce", "DataExfiltration"}

    def test_each_concept_is_concept_subgraph(self, concept_library):
        for c in concept_library.values():
            assert isinstance(c, ConceptSubgraph)

    def test_core_features_present(self, concept_library):
        assert "L4_DST_PORT" in concept_library["ScanBehaviour"].core_features
        assert "FTP_COMMAND_RET_CODE" in concept_library["BruteForce"].core_features
        assert "OUT_BYTES" in concept_library["DataExfiltration"].core_features

    def test_dag_subgraph_is_subgraph(self, concept_library, dag):
        for c in concept_library.values():
            for node in c.dag_subgraph.nodes():
                assert node in dag.nodes()

    def test_ancestors_included(self, concept_library):
        # PROTOCOL is an ancestor of L4_DST_PORT → must appear in ScanBehaviour subgraph
        assert "PROTOCOL" in concept_library["ScanBehaviour"].all_features

    def test_directions_length_matches_core(self, concept_library):
        for c in concept_library.values():
            assert len(c.directions) == len(c.core_features)

    def test_weights_length_matches_core(self, concept_library):
        for c in concept_library.values():
            assert len(c.weights) == len(c.core_features)

    def test_mitre_techniques_assigned(self, concept_library):
        assert concept_library["ScanBehaviour"].mitre_technique == "T1046"
        assert concept_library["BruteForce"].mitre_technique == "T1110"
        assert concept_library["DataExfiltration"].mitre_technique == "T1041"


# ── activation.py tests ───────────────────────────────────────────────────────

class TestComputeBackgroundStats:
    def test_returns_mean_std_keys(self, benign_background):
        stats = compute_background_stats(benign_background)
        assert "mean" in stats and "std" in stats

    def test_shapes_match_features(self, benign_background):
        stats = compute_background_stats(benign_background)
        assert stats["mean"].shape == (N_FEATURES,)
        assert stats["std"].shape == (N_FEATURES,)

    def test_std_nonzero(self, benign_background):
        stats = compute_background_stats(benign_background)
        assert (stats["std"] > 0).all()


class TestConceptActivationScore:
    def test_returns_float_in_unit_interval(self, concept_library, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        for concept in concept_library.values():
            score = concept_activation_score(x, concept, bg_stats, FEATURE_NAMES)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_high_ftp_code_activates_bruteforce(self, concept_library, bg_stats):
        x = np.full(N_FEATURES, 0.5)
        x[_FI["FTP_COMMAND_RET_CODE"]] = 5.0   # very high failure code → high z
        x[_FI["FLOW_DURATION_MILLISECONDS"]] = 5.0   # repeated-session duration
        x[_FI["IN_BYTES"]] = 5.0   # increasing inbound bytes
        score = concept_activation_score(x, concept_library["BruteForce"], bg_stats, FEATURE_NAMES)
        assert score > 0.5

    def test_high_out_bytes_activates_exfiltration(self, concept_library, bg_stats):
        x = np.full(N_FEATURES, 0.5)
        x[_FI["OUT_BYTES"]] = 5.0   # very high
        x[_FI["IN_BYTES"]] = -5.0   # very low (asymmetric)
        score = concept_activation_score(x, concept_library["DataExfiltration"], bg_stats, FEATURE_NAMES)
        assert score > 0.5

    def test_benign_flow_gives_low_scan_score(self, concept_library, bg_stats):
        x = np.full(N_FEATURES, 0.5)   # average flow
        score = concept_activation_score(x, concept_library["ScanBehaviour"], bg_stats, FEATURE_NAMES)
        assert score < 0.75   # should be near 0.5 (average), not strongly activated

    def test_missing_feature_graceful(self, concept_library, bg_stats):
        # Use a feature names list that omits FTP_COMMAND_RET_CODE
        short_names = [f for f in FEATURE_NAMES if f != "FTP_COMMAND_RET_CODE"]
        short_x = np.full(len(short_names), 0.5)
        short_stats = {"mean": np.full(len(short_names), 0.5), "std": np.full(len(short_names), 0.1)}
        score = concept_activation_score(short_x, concept_library["BruteForce"], short_stats, short_names)
        assert 0.0 <= score <= 1.0


class TestCalibrateThresholds:
    def test_returns_all_concepts(self, thresholds, concept_library):
        assert set(thresholds.keys()) == set(concept_library.keys())

    def test_thresholds_in_unit_interval(self, thresholds):
        for v in thresholds.values():
            assert 0.0 <= v <= 1.0

    def test_fpr_satisfied_on_background(self, concept_library, benign_background, bg_stats, thresholds):
        for name, concept in concept_library.items():
            scores = np.array([
                concept_activation_score(x, concept, bg_stats, FEATURE_NAMES)
                for x in benign_background
            ])
            fpr = (scores > thresholds[name]).mean()
            assert fpr <= 0.10, f"{name}: FPR={fpr:.3f} > 0.10"


# ── abduce.py tests ───────────────────────────────────────────────────────────

class TestAbductiveExplanation:
    def test_returns_abductive_explanation(self, concept_library, thresholds, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        result = abductive_explanation(x, concept_library, thresholds, bg_stats, FEATURE_NAMES)
        assert isinstance(result, AbductiveExplanation)

    def test_present_absent_partition_all_concepts(self, concept_library, thresholds, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        result = abductive_explanation(x, concept_library, thresholds, bg_stats, FEATURE_NAMES)
        all_named = set(result.present_concepts) | set(result.absent_concepts)
        assert all_named <= set(concept_library.keys())

    def test_activation_scores_all_concepts(self, concept_library, thresholds, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        result = abductive_explanation(x, concept_library, thresholds, bg_stats, FEATURE_NAMES)
        assert set(result.activation_scores.keys()) == set(concept_library.keys())

    def test_scores_in_unit_interval(self, concept_library, thresholds, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        result = abductive_explanation(x, concept_library, thresholds, bg_stats, FEATURE_NAMES)
        for v in result.activation_scores.values():
            assert 0.0 <= v <= 1.0

    def test_natural_language_is_string(self, concept_library, thresholds, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        result = abductive_explanation(x, concept_library, thresholds, bg_stats, FEATURE_NAMES)
        assert isinstance(result.natural_language, str)
        assert len(result.natural_language) > 0

    def test_scan_pattern_activates_scan(self, concept_library, bg_stats):
        # Craft a flow that strongly matches ScanBehaviour
        x = np.full(N_FEATURES, 0.5)
        x[_FI["L4_DST_PORT"]] = 5.0           # unusual port
        x[_FI["FLOW_DURATION_MILLISECONDS"]] = -5.0   # very short
        x[_FI["IN_PKTS"]] = -5.0              # few packets
        # Use very low thresholds to force activation
        low_thresholds = {n: 0.0 for n in concept_library}
        result = abductive_explanation(x, concept_library, low_thresholds, bg_stats, FEATURE_NAMES)
        assert "ScanBehaviour" in result.present_concepts

    def test_exfil_pattern_activates_exfil(self, concept_library, bg_stats):
        x = np.full(N_FEATURES, 0.5)
        x[_FI["OUT_BYTES"]] = 5.0
        x[_FI["IN_BYTES"]] = -5.0
        x[_FI["L4_DST_PORT"]] = 5.0
        low_thresholds = {n: 0.0 for n in concept_library}
        result = abductive_explanation(x, concept_library, low_thresholds, bg_stats, FEATURE_NAMES)
        assert "DataExfiltration" in result.present_concepts

    def test_top_k_limits_absent(self, concept_library, bg_stats):
        x = np.full(N_FEATURES, 0.5)
        high_thresholds = {n: 1.0 for n in concept_library}   # nothing present
        result = abductive_explanation(x, concept_library, high_thresholds, bg_stats, FEATURE_NAMES, top_k=2)
        assert len(result.absent_concepts) <= 2


# ── fidelity.py tests ─────────────────────────────────────────────────────────

class TestConceptFidelity:
    def test_returns_float_in_unit_interval(self, concept_library, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        detector = _MockDetector()
        low_thresholds = {n: 0.0 for n in concept_library}
        explanation = abductive_explanation(x, concept_library, low_thresholds, bg_stats, FEATURE_NAMES)
        score = concept_fidelity(detector, x, explanation, concept_library, FEATURE_NAMES, bg_stats)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_no_present_concepts_returns_one(self, concept_library, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        detector = _MockDetector()
        explanation = AbductiveExplanation(
            present_concepts=[], absent_concepts=list(concept_library.keys()),
            activation_scores={n: 0.0 for n in concept_library}, natural_language="n/a"
        )
        score = concept_fidelity(detector, x, explanation, concept_library, FEATURE_NAMES, bg_stats)
        assert score == 1.0

    def test_zero_orig_score_returns_one(self, concept_library, bg_stats):
        x = np.zeros(N_FEATURES)   # zero vector → _MockDetector score = 0

        class _ZeroDetector:
            def score(self, X): return np.zeros(len(X))

        low_thresholds = {n: 0.0 for n in concept_library}
        explanation = abductive_explanation(x, concept_library, low_thresholds, bg_stats, FEATURE_NAMES)
        score = concept_fidelity(_ZeroDetector(), x, explanation, concept_library, FEATURE_NAMES, bg_stats)
        assert score == 1.0


class TestConceptRankStability:
    def test_returns_float_in_valid_range(self, concept_library, dag, bg_stats, rng):
        x = rng.normal(0.5, 0.2, N_FEATURES)
        rho = concept_rank_stability(x, concept_library, bg_stats, FEATURE_NAMES, dag,
                                     n_perturbations=5, seed=42)
        assert isinstance(rho, float)
        assert -1.0 <= rho <= 1.0

    def test_identical_dag_gives_high_stability(self, concept_library, dag, bg_stats, rng):
        # With edge_drop_frac=0, DAG is unchanged → rankings identical → rho=1
        x = rng.normal(0.5, 0.2, N_FEATURES)
        rho = concept_rank_stability(x, concept_library, bg_stats, FEATURE_NAMES, dag,
                                     n_perturbations=5, edge_drop_frac=0.0, seed=42)
        assert rho >= 0.95
