"""Unit tests for Module 7 — experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from caushap_nids.experiments.seed_management import derive_seed, set_all_seeds
from caushap_nids.experiments.runner import (
    CONFIG_NAMES,
    DATASETS,
    DEFAULT_SEEDS,
    RunResult,
    _EnsembleScoreAdapter,
    load_all_results,
    _random_rewired_dag,
    _select_explanation_indices,
)


# ── seed_management ──────────────────────────────────────────────────────────

class TestSetAllSeeds:
    def test_returns_rng(self):
        rng = set_all_seeds(42)
        assert isinstance(rng, np.random.Generator)

    def test_numpy_deterministic(self):
        set_all_seeds(42)
        a = np.random.randint(0, 1000)
        set_all_seeds(42)
        b = np.random.randint(0, 1000)
        assert a == b

    def test_different_seeds_differ(self):
        rng1 = set_all_seeds(42)
        rng2 = set_all_seeds(99)
        a = rng1.integers(0, 10_000)
        b = rng2.integers(0, 10_000)
        assert a != b

    def test_python_random_seeded(self):
        import random
        set_all_seeds(42)
        a = random.random()
        set_all_seeds(42)
        b = random.random()
        assert a == b


class TestDeriveSeed:
    def test_deterministic(self):
        s1 = derive_seed(42, "A4_full", "nf_cic2018")
        s2 = derive_seed(42, "A4_full", "nf_cic2018")
        assert s1 == s2

    def test_different_configs_differ(self):
        s1 = derive_seed(42, "A0_baseline")
        s2 = derive_seed(42, "A4_full")
        assert s1 != s2

    def test_returns_non_negative_int(self):
        s = derive_seed(42, "test")
        assert isinstance(s, int)
        assert s >= 0

    def test_integer_tag(self):
        s1 = derive_seed(42, 0)
        s2 = derive_seed(42, 1)
        assert s1 != s2


# ── RunResult ────────────────────────────────────────────────────────────────

class TestRunResult:
    def _make_result(self, status="ok", error=""):
        return RunResult(
            config_name="A4_full",
            dataset="nf_cic2018",
            seed=42,
            elapsed_seconds=1.5,
            detection={"macro_f1": (0.92, 0.90, 0.94)},
            faithfulness={"sufficiency": 0.88},
            cf_metrics={"validity": 0.95},
            concept_metrics={"mean_fidelity": 0.93},
            status=status,
            error=error,
        )

    def test_to_dict_has_required_keys(self):
        r = self._make_result()
        d = r.to_dict()
        assert "config_name" in d
        assert "dataset" in d
        assert "seed" in d
        assert "detection" in d
        assert "status" in d

    def test_save_and_reload(self, tmp_path):
        r = self._make_result()
        out = tmp_path / "result.json"
        r.save(out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["config_name"] == "A4_full"
        assert loaded["seed"] == 42
        assert loaded["detection"]["macro_f1"] == [0.92, 0.90, 0.94]

    def test_save_creates_parent_dirs(self, tmp_path):
        r = self._make_result()
        out = tmp_path / "a" / "b" / "c" / "result.json"
        r.save(out)
        assert out.exists()

    def test_error_result(self):
        r = self._make_result(status="error", error="dataset not found")
        assert r.status == "error"
        assert "not found" in r.error


# ── load_all_results ──────────────────────────────────────────────────────────

class TestLoadAllResults:
    def test_loads_saved_results(self, tmp_path):
        r = RunResult("A0_baseline", "nf_cic2018", 42, 2.0,
                      detection={"macro_f1": (0.8, 0.78, 0.82)})
        r.save(tmp_path / "results" / "A0_baseline" / "nf_cic2018" / "42" / "result.json")

        loaded = load_all_results(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].config_name == "A0_baseline"

    def test_loads_multiple(self, tmp_path):
        for cfg in ("A0_baseline", "A4_full"):
            r = RunResult(cfg, "nf_cic2018", 42, 1.0, detection={})
            r.save(tmp_path / "results" / cfg / "nf_cic2018" / "42" / "result.json")
        loaded = load_all_results(tmp_path)
        assert len(loaded) == 2

    def test_empty_dir_returns_empty_list(self, tmp_path):
        (tmp_path / "results").mkdir()
        assert load_all_results(tmp_path) == []

    def test_filters_by_criteria_version_and_n_explain(self, tmp_path):
        fresh = RunResult(
            "A0_baseline",
            "nf_cic2018",
            42,
            1.0,
            detection={},
            metadata={"criteria_version": "v2", "n_explain": 200},
        )
        stale = RunResult(
            "A1_stl_only",
            "nf_cic2018",
            42,
            1.0,
            detection={},
            metadata={"criteria_version": "v1", "n_explain": 200},
        )
        smoke = RunResult(
            "A4_full",
            "nf_cic2018",
            42,
            1.0,
            detection={},
            metadata={"criteria_version": "v2", "n_explain": 1},
        )
        fresh.save(tmp_path / "results" / "A0_baseline" / "nf_cic2018" / "42" / "result.json")
        stale.save(tmp_path / "results" / "A1_stl_only" / "nf_cic2018" / "42" / "result.json")
        smoke.save(tmp_path / "results" / "A4_full" / "nf_cic2018" / "42" / "result.json")

        loaded = load_all_results(tmp_path, criteria_version="v2", n_explain=200)
        assert [r.config_name for r in loaded] == ["A0_baseline"]


# ── XAI run selection / random DAG ablation ──────────────────────────────────

class _LinearScore:
    def __init__(self, scale):
        self.scale = scale

    def score(self, X):
        return np.asarray(X).sum(axis=1) * self.scale


class _IdentityNorm:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def test_ensemble_score_adapter_matches_geometric_ensemble():
    from caushap_nids.models import geometric_ensemble

    X = np.array([[0.2, 0.3], [0.4, 0.1]])
    adapter = _EnsembleScoreAdapter(
        ae=_LinearScore(0.5),
        if_det=_LinearScore(0.25),
        ae_norm=_IdentityNorm(),
        if_norm=_IdentityNorm(),
        alpha=0.4,
    )

    expected = geometric_ensemble(
        np.array([0.25, 0.25]),
        np.array([0.125, 0.125]),
        alpha=0.4,
    )
    np.testing.assert_allclose(adapter.score(X), expected)


def test_n_explain_zero_selects_all_predicted_anomalies():
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    rng = np.random.default_rng(42)

    selected = _select_explanation_indices(y_pred, rng, n_explain=0)

    assert selected.tolist() == [1, 3, 4]


def test_n_explain_positive_subsamples_predicted_anomalies():
    y_pred = np.array([0, 1, 1, 1, 1, 0])
    rng = np.random.default_rng(42)

    selected = _select_explanation_indices(y_pred, rng, n_explain=2)

    assert len(selected) == 2
    assert set(selected).issubset({1, 2, 3, 4})


def test_random_rewired_dag_preserves_nodes_edges_and_acyclicity():
    source = nx.DiGraph()
    source.add_edges_from([("a", "b"), ("b", "c"), ("a", "d")])
    source.add_node("e")

    randomized = _random_rewired_dag(source, seed=42)

    assert set(randomized.nodes()) == set(source.nodes())
    assert randomized.number_of_edges() == source.number_of_edges()
    assert nx.is_directed_acyclic_graph(randomized)


# ── constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_config_names_count(self):
        assert len(CONFIG_NAMES) == 6

    def test_datasets_count(self):
        assert len(DATASETS) == 4

    def test_seeds_count(self):
        assert len(DEFAULT_SEEDS) == 3

    def test_ablation_matrix_size(self):
        assert len(CONFIG_NAMES) * len(DATASETS) * len(DEFAULT_SEEDS) == 72

    def test_all_expected_configs_present(self):
        expected = {"A0_baseline", "A1_stl_only", "A2_causal_shap",
                    "A3_moocf", "A4_full", "A5_random_dag"}
        assert set(CONFIG_NAMES) == expected

    def test_all_expected_datasets_present(self):
        expected = {"nf_cic2018", "nf_unsw15", "edge_iiotset", "5g_nidd"}
        assert set(DATASETS) == expected


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_parser_defaults(self):
        from caushap_nids.experiments.cli import _build_parser
        args = _build_parser().parse_args([])
        assert args.config is None
        assert args.dataset is None
        assert not args.all
        assert args.n_explain == 200

    def test_parser_single_run(self):
        from caushap_nids.experiments.cli import _build_parser
        args = _build_parser().parse_args(
            ["--config", "A4_full", "--dataset", "nf_cic2018", "--seeds", "42"]
        )
        assert args.config == ["A4_full"]
        assert args.dataset == ["nf_cic2018"]
        assert args.seeds == [42]

    def test_parser_all_flag(self):
        from caushap_nids.experiments.cli import _build_parser
        args = _build_parser().parse_args(["--all"])
        assert args.all

    def test_missing_data_dir_exits_2(self, tmp_path):
        from caushap_nids.experiments.cli import main
        code = main([
            "--config", "A4_full",
            "--dataset", "nf_cic2018",
            "--seeds", "42",
            "--data-dir", str(tmp_path / "nonexistent"),
        ])
        assert code == 2

    def test_zero_n_explain_passes_through_to_runner(self, tmp_path):
        from caushap_nids.experiments.cli import main

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = RunResult(
            "A4_full",
            "nf_cic2018",
            42,
            1.0,
            detection={"macro_f1": (0.9, 0.88, 0.92)},
        )

        with patch("caushap_nids.experiments.cli.run_single", return_value=result) as run_mock:
            code = main([
                "--config", "A4_full",
                "--dataset", "nf_cic2018",
                "--seeds", "42",
                "--data-dir", str(data_dir),
                "--n-explain", "0",
            ])

        assert code == 0
        assert run_mock.call_args.kwargs["n_explain"] == 0
