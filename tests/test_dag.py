"""Tests for Module 3: dag.

Coverage targets (CODING_PLAN §15):
- 100% line coverage on dag/validate.py
- DAG acyclicity and 43-feature coverage
- GraphML round-trip without edge loss
- Adjudication rules 1, 2, 3
"""

import networkx as nx
import pytest

from caushap_nids.dag.constants import NF_V2_FEATURES
from caushap_nids.dag.construct import build_dag_v0
from caushap_nids.dag.edge_types import EdgeType
from caushap_nids.dag.validate import validate_dag
from caushap_nids.dag.io import to_graphml, from_graphml
from caushap_nids.dag.adjudicate import AdjudicationProtocol, adjudicate


# ── build_dag_v0 / validate_dag ─────────────────────────────────────────────

class TestBuildDagV0:
    def test_is_dag(self):
        dag = build_dag_v0()
        assert nx.is_directed_acyclic_graph(dag)

    def test_all_nf_features_present(self):
        dag = build_dag_v0()
        for feat in NF_V2_FEATURES:
            assert feat in dag.nodes, f"Missing node: {feat}"

    def test_at_least_30_edges(self):
        dag = build_dag_v0()
        assert dag.number_of_edges() >= 30, (
            f"Only {dag.number_of_edges()} edges; plan requires >= 30"
        )

    def test_all_edges_have_edge_type(self):
        dag = build_dag_v0()
        for src, dst, attrs in dag.edges(data=True):
            assert "edge_type" in attrs, f"Missing edge_type on {src}→{dst}"
            assert isinstance(attrs["edge_type"], EdgeType)

    def test_all_edges_have_justification(self):
        dag = build_dag_v0()
        for src, dst, attrs in dag.edges(data=True):
            assert "justification" in attrs and attrs["justification"], (
                f"Missing justification on {src}→{dst}"
            )

    def test_extra_edges_added(self):
        extra = [("IN_BYTES", "OUT_BYTES", EdgeType.D, "test extra edge")]
        dag = build_dag_v0(extra_edges=extra)
        assert dag.has_edge("IN_BYTES", "OUT_BYTES")

    def test_invalid_feature_raises(self):
        with pytest.raises(ValueError, match="not a recognised"):
            build_dag_v0(extra_edges=[("NONEXISTENT_FEAT", "IN_BYTES", EdgeType.D, "x")])


class TestValidateDag:
    def test_cyclic_raises(self):
        dag = nx.DiGraph()
        dag.add_nodes_from(NF_V2_FEATURES)
        dag.add_edge("IN_BYTES", "OUT_BYTES")
        dag.add_edge("OUT_BYTES", "IN_BYTES")  # creates cycle
        with pytest.raises(ValueError, match="cycle"):
            validate_dag(dag)

    def test_missing_feature_raises(self):
        dag = nx.DiGraph()
        # Only add 40 of 41 features
        dag.add_nodes_from(NF_V2_FEATURES[:-1])
        with pytest.raises(ValueError, match="missing"):
            validate_dag(dag)

    def test_valid_dag_passes(self):
        dag = build_dag_v0()
        validate_dag(dag)  # must not raise


# ── GraphML round-trip ───────────────────────────────────────────────────────

class TestGraphMLIO:
    def test_round_trip_no_edge_loss(self, tmp_path):
        dag = build_dag_v0()
        path = tmp_path / "nf_dag_v1.graphml"
        to_graphml(dag, path)
        dag2 = from_graphml(path)
        assert set(dag.edges()) == set(dag2.edges())

    def test_round_trip_edge_types_restored(self, tmp_path):
        dag = build_dag_v0()
        path = tmp_path / "nf_dag_v1.graphml"
        to_graphml(dag, path)
        dag2 = from_graphml(path)
        for src, dst in dag.edges():
            orig  = dag[src][dst]["edge_type"]
            restored = dag2[src][dst]["edge_type"]
            assert isinstance(restored, EdgeType), (
                f"Edge {src}→{dst}: expected EdgeType, got {type(restored)}"
            )
            assert orig == restored

    def test_round_trip_node_coverage(self, tmp_path):
        dag = build_dag_v0()
        path = tmp_path / "nf_dag_v1.graphml"
        to_graphml(dag, path)
        dag2 = from_graphml(path)
        assert set(dag.nodes()) == set(dag2.nodes())


# ── adjudicate ───────────────────────────────────────────────────────────────

class TestAdjudicate:
    def _base_dags(self):
        dag_v0 = build_dag_v0()
        # Empty data-driven DAGs (no contradictions, no additions)
        pc_dag  = nx.DiGraph(); pc_dag.add_nodes_from(NF_V2_FEATURES)
        nt_dag  = nx.DiGraph(); nt_dag.add_nodes_from(NF_V2_FEATURES)
        return dag_v0, pc_dag, nt_dag

    def test_passthrough_with_empty_data_dags(self):
        dag_v0, pc_dag, nt_dag = self._base_dags()
        dag_v1 = adjudicate(dag_v0, pc_dag, nt_dag)
        assert set(dag_v0.edges()).issubset(set(dag_v1.edges()))

    def test_rule1_contradicted_edge_downgraded(self):
        dag_v0, pc_dag, nt_dag = self._base_dags()
        # Contradict a P-edge by putting the reverse in both data DAGs
        # (or just leaving it absent, which _contradicts() treats as contradiction)
        # Already empty — so all P/M edges are "contradicted" by both
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dag_v1 = adjudicate(dag_v0, pc_dag, nt_dag)
        # All P/M edges should be downgraded to U
        for src, dst, attrs in dag_v1.edges(data=True):
            orig = dag_v0[src][dst]["edge_type"] if dag_v0.has_edge(src, dst) else None
            if orig in (EdgeType.P, EdgeType.M):
                assert attrs["edge_type"] == EdgeType.U

    def test_rule2_data_edge_added(self):
        dag_v0, pc_dag, nt_dag = self._base_dags()
        # Add the same new edge to both data DAGs — should be added to v1
        pc_dag.add_edge("IN_BYTES", "OUT_BYTES",
                        edge_type=EdgeType.D, justification="test")
        nt_dag.add_edge("IN_BYTES", "OUT_BYTES",
                        edge_type=EdgeType.D, justification="test")
        dag_v1 = adjudicate(dag_v0, pc_dag, nt_dag)
        assert dag_v1.has_edge("IN_BYTES", "OUT_BYTES")

    def test_output_is_valid_dag(self):
        dag_v0, pc_dag, nt_dag = self._base_dags()
        dag_v1 = adjudicate(dag_v0, pc_dag, nt_dag)
        validate_dag(dag_v1)


# ── EdgeType ─────────────────────────────────────────────────────────────────

class TestEdgeType:
    def test_priority_order(self):
        assert EdgeType.P.priority > EdgeType.M.priority
        assert EdgeType.M.priority > EdgeType.D.priority
        assert EdgeType.D.priority > EdgeType.U.priority
