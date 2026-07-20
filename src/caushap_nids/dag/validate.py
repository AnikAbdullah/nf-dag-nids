import networkx as nx

from .constants import NF_V2_FEATURES


def validate_dag(dag: nx.DiGraph) -> None:
    """Assert two invariants required by CODING_PLAN §6:

    1. Graph is a DAG (no cycles).
    2. All NF-v2 feature names are present as nodes.

    Raises ValueError on any violation.
    """
    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        raise ValueError(
            f"DAG contains {len(cycles)} cycle(s). First cycle: {cycles[0]}"
        )

    missing = [f for f in NF_V2_FEATURES if f not in dag.nodes]
    if missing:
        raise ValueError(
            f"DAG is missing {len(missing)} NF-v2 feature node(s): {missing}"
        )
