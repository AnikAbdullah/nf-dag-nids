from pathlib import Path

import networkx as nx

from .edge_types import EdgeType


def to_graphml(dag: nx.DiGraph, path: str | Path) -> None:
    """Serialise NF-DAG-v1 to GraphML.

    Each edge carries: source, target, edge_type (string), justification.
    Output is readable by Gephi and Cytoscape.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dag_out = dag.copy()
    for src, dst, attrs in dag_out.edges(data=True):
        etype = attrs.get("edge_type")
        if isinstance(etype, EdgeType):
            dag_out[src][dst]["edge_type"] = etype.value
        if "edge_type" not in dag_out[src][dst]:
            dag_out[src][dst]["edge_type"] = EdgeType.U.value
        if "justification" not in dag_out[src][dst]:
            dag_out[src][dst]["justification"] = ""

    nx.write_graphml(dag_out, str(path))


def from_graphml(path: str | Path) -> nx.DiGraph:
    """Load NF-DAG-v1 from GraphML. Restores EdgeType enum from string attributes."""
    dag = nx.read_graphml(str(path))

    _value_to_etype = {e.value: e for e in EdgeType}
    for src, dst in list(dag.edges()):
        raw = dag[src][dst].get("edge_type", "uncertain")
        dag[src][dst]["edge_type"] = _value_to_etype.get(raw, EdgeType.U)

    return dag
