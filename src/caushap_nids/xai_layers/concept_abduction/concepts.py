from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

# Core feature sets and activation directions for each concept.
# direction +1 → high value activates the concept (more anomalous in that direction).
# direction -1 → low value activates the concept.
_CONCEPT_SPECS: dict[str, dict] = {
    "ScanBehaviour": {
        "core_features": ["L4_DST_PORT", "FLOW_DURATION_MILLISECONDS", "IN_PKTS"],
        "directions":    [+1, -1, -1],
        "weights":       [3.0, 0.25, 0.5],
        "description":   "Port scan: unusual destination port, very short flow, few packets",
        "mitre_technique": "T1046",
    },
    "BruteForce": {
        "core_features": ["FTP_COMMAND_RET_CODE", "FLOW_DURATION_MILLISECONDS",
                          "IN_BYTES", "L4_DST_PORT"],
        "directions":    [+1, +1, +1, -1],
        "weights":       [0.25, 1.0, 1.0, 2.0],
        "description":   "Brute-force credential attack: failed-login evidence, repeated-session duration, increasing inbound bytes, on a well-known auth service port (low L4_DST_PORT) — discriminates against Bot C2 on high ports",
        "mitre_technique": "T1110",
    },
    "DataExfiltration": {
        "core_features": ["OUT_BYTES", "IN_BYTES", "L4_DST_PORT"],
        "directions":    [+1, -1, +1],
        "weights":       [1.0, 1.0, 1.0],
        "description":   "Data exfiltration: high outbound bytes, asymmetric I/O, unusual destination port",
        "mitre_technique": "T1041",
    },
}


@dataclass
class ConceptSubgraph:
    name: str
    core_features: list[str]
    directions: list[int]          # +1 / -1 per core feature
    weights: list[float]           # feature weights for the directed z-score aggregate
    dag_subgraph: nx.DiGraph
    description: str
    mitre_technique: str
    all_features: list[str] = field(default_factory=list)   # core + ancestors present in DAG


def build_concept_subgraphs(dag: nx.DiGraph) -> dict[str, ConceptSubgraph]:
    """
    Extract sub-DAGs from NF-DAG-v1 for each concept.
    Sub-DAG contains each core feature plus all its DAG ancestors, so causal
    context is preserved (e.g. PROTOCOL ancestors of L4_DST_PORT are included).
    """
    result: dict[str, ConceptSubgraph] = {}
    dag_nodes = set(dag.nodes())
    for name, spec in _CONCEPT_SPECS.items():
        core = spec["core_features"]
        subgraph_nodes: set[str] = set()
        for feat in core:
            if feat in dag_nodes:
                subgraph_nodes.add(feat)
                subgraph_nodes |= nx.ancestors(dag, feat)
        subgraph = dag.subgraph(subgraph_nodes).copy()
        all_features = sorted(subgraph_nodes)
        result[name] = ConceptSubgraph(
            name=name,
            core_features=core,
            directions=spec["directions"],
            weights=spec.get("weights", [1.0] * len(core)),
            dag_subgraph=subgraph,
            description=spec["description"],
            mitre_technique=spec["mitre_technique"],
            all_features=all_features,
        )
    return result
