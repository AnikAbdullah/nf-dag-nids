import logging
import warnings
from dataclasses import dataclass, field

import networkx as nx

from .edge_types import EdgeType
from .validate import validate_dag

logger = logging.getLogger(__name__)


@dataclass
class AdjudicationProtocol:
    """Priority order: P > M > D.

    Uncertain edges are flagged but kept if >= 2 methods agree on the edge's
    existence (regardless of direction). Direction conflicts always defer to
    protocol semantics (P).
    """
    agreement_threshold: int = 2  # min methods that must agree to keep a data edge


def adjudicate(
    dag_v0: nx.DiGraph,
    pc_dag: nx.DiGraph,
    notears_dag: nx.DiGraph,
    protocol: AdjudicationProtocol | None = None,
) -> nx.DiGraph:
    """Reconcile expert DAG (v0) with data-driven outputs to produce NF-DAG-v1.

    Rules (CODING_PLAN §6):
    1. Edge in dag_v0 (type P or M) AND contradicted by BOTH data-driven methods
       → flag as EdgeType.U; keep; log warning.
    2. Edge absent from dag_v0 BUT present in BOTH data-driven methods
       → add as EdgeType.D if domain-plausible (no cycle created).
    3. Direction conflict: protocol semantics (P) always win over data evidence.

    Returns NF-DAG-v1 as a new DiGraph.
    """
    if protocol is None:
        protocol = AdjudicationProtocol()

    dag_v1: nx.DiGraph = dag_v0.copy()

    # ── Rule 1: flag P/M edges contradicted by both data-driven methods ──
    for src, dst, attrs in list(dag_v0.edges(data=True)):
        etype: EdgeType = attrs.get("edge_type", EdgeType.U)
        if etype not in (EdgeType.P, EdgeType.M):
            continue

        # "contradicted" = either reversed or absent in the data-driven graph
        pc_contra      = _contradicts(pc_dag, src, dst)
        notears_contra = _contradicts(notears_dag, src, dst)

        if pc_contra and notears_contra:
            warnings.warn(
                f"Expert edge {src}→{dst} ({etype.name}) contradicted by BOTH "
                f"PC and NOTEARS. Keeping edge; downgrading to Uncertain.",
                stacklevel=2,
            )
            dag_v1[src][dst]["edge_type"] = EdgeType.U
            dag_v1[src][dst]["adjudication_note"] = "Contradicted by PC + NOTEARS"
            logger.warning("Rule-1 downgrade: %s→%s (%s→U)", src, dst, etype.name)

    # ── Rule 2: add data-driven edges absent from expert DAG ─────────────
    all_data_edges: set[tuple[str, str]] = (
        set(pc_dag.edges()) | set(notears_dag.edges())
    )
    both_agree: set[tuple[str, str]] = (
        set(pc_dag.edges()) & set(notears_dag.edges())
    )

    for src, dst in both_agree:
        if dag_v1.has_edge(src, dst):
            continue

        if nx.has_path(dag_v1, dst, src):
            logger.debug("Skipping data edge %s→%s: would create cycle", src, dst)
            continue

        dag_v1.add_edge(
            src, dst,
            edge_type=EdgeType.D,
            justification="Added: both PC and NOTEARS agree on this edge",
        )
        logger.info("Rule-2 add: %s→%s (D)", src, dst)

    # ── Rule 3: direction conflicts — P always wins ───────────────────────
    for src, dst, attrs in list(dag_v1.edges(data=True)):
        etype = attrs.get("edge_type", EdgeType.U)
        if etype != EdgeType.P:
            continue

        # If data methods strongly suggest the reverse direction, log it
        if pc_dag.has_edge(dst, src) and notears_dag.has_edge(dst, src):
            logger.warning(
                "Direction conflict: P-edge %s→%s, but both data methods suggest %s→%s. "
                "Protocol semantics kept.",
                src, dst, dst, src,
            )
            dag_v1[src][dst]["adjudication_note"] = (
                f"Direction conflict: data suggests {dst}→{src}; P-edge wins"
            )

    validate_dag(dag_v1)
    return dag_v1


def _contradicts(data_dag: nx.DiGraph, src: str, dst: str) -> bool:
    """True if data_dag has the reverse edge OR does not have the forward edge."""
    return not data_dag.has_edge(src, dst)
