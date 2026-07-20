import networkx as nx

from .constants import NF_V2_FEATURES
from .edge_types import EdgeType
from .validate import validate_dag

# Seed edges: (source, target, EdgeType, justification)
# RFC 3954/7011 protocol-derived (P) and MITRE ATT&CK-derived (M)
_SEED_EDGES: list[tuple[str, str, EdgeType, str]] = [
    # ── RFC 3954/7011 protocol semantics (P) ─────────────────────────────
    ("FLOW_DURATION_MILLISECONDS", "SRC_TO_DST_AVG_THROUGHPUT",  EdgeType.P,
     "Throughput = bytes / duration (RFC 3954 §5.8)"),
    ("FLOW_DURATION_MILLISECONDS", "DST_TO_SRC_AVG_THROUGHPUT",  EdgeType.P,
     "Throughput = bytes / duration (RFC 3954 §5.8)"),
    ("FLOW_DURATION_MILLISECONDS", "DURATION_IN",                EdgeType.P,
     "DURATION_IN <= FLOW_DURATION by definition"),
    ("FLOW_DURATION_MILLISECONDS", "DURATION_OUT",               EdgeType.P,
     "DURATION_OUT <= FLOW_DURATION by definition"),

    ("IN_BYTES",  "SRC_TO_DST_SECOND_BYTES",   EdgeType.P,
     "Per-second rate derived from total bytes and duration"),
    ("IN_BYTES",  "SRC_TO_DST_AVG_THROUGHPUT", EdgeType.P,
     "Throughput numerator is IN_BYTES"),
    ("IN_BYTES",  "RETRANSMITTED_IN_BYTES",     EdgeType.P,
     "Retransmitted bytes are a strict subset of total IN_BYTES"),
    ("IN_BYTES",  "NUM_PKTS_UP_TO_128_BYTES",   EdgeType.P,
     "Packet-size histogram bins are derived from the byte stream"),
    ("IN_BYTES",  "NUM_PKTS_128_TO_256_BYTES",  EdgeType.P,
     "Packet-size histogram bins are derived from the byte stream"),
    ("IN_BYTES",  "NUM_PKTS_256_TO_512_BYTES",  EdgeType.P,
     "Packet-size histogram bins are derived from the byte stream"),
    ("IN_BYTES",  "NUM_PKTS_512_TO_1024_BYTES", EdgeType.P,
     "Packet-size histogram bins are derived from the byte stream"),
    ("IN_BYTES",  "NUM_PKTS_1024_TO_1514_BYTES",EdgeType.P,
     "Packet-size histogram bins are derived from the byte stream"),

    ("OUT_BYTES", "DST_TO_SRC_SECOND_BYTES",    EdgeType.P,
     "Per-second rate derived from total bytes and duration"),
    ("OUT_BYTES", "DST_TO_SRC_AVG_THROUGHPUT",  EdgeType.P,
     "Throughput numerator is OUT_BYTES"),
    ("OUT_BYTES", "RETRANSMITTED_OUT_BYTES",     EdgeType.P,
     "Retransmitted bytes are a strict subset of total OUT_BYTES"),

    ("IN_PKTS",  "RETRANSMITTED_IN_PKTS",  EdgeType.P,
     "Retransmitted packets are a strict subset of total IN_PKTS"),
    ("IN_PKTS",  "LONGEST_FLOW_PKT",       EdgeType.P,
     "Max packet size is a property of the inbound packet sequence"),
    ("IN_PKTS",  "SHORTEST_FLOW_PKT",      EdgeType.P,
     "Min packet size is a property of the inbound packet sequence"),
    ("IN_PKTS",  "MIN_IP_PKT_LEN",         EdgeType.P,
     "IP packet length stats derived from the inbound packet sequence"),
    ("IN_PKTS",  "MAX_IP_PKT_LEN",         EdgeType.P,
     "IP packet length stats derived from the inbound packet sequence"),

    ("OUT_PKTS", "RETRANSMITTED_OUT_PKTS", EdgeType.P,
     "Retransmitted packets are a strict subset of total OUT_PKTS"),

    ("PROTOCOL", "TCP_FLAGS",              EdgeType.P,
     "TCP flags field only exists when PROTOCOL=6 (TCP)"),
    ("PROTOCOL", "MIN_TTL",               EdgeType.P,
     "TTL initial values differ by protocol (ICMP/TCP/UDP defaults differ)"),
    ("PROTOCOL", "MAX_TTL",               EdgeType.P,
     "TTL initial values differ by protocol"),
    ("PROTOCOL", "ICMP_TYPE",             EdgeType.M,
     "ICMP_TYPE only populated when PROTOCOL=1 (ICMP)"),
    ("PROTOCOL", "ICMP_IPV4_TYPE",        EdgeType.M,
     "ICMP_IPV4_TYPE only populated when PROTOCOL=1"),

    ("TCP_FLAGS", "CLIENT_TCP_FLAGS",     EdgeType.P,
     "CLIENT_TCP_FLAGS is a subset of the TCP_FLAGS aggregate"),
    ("TCP_FLAGS", "SERVER_TCP_FLAGS",     EdgeType.P,
     "SERVER_TCP_FLAGS is a subset of the TCP_FLAGS aggregate"),
    ("TCP_FLAGS", "TCP_WIN_MAX_IN",       EdgeType.P,
     "TCP window fields only exist when TCP_FLAGS != 0 (TCP traffic)"),
    ("TCP_FLAGS", "TCP_WIN_MAX_OUT",      EdgeType.P,
     "TCP window fields only exist when TCP_FLAGS != 0 (TCP traffic)"),

    # ── MITRE ATT&CK / application-layer semantics (M) ───────────────────
    ("PROTOCOL",    "L4_DST_PORT",        EdgeType.M,
     "Protocol constrains valid destination port ranges (RFC 793/768)"),
    ("L4_DST_PORT", "L7_PROTO",           EdgeType.M,
     "Well-known ports (80/443/21/53) strongly imply application protocol"),
    ("L4_SRC_PORT", "L7_PROTO",           EdgeType.M,
     "Ephemeral vs reserved source ports correlate with L7 protocol"),

    ("L7_PROTO", "DNS_QUERY_ID",          EdgeType.M,
     "DNS fields only populated for DNS traffic (L7_PROTO=DNS)"),
    ("L7_PROTO", "DNS_QUERY_TYPE",        EdgeType.M,
     "DNS fields only populated for DNS traffic"),
    ("L7_PROTO", "DNS_TTL_ANSWER",        EdgeType.M,
     "DNS TTL answer only populated for DNS traffic"),
    ("L7_PROTO", "FTP_COMMAND_RET_CODE",  EdgeType.M,
     "FTP return code only populated for FTP traffic (L7_PROTO=FTP); "
     "critical for T1110 Brute Force detection"),
]


def build_dag_v0(
    extra_edges: list[tuple[str, str, EdgeType, str]] | None = None,
) -> nx.DiGraph:
    """Build NF-DAG-v0 from RFC + MITRE knowledge sources.

    All 43 NF-v2 features are added as nodes. Every edge carries EdgeType and
    a justification string. The returned graph is validated to be a DAG.

    extra_edges: optional additional (src, dst, EdgeType, justification) tuples
                 from domain-expert elicitation workshops.
    """
    dag: nx.DiGraph = nx.DiGraph()
    dag.add_nodes_from(NF_V2_FEATURES)

    edges = list(_SEED_EDGES)
    if extra_edges:
        edges.extend(extra_edges)

    for src, dst, etype, justification in edges:
        if src not in dag.nodes:
            raise ValueError(f"Edge source '{src}' is not a recognised NF-v2 feature")
        if dst not in dag.nodes:
            raise ValueError(f"Edge target '{dst}' is not a recognised NF-v2 feature")
        dag.add_edge(src, dst, edge_type=etype, justification=justification)

    validate_dag(dag)
    return dag
