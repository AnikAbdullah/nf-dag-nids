"""
Convert Edge-IIoTset ML-EdgeIIoT-dataset.csv → data/Edge-IIoTset.parquet
in NF-v2 format (41 features + label + attack_family).

Edge-IIoTset is a packet-level Wireshark dataset (63 cols), not a flow-level
NetFlow dataset. This script maps available packet-level fields to their
closest NF-v2 equivalents and fills unmappable features with 0.
The mapping is documented in paper §4 (IoT generalization subsection).

Run from project root:
    python scripts/preprocess_edge_iiotset.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).parent.parent
SRC_CSV = PROJECT_ROOT / "data" / "Edge-IIoTset dataset" / \
          "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv"
OUT_PARQUET = PROJECT_ROOT / "data" / "Edge-IIoTset.parquet"

NF_V2_FEATURES = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]

# L7_PROTO lookup by common dst port (matches NF-v2 nDPI codes approximately)
_PORT_TO_L7: dict[int, int] = {
    80: 7, 443: 91, 53: 5, 21: 1, 22: 92, 25: 3,
    110: 18, 143: 19, 8080: 7, 8443: 91,
}


def _to_numeric(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def convert(src: Path, dst: Path) -> None:
    print(f"Reading {src} ...")
    df = pd.read_csv(src, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    out = pd.DataFrame(index=df.index)

    # ── Ports ────────────────────────────────────────────────────────────────
    tcp_src = _to_numeric(df["tcp.srcport"])
    tcp_dst = _to_numeric(df["tcp.dstport"])
    udp_port = _to_numeric(df["udp.port"])

    # For UDP rows, tcp ports are 0; use udp.port as both src and dst
    out["L4_SRC_PORT"] = np.where(tcp_src > 0, tcp_src, udp_port)
    out["L4_DST_PORT"] = np.where(tcp_dst > 0, tcp_dst, udp_port)

    # ── Protocol ─────────────────────────────────────────────────────────────
    # 6=TCP  17=UDP  1=ICMP  0=other
    tcp_len = _to_numeric(df["tcp.len"])
    icmp_check = _to_numeric(df["icmp.checksum"])
    is_tcp  = (tcp_len > 0) | (_to_numeric(df["tcp.connection.syn"]) > 0)
    is_udp  = (udp_port > 0) & ~is_tcp
    is_icmp = (icmp_check > 0) & ~is_tcp & ~is_udp

    out["PROTOCOL"] = np.select(
        [is_tcp, is_udp, is_icmp],
        [6,      17,     1     ],
        default=0,
    ).astype(float)

    # ── L7 protocol (inferred from dst port) ─────────────────────────────────
    out["L7_PROTO"] = out["L4_DST_PORT"].map(
        lambda p: _PORT_TO_L7.get(int(p), 0) if pd.notna(p) else 0
    ).astype(float)

    # ── Byte / packet counts (packet-level → treat each row as 1 packet) ────
    out["IN_BYTES"]  = tcp_len
    out["IN_PKTS"]   = 1.0
    out["OUT_BYTES"] = 0.0
    out["OUT_PKTS"]  = 0.0

    # ── TCP flags ────────────────────────────────────────────────────────────
    # tcp.flags is a hex string like "0x0002"; convert to int
    tcp_flags_raw = df["tcp.flags"].astype(str).str.replace("0x", "", regex=False)
    tcp_flags_int = pd.to_numeric(tcp_flags_raw, errors="coerce").fillna(0)
    out["TCP_FLAGS"]        = tcp_flags_int
    out["CLIENT_TCP_FLAGS"] = tcp_flags_int
    out["SERVER_TCP_FLAGS"] = 0.0

    # ── Flow duration (UDP time delta in seconds → ms) ───────────────────────
    udp_td = _to_numeric(df["udp.time_delta"])
    out["FLOW_DURATION_MILLISECONDS"] = udp_td * 1000.0
    out["DURATION_IN"]  = 0.0
    out["DURATION_OUT"] = 0.0

    # ── TTL (not in dataset) ──────────────────────────────────────────────────
    out["MIN_TTL"] = 0.0
    out["MAX_TTL"] = 0.0

    # ── Packet size features ──────────────────────────────────────────────────
    pkt_len = tcp_len.clip(lower=0)
    out["LONGEST_FLOW_PKT"]  = pkt_len
    out["SHORTEST_FLOW_PKT"] = pkt_len
    out["MIN_IP_PKT_LEN"]    = pkt_len
    out["MAX_IP_PKT_LEN"]    = pkt_len

    # ── Throughput / retransmit (not available at packet level) ──────────────
    for col in [
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    ]:
        out[col] = 0.0

    # ── Packet size buckets ───────────────────────────────────────────────────
    out["NUM_PKTS_UP_TO_128_BYTES"]    = (pkt_len <= 128).astype(float)
    out["NUM_PKTS_128_TO_256_BYTES"]   = ((pkt_len > 128)  & (pkt_len <= 256)).astype(float)
    out["NUM_PKTS_256_TO_512_BYTES"]   = ((pkt_len > 256)  & (pkt_len <= 512)).astype(float)
    out["NUM_PKTS_512_TO_1024_BYTES"]  = ((pkt_len > 512)  & (pkt_len <= 1024)).astype(float)
    out["NUM_PKTS_1024_TO_1514_BYTES"] = ((pkt_len > 1024) & (pkt_len <= 1514)).astype(float)

    # ── ICMP ─────────────────────────────────────────────────────────────────
    out["ICMP_TYPE"]      = (icmp_check > 0).astype(float)
    out["ICMP_IPV4_TYPE"] = 0.0

    # ── DNS ──────────────────────────────────────────────────────────────────
    out["DNS_QUERY_ID"]   = 0.0
    out["DNS_QUERY_TYPE"] = _to_numeric(df["dns.qry.type"])
    out["DNS_TTL_ANSWER"] = 0.0

    # ── FTP (not available) ───────────────────────────────────────────────────
    out["FTP_COMMAND_RET_CODE"] = 0.0

    # ── Labels ────────────────────────────────────────────────────────────────
    out["label"]         = _to_numeric(df["Attack_label"]).astype(int)
    out["attack_family"] = df["Attack_type"].astype(str)

    # ── Validate all NF-v2 features present ──────────────────────────────────
    missing = [f for f in NF_V2_FEATURES if f not in out.columns]
    if missing:
        raise RuntimeError(f"Missing NF-v2 features after mapping: {missing}")

    # Cast all feature columns to float32
    for col in NF_V2_FEATURES:
        out[col] = out[col].astype("float32")

    final_cols = NF_V2_FEATURES + ["label", "attack_family"]
    out = out[final_cols]

    print(f"  Mapped to {len(NF_V2_FEATURES)} NF-v2 features + label + attack_family")
    print(f"  Benign: {(out['label']==0).sum():,}  Attack: {(out['label']==1).sum():,}")
    print(f"  Attack families: {sorted(out['attack_family'].unique())}")

    print(f"Writing {dst} ...")
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, dst, compression="snappy")
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  Done — {size_mb:.1f} MB")

    # Print mapping summary for paper §4
    print("\nColumn mapping summary (for paper §4):")
    print("  Directly mapped : L4_SRC_PORT, L4_DST_PORT, PROTOCOL, IN_BYTES,")
    print("                    TCP_FLAGS, CLIENT_TCP_FLAGS, DNS_QUERY_TYPE,")
    print("                    ICMP_TYPE, FLOW_DURATION_MILLISECONDS (UDP only),")
    print("                    NUM_PKTS_* (from tcp.len bucket), L7_PROTO (port-inferred)")
    print("  Set to zero     : OUT_BYTES, OUT_PKTS, TTL fields, throughput fields,")
    print("                    retransmit fields, WIN fields, DNS_TTL_ANSWER,")
    print("                    FTP_COMMAND_RET_CODE, SERVER_TCP_FLAGS")


if __name__ == "__main__":
    if not SRC_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SRC_CSV}")
    convert(SRC_CSV, OUT_PARQUET)
