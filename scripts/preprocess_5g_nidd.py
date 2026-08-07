"""
Convert 5G-NIDD Combined.csv → data/5G-NIDD.parquet in NF-v2 format.

5G-NIDD uses Argus network flow features (not standard NetFlow/NF-v2).
Source and destination ports were explicitly removed by the dataset authors
for generality. This script maps available Argus fields to their closest
NF-v2 equivalents and fills unmappable features with 0.
The mapping is documented in paper §4 (IoT/5G generalization subsection).

Run from project root:
    python scripts/preprocess_5g_nidd.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).parent.parent
SRC_CSV  = PROJECT_ROOT / "data" / "Combined.csv"
OUT_PARQUET = PROJECT_ROOT / "data" / "5G-NIDD.parquet"

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

# Argus State string → TCP flags bitmask (approximate)
_STATE_TO_FLAGS: dict[str, int] = {
    "REQ": 0x02,   # SYN — connection request
    "CON": 0x18,   # ACK+PSH — established flow
    "FIN": 0x01,   # FIN
    "RST": 0x04,   # RST
    "ECO": 0x00,   # ICMP echo — no TCP flags
    "INT": 0x00,   # interrupted
    "TST": 0x00,   # test
}

# Argus Proto string → IANA protocol number
_PROTO_TO_INT: dict[str, int] = {
    "tcp": 6, "udp": 17, "icmp": 1, "sctp": 132,
    "igmp": 2, "esp": 50, "ah": 51, "gre": 47,
}


def _to_numeric(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def convert(src: Path, dst: Path) -> None:
    print(f"Reading {src} ...")
    df = pd.read_csv(src, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    out = pd.DataFrame(index=df.index)

    # ── Ports (not available — explicitly removed by dataset authors) ─────────
    out["L4_SRC_PORT"] = 0.0
    out["L4_DST_PORT"] = 0.0

    # ── Protocol ─────────────────────────────────────────────────────────────
    proto_str = df["Proto"].astype(str).str.lower().str.strip()
    out["PROTOCOL"] = proto_str.map(lambda p: _PROTO_TO_INT.get(p, 0)).astype(float)

    # ── L7 protocol (not available without ports) ─────────────────────────────
    out["L7_PROTO"] = 0.0

    # ── Byte / packet counts ──────────────────────────────────────────────────
    out["IN_BYTES"]  = _to_numeric(df["SrcBytes"])
    out["IN_PKTS"]   = _to_numeric(df["SrcPkts"])
    out["OUT_BYTES"] = _to_numeric(df["DstBytes"])
    out["OUT_PKTS"]  = _to_numeric(df["DstPkts"])

    # ── TCP flags (approximated from Argus State string) ──────────────────────
    state_str = df["State"].astype(str).str.strip().str.upper()
    tcp_flags = state_str.map(lambda s: _STATE_TO_FLAGS.get(s, 0)).astype(float)
    out["TCP_FLAGS"]        = tcp_flags
    out["CLIENT_TCP_FLAGS"] = tcp_flags
    out["SERVER_TCP_FLAGS"] = 0.0

    # ── Flow duration ─────────────────────────────────────────────────────────
    out["FLOW_DURATION_MILLISECONDS"] = _to_numeric(df["Dur"]) * 1000.0
    out["DURATION_IN"]  = 0.0
    out["DURATION_OUT"] = 0.0

    # ── TTL ───────────────────────────────────────────────────────────────────
    s_ttl = _to_numeric(df["sTtl"])
    d_ttl = _to_numeric(df["dTtl"])
    out["MIN_TTL"] = np.minimum(s_ttl, d_ttl)
    out["MAX_TTL"] = np.maximum(s_ttl, d_ttl)

    # ── Packet size features ──────────────────────────────────────────────────
    pkt_min = _to_numeric(df["Min"]).clip(lower=0)
    pkt_max = _to_numeric(df["Max"]).clip(lower=0)
    out["LONGEST_FLOW_PKT"]  = pkt_max
    out["SHORTEST_FLOW_PKT"] = pkt_min
    out["MIN_IP_PKT_LEN"]    = pkt_min
    out["MAX_IP_PKT_LEN"]    = pkt_max

    # ── Throughput / rate ─────────────────────────────────────────────────────
    # SrcRate / DstRate are in bytes/sec in Argus
    src_rate = _to_numeric(df["SrcRate"])
    dst_rate = _to_numeric(df["DstRate"])
    out["SRC_TO_DST_SECOND_BYTES"]    = src_rate
    out["DST_TO_SRC_SECOND_BYTES"]    = dst_rate
    out["SRC_TO_DST_AVG_THROUGHPUT"]  = src_rate
    out["DST_TO_SRC_AVG_THROUGHPUT"]  = dst_rate

    # ── Retransmissions (Argus Loss fields) ───────────────────────────────────
    # SrcLoss / DstLoss = bytes lost (closest proxy for retransmitted bytes)
    out["RETRANSMITTED_IN_BYTES"]  = _to_numeric(df["SrcLoss"])
    out["RETRANSMITTED_IN_PKTS"]   = 0.0   # packet-level loss not available
    out["RETRANSMITTED_OUT_BYTES"] = _to_numeric(df["DstLoss"])
    out["RETRANSMITTED_OUT_PKTS"]  = 0.0

    # ── Packet size buckets (not available — only mean sizes given) ───────────
    for col in [
        "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES",
    ]:
        out[col] = 0.0

    # ── TCP window sizes ──────────────────────────────────────────────────────
    out["TCP_WIN_MAX_IN"]  = _to_numeric(df["SrcWin"])
    out["TCP_WIN_MAX_OUT"] = _to_numeric(df["DstWin"])

    # ── ICMP ──────────────────────────────────────────────────────────────────
    out["ICMP_TYPE"]      = (proto_str == "icmp").astype(float)
    out["ICMP_IPV4_TYPE"] = 0.0

    # ── DNS / FTP (not available) ──────────────────────────────────────────────
    out["DNS_QUERY_ID"]        = 0.0
    out["DNS_QUERY_TYPE"]      = 0.0
    out["DNS_TTL_ANSWER"]      = 0.0
    out["FTP_COMMAND_RET_CODE"] = 0.0

    # ── Labels ────────────────────────────────────────────────────────────────
    out["label"]         = (df["Label"].astype(str) != "Benign").astype(int)
    out["attack_family"] = df["Attack Type"].astype(str)

    # ── Validate ──────────────────────────────────────────────────────────────
    missing = [f for f in NF_V2_FEATURES if f not in out.columns]
    if missing:
        raise RuntimeError(f"Missing NF-v2 features after mapping: {missing}")

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

    print("\nColumn mapping summary (for paper §4):")
    print("  Directly mapped : PROTOCOL, IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS,")
    print("                    FLOW_DURATION_MILLISECONDS, MIN_TTL, MAX_TTL,")
    print("                    LONGEST/SHORTEST_FLOW_PKT, MIN/MAX_IP_PKT_LEN,")
    print("                    SRC/DST_SECOND_BYTES, SRC/DST_AVG_THROUGHPUT,")
    print("                    RETRANSMITTED_IN/OUT_BYTES, TCP_WIN_MAX_IN/OUT,")
    print("                    TCP_FLAGS (from State string), ICMP_TYPE")
    print("  Set to zero     : L4_SRC_PORT, L4_DST_PORT (removed by authors),")
    print("                    L7_PROTO, CLIENT/SERVER_TCP_FLAGS, DURATION_IN/OUT,")
    print("                    RETRANSMITTED_IN/OUT_PKTS, NUM_PKTS_* buckets,")
    print("                    DNS_*, FTP_COMMAND_RET_CODE, ICMP_IPV4_TYPE")


if __name__ == "__main__":
    if not SRC_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SRC_CSV}")
    convert(SRC_CSV, OUT_PARQUET)
