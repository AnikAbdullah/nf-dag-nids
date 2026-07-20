from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EdgeSupportSummary:
    total_edges: int
    supported_edges: int
    support_rate: float
    threshold: float
    status_counts: dict[str, int]

    @property
    def passed(self) -> bool:
        return self.support_rate >= self.threshold


@dataclass(frozen=True)
class ParquetInvariantSummary:
    rows_checked: int
    violations: dict[str, int]


REQUIRED_COLUMNS = {
    "source",
    "target",
    "edge_type",
    "evidence_source",
    "validation_method",
    "result",
    "status",
    "notes",
}


def load_edge_validation_rows(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the stratified edge-validation checklist."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
        rows = list(reader)

    bad_rows = [
        idx
        for idx, row in enumerate(rows, start=2)
        if not row.get("source") or not row.get("target") or not row.get("status")
    ]
    if bad_rows:
        raise ValueError(f"{path} has malformed row(s): {bad_rows}")
    return rows


def summarize_edge_support(
    rows: Iterable[dict[str, str]],
    threshold: float = 0.70,
) -> EdgeSupportSummary:
    rows = list(rows)
    total = len(rows)
    if total == 0:
        raise ValueError("edge-validation checklist is empty")

    status_counts = Counter(row["status"].strip().lower() for row in rows)
    supported = status_counts["supported"]
    return EdgeSupportSummary(
        total_edges=total,
        supported_edges=supported,
        support_rate=supported / total,
        threshold=threshold,
        status_counts=dict(status_counts),
    )


def run_nf_cic2018_invariant_checks(
    parquet_path: str | Path,
    batch_size: int = 250_000,
) -> ParquetInvariantSummary:
    """Run streaming invariant checks used by dag_validation_summary.md.

    The checks are intentionally limited to deterministic feature-derived
    relationships. They should not be used to promote attack-conditional edges.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet invariant checks") from exc

    columns = [
        "FLOW_DURATION_MILLISECONDS",
        "DURATION_IN",
        "DURATION_OUT",
        "IN_BYTES",
        "OUT_BYTES",
        "IN_PKTS",
        "OUT_PKTS",
        "RETRANSMITTED_IN_BYTES",
        "RETRANSMITTED_OUT_BYTES",
        "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_PKTS",
        "MIN_IP_PKT_LEN",
        "MAX_IP_PKT_LEN",
        "SHORTEST_FLOW_PKT",
        "LONGEST_FLOW_PKT",
        "PROTOCOL",
        "ICMP_TYPE",
        "ICMP_IPV4_TYPE",
        "TCP_FLAGS",
        "TCP_WIN_MAX_IN",
        "TCP_WIN_MAX_OUT",
    ]

    violations: Counter[str] = Counter()
    rows_checked = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        data = batch.to_pydict()
        rows_checked += len(data["PROTOCOL"])
        arr = {key: np.asarray(value, dtype="float64") for key, value in data.items()}
        eps = 1e-9

        violations["duration_in_gt_flow"] += int(
            np.sum(arr["DURATION_IN"] > arr["FLOW_DURATION_MILLISECONDS"] + eps)
        )
        violations["duration_out_gt_flow"] += int(
            np.sum(arr["DURATION_OUT"] > arr["FLOW_DURATION_MILLISECONDS"] + eps)
        )
        violations["ret_in_bytes_gt_in_bytes"] += int(
            np.sum(arr["RETRANSMITTED_IN_BYTES"] > arr["IN_BYTES"] + eps)
        )
        violations["ret_out_bytes_gt_out_bytes"] += int(
            np.sum(arr["RETRANSMITTED_OUT_BYTES"] > arr["OUT_BYTES"] + eps)
        )
        violations["ret_in_pkts_gt_in_pkts"] += int(
            np.sum(arr["RETRANSMITTED_IN_PKTS"] > arr["IN_PKTS"] + eps)
        )
        violations["ret_out_pkts_gt_out_pkts"] += int(
            np.sum(arr["RETRANSMITTED_OUT_PKTS"] > arr["OUT_PKTS"] + eps)
        )
        violations["min_ip_gt_max_ip"] += int(
            np.sum(arr["MIN_IP_PKT_LEN"] > arr["MAX_IP_PKT_LEN"] + eps)
        )
        violations["shortest_gt_longest"] += int(
            np.sum(arr["SHORTEST_FLOW_PKT"] > arr["LONGEST_FLOW_PKT"] + eps)
        )

        non_icmp = arr["PROTOCOL"] != 1
        violations["icmp_type_nonzero_when_not_icmp"] += int(
            np.sum(non_icmp & (arr["ICMP_TYPE"] != 0))
        )
        violations["icmp_ipv4_type_nonzero_when_not_icmp"] += int(
            np.sum(non_icmp & (arr["ICMP_IPV4_TYPE"] != 0))
        )

        flags_zero = arr["TCP_FLAGS"] == 0
        not_tcp = arr["PROTOCOL"] != 6
        violations["tcp_win_in_nonzero_when_flags_zero"] += int(
            np.sum(flags_zero & (arr["TCP_WIN_MAX_IN"] != 0))
        )
        violations["tcp_win_out_nonzero_when_flags_zero"] += int(
            np.sum(flags_zero & (arr["TCP_WIN_MAX_OUT"] != 0))
        )
        violations["tcp_win_in_nonzero_when_not_tcp"] += int(
            np.sum(not_tcp & (arr["TCP_WIN_MAX_IN"] != 0))
        )
        violations["tcp_win_out_nonzero_when_not_tcp"] += int(
            np.sum(not_tcp & (arr["TCP_WIN_MAX_OUT"] != 0))
        )

    return ParquetInvariantSummary(rows_checked=rows_checked, violations=dict(violations))


def format_edge_support_summary(summary: EdgeSupportSummary) -> str:
    status = "PASS" if summary.passed else "FAIL"
    lines = [
        f"Stratified edge-support gate: {status}",
        f"Supported edges: {summary.supported_edges}/{summary.total_edges}",
        f"Support rate: {summary.support_rate:.1%}",
        f"Threshold: {summary.threshold:.1%}",
        f"Status counts: {summary.status_counts}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the NF-DAG-v1 stratified gate.")
    parser.add_argument(
        "--checklist",
        default="edge_validation_checklist.csv",
        help="Path to the edge validation checklist CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum supported-edge fraction required to pass.",
    )
    parser.add_argument(
        "--parquet",
        help="Optional NF-CSE-CIC-IDS2018-v2 parquet path for invariant checks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
        help="Streaming parquet batch size.",
    )
    args = parser.parse_args()

    rows = load_edge_validation_rows(args.checklist)
    edge_summary = summarize_edge_support(rows, threshold=args.threshold)
    print(format_edge_support_summary(edge_summary))

    if args.parquet:
        invariant_summary = run_nf_cic2018_invariant_checks(
            args.parquet,
            batch_size=args.batch_size,
        )
        print(f"\nRows checked: {invariant_summary.rows_checked}")
        for name, count in sorted(invariant_summary.violations.items()):
            print(f"{name}: {count}")

    raise SystemExit(0 if edge_summary.passed else 1)


if __name__ == "__main__":
    main()
