"""Graph-independent physical plausibility for NF-v2 counterfactuals.

Every criterion here is a deterministic property of the NetFlow-v2 export format
itself -- a flow cannot retransmit more bytes than it carried, a direction cannot
last longer than the flow containing it -- so a counterfactual violating one is
physically impossible whatever causal graph produced it.

That independence is the point.  DAG-based feasibility cannot arbitrate between
two candidate graphs, because each configuration is scored against the very
structure under test.  These invariants are fixed across configurations, so they
give the random-DAG ablation a metric that does not move with the graph.

Invariants are stated over RAW NF-v2 values.  Counterfactual search runs in the
preprocessed space, so use :class:`FeatureTransform` to map back first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Index into NF_V2_FEATURE_COLS, resolved lazily so this module stays importable
# without the data pipeline.
_F: dict[str, int] = {}


def _idx(feature_names: list[str]) -> dict[str, int]:
    return {f: i for i, f in enumerate(feature_names)}


@dataclass(frozen=True)
class Invariant:
    """One deterministic NF-v2 constraint, checked on a raw feature vector."""

    name: str
    description: str
    check: Callable[[np.ndarray, dict[str, int]], bool]


def _le(a: str, b: str) -> Callable[[np.ndarray, dict[str, int]], bool]:
    """a <= b, with a small relative tolerance for float round-trips."""

    def _f(x: np.ndarray, ix: dict[str, int]) -> bool:
        if a not in ix or b not in ix:
            return True
        va, vb = float(x[ix[a]]), float(x[ix[b]])
        return va <= vb + 1e-6 * max(1.0, abs(vb))

    return _f


def _nonneg(a: str) -> Callable[[np.ndarray, dict[str, int]], bool]:
    def _f(x: np.ndarray, ix: dict[str, int]) -> bool:
        return True if a not in ix else float(x[ix[a]]) >= -1e-6

    return _f


def _bins_within_packets(x: np.ndarray, ix: dict[str, int]) -> bool:
    bins = [
        "NUM_PKTS_UP_TO_128_BYTES",
        "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES",
    ]
    if not all(b in ix for b in bins):
        return True
    if "IN_PKTS" not in ix or "OUT_PKTS" not in ix:
        return True
    total = float(x[ix["IN_PKTS"]]) + float(x[ix["OUT_PKTS"]])
    return sum(float(x[ix[b]]) for b in bins) <= total + 1e-6 * max(1.0, total)


#: Candidate invariants. Verify against real traffic before relying on one --
#: `scripts/verify_nf_v2_invariants.py` reports the per-invariant violation rate
#: and is the basis for which of these are cited as exact.
CANDIDATE_INVARIANTS: tuple[Invariant, ...] = (
    Invariant("duration_in_le_flow", "DURATION_IN <= FLOW_DURATION_MILLISECONDS",
              _le("DURATION_IN", "FLOW_DURATION_MILLISECONDS")),
    Invariant("duration_out_le_flow", "DURATION_OUT <= FLOW_DURATION_MILLISECONDS",
              _le("DURATION_OUT", "FLOW_DURATION_MILLISECONDS")),
    Invariant("retrans_in_bytes_le_in_bytes", "RETRANSMITTED_IN_BYTES <= IN_BYTES",
              _le("RETRANSMITTED_IN_BYTES", "IN_BYTES")),
    Invariant("retrans_out_bytes_le_out_bytes", "RETRANSMITTED_OUT_BYTES <= OUT_BYTES",
              _le("RETRANSMITTED_OUT_BYTES", "OUT_BYTES")),
    Invariant("retrans_in_pkts_le_in_pkts", "RETRANSMITTED_IN_PKTS <= IN_PKTS",
              _le("RETRANSMITTED_IN_PKTS", "IN_PKTS")),
    Invariant("retrans_out_pkts_le_out_pkts", "RETRANSMITTED_OUT_PKTS <= OUT_PKTS",
              _le("RETRANSMITTED_OUT_PKTS", "OUT_PKTS")),
    Invariant("shortest_le_longest_pkt", "SHORTEST_FLOW_PKT <= LONGEST_FLOW_PKT",
              _le("SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT")),
    Invariant("min_le_max_ip_pkt_len", "MIN_IP_PKT_LEN <= MAX_IP_PKT_LEN",
              _le("MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN")),
    Invariant("min_le_max_ttl", "MIN_TTL <= MAX_TTL", _le("MIN_TTL", "MAX_TTL")),
    Invariant("pkts_le_bytes_in", "IN_PKTS <= IN_BYTES (>=1 byte per packet)",
              _le("IN_PKTS", "IN_BYTES")),
    Invariant("pkts_le_bytes_out", "OUT_PKTS <= OUT_BYTES (>=1 byte per packet)",
              _le("OUT_PKTS", "OUT_BYTES")),
    Invariant("size_bins_within_packet_total",
              "sum(NUM_PKTS_* bins) <= IN_PKTS + OUT_PKTS", _bins_within_packets),
    Invariant("nonneg_in_bytes", "IN_BYTES >= 0", _nonneg("IN_BYTES")),
    Invariant("nonneg_out_bytes", "OUT_BYTES >= 0", _nonneg("OUT_BYTES")),
    Invariant("nonneg_flow_duration", "FLOW_DURATION_MILLISECONDS >= 0",
              _nonneg("FLOW_DURATION_MILLISECONDS")),
)


@dataclass
class FeatureTransform:
    """The notebook-01 preprocessing pipeline, with an inverse.

    Forward:  clip>=0 -> clip to [p1, p99] -> log1p -> robust scale -> clip +-10.
    Inverse recovers the value in the *clipped* raw domain, which is the domain
    the detector and the counterfactual search actually operate over.
    """

    pct_low: np.ndarray
    pct_high: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X_raw, dtype=np.float64), 0.0, None)
        X = np.log1p(np.clip(X, self.pct_low, self.pct_high))
        return np.clip((X - self.center) / self.scale, -10.0, 10.0)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        X = np.asarray(X_scaled, dtype=np.float64) * self.scale + self.center
        return np.clip(np.expm1(X), 0.0, None)

    @classmethod
    def fit(cls, X_benign_raw: np.ndarray) -> FeatureTransform:
        from sklearn.preprocessing import RobustScaler

        X = np.clip(np.asarray(X_benign_raw, dtype=np.float64), 0.0, None)
        lo, hi = np.percentile(X, [1, 99], axis=0)
        lo = np.nan_to_num(lo, nan=0.0, posinf=0.0, neginf=0.0)
        hi = np.maximum(np.nan_to_num(hi, nan=0.0, posinf=0.0, neginf=0.0), lo + 1e-6)
        Xl = np.log1p(np.clip(X, lo, hi))
        sc = RobustScaler().fit(Xl)
        return cls(pct_low=lo, pct_high=hi, center=sc.center_, scale=sc.scale_)


#: Invariants measured at a 0% violation rate on all four NF-v2 datasets by
#: `scripts/verify_nf_v2_invariants.py` (200k rows each).  Two candidates are
#: excluded because real traffic breaks them:
#:
#:   size_bins_within_packet_total -- 6.0% of NF-CIC2018 and 20.4% of
#:       NF-UNSW-NB15 rows, so the NUM_PKTS_* bins are not a strict subset of
#:       the packet total in these exports.
#:   pkts_le_bytes_in -- 73.2% of Edge-IIoTset rows report fewer IN_BYTES than
#:       IN_PKTS, which is impossible at >=1 byte per packet.  This is a
#:       field-integrity problem in that re-export, not a property of NetFlow;
#:       treat Edge-IIoTset byte/packet ratios with suspicion.
VERIFIED_INVARIANTS: tuple[Invariant, ...] = tuple(
    inv
    for inv in CANDIDATE_INVARIANTS
    if inv.name not in {"size_bins_within_packet_total", "pkts_le_bytes_in"}
)


def violated_invariants(
    x_raw: np.ndarray,
    feature_names: list[str],
    invariants: tuple[Invariant, ...] = VERIFIED_INVARIANTS,
) -> list[str]:
    """Names of the invariants a raw feature vector breaks."""
    ix = _idx(feature_names)
    return [inv.name for inv in invariants if not inv.check(np.asarray(x_raw, dtype=np.float64), ix)]


def plausibility_rate(
    x_raw: np.ndarray,
    feature_names: list[str],
    invariants: tuple[Invariant, ...] = VERIFIED_INVARIANTS,
) -> float:
    """Fraction of invariants satisfied. 1.0 = physically possible flow."""
    if not invariants:
        return 1.0
    return 1.0 - len(violated_invariants(x_raw, feature_names, invariants)) / len(invariants)


def is_physically_possible(
    x_raw: np.ndarray,
    feature_names: list[str],
    invariants: tuple[Invariant, ...] = VERIFIED_INVARIANTS,
) -> bool:
    """True when the flow breaks no invariant at all."""
    return not violated_invariants(x_raw, feature_names, invariants)
