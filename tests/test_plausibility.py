"""Tests for the graph-independent NF-v2 plausibility oracle."""

from __future__ import annotations

import numpy as np
import pytest

from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS
from caushap_nids.xai_layers.multi_obj_cf.plausibility import (
    CANDIDATE_INVARIANTS,
    VERIFIED_INVARIANTS,
    FeatureTransform,
    is_physically_possible,
    plausibility_rate,
    violated_invariants,
)

NAMES = list(NF_V2_FEATURE_COLS)
IX = {f: i for i, f in enumerate(NAMES)}


def _valid_flow() -> np.ndarray:
    """A small, internally consistent NF-v2 flow."""
    x = np.zeros(len(NAMES), dtype=np.float64)
    x[IX["PROTOCOL"]] = 6
    x[IX["IN_BYTES"]] = 4000
    x[IX["IN_PKTS"]] = 10
    x[IX["OUT_BYTES"]] = 2000
    x[IX["OUT_PKTS"]] = 8
    x[IX["FLOW_DURATION_MILLISECONDS"]] = 5000
    x[IX["DURATION_IN"]] = 3000
    x[IX["DURATION_OUT"]] = 2000
    x[IX["MIN_TTL"]] = 54
    x[IX["MAX_TTL"]] = 64
    x[IX["SHORTEST_FLOW_PKT"]] = 40
    x[IX["LONGEST_FLOW_PKT"]] = 1500
    x[IX["MIN_IP_PKT_LEN"]] = 40
    x[IX["MAX_IP_PKT_LEN"]] = 1500
    x[IX["RETRANSMITTED_IN_BYTES"]] = 200
    x[IX["RETRANSMITTED_IN_PKTS"]] = 2
    return x


def test_valid_flow_passes_every_invariant():
    assert violated_invariants(_valid_flow(), NAMES) == []
    assert is_physically_possible(_valid_flow(), NAMES)
    assert plausibility_rate(_valid_flow(), NAMES) == pytest.approx(1.0)


def test_retransmitting_more_than_received_is_impossible():
    x = _valid_flow()
    x[IX["RETRANSMITTED_IN_BYTES"]] = x[IX["IN_BYTES"]] + 1
    assert "retrans_in_bytes_le_in_bytes" in violated_invariants(x, NAMES)
    assert not is_physically_possible(x, NAMES)


def test_direction_longer_than_flow_is_impossible():
    x = _valid_flow()
    x[IX["DURATION_IN"]] = x[IX["FLOW_DURATION_MILLISECONDS"]] + 1
    assert "duration_in_le_flow" in violated_invariants(x, NAMES)


def test_min_above_max_is_impossible():
    x = _valid_flow()
    x[IX["MIN_TTL"]] = 200
    assert "min_le_max_ttl" in violated_invariants(x, NAMES)


def test_plausibility_rate_degrades_with_more_violations():
    x = _valid_flow()
    first = plausibility_rate(x, NAMES)
    x[IX["DURATION_IN"]] = 1e9
    second = plausibility_rate(x, NAMES)
    x[IX["RETRANSMITTED_IN_BYTES"]] = 1e9
    third = plausibility_rate(x, NAMES)
    assert first > second > third


def test_verified_set_excludes_the_two_that_real_traffic_breaks():
    names = {inv.name for inv in VERIFIED_INVARIANTS}
    assert "size_bins_within_packet_total" not in names
    assert "pkts_le_bytes_in" not in names
    assert len(VERIFIED_INVARIANTS) == len(CANDIDATE_INVARIANTS) - 2


def test_unknown_features_are_skipped_not_failed():
    """A feature vocabulary missing the referenced columns must not report violations."""
    assert violated_invariants(np.zeros(3), ["a", "b", "c"]) == []


def test_transform_roundtrip_recovers_clipped_raw_values():
    rng = np.random.default_rng(0)
    X_raw = rng.gamma(shape=2.0, scale=50.0, size=(2000, len(NAMES)))
    ft = FeatureTransform.fit(X_raw)

    X_scaled = ft.transform(X_raw)
    X_back = ft.inverse_transform(X_scaled)

    # Values inside the [p1, p99] band round-trip; clipped tails are not expected to.
    inside = (X_raw >= ft.pct_low) & (X_raw <= ft.pct_high)
    assert np.allclose(X_back[inside], X_raw[inside], rtol=1e-4, atol=1e-4)


def test_transform_inverse_is_non_negative():
    rng = np.random.default_rng(1)
    X_raw = rng.gamma(shape=2.0, scale=10.0, size=(500, len(NAMES)))
    ft = FeatureTransform.fit(X_raw)
    back = ft.inverse_transform(rng.normal(scale=3.0, size=(50, len(NAMES))))
    assert (back >= 0).all()
