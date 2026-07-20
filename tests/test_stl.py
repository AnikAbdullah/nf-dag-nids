"""Tests for Module 4: stl.

Coverage targets:
  - STLNode robustness semantics (all node types)
  - Signal extraction from Polars DataFrames
  - All 4 MITRE technique formulae (true positive + true negative cases)
  - Classifier: predict() and window_ground_truth()
  - Metrics: precision/recall arithmetic, W22 gate logic
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from caushap_nids.stl.evaluator import (
    ge, le, eq, neg, and_, or_, globally, eventually,
    Ge, Le, Eq, Globally, Eventually, And, Or,
)
from caushap_nids.stl.signals import extract_signals
from caushap_nids.stl.formulae import (
    FTP_BRUTE_FORCE,
    SSH_BRUTE_FORCE,
    DDOS_VOLUMETRIC,
    ENDPOINT_DOS,
    PORT_SCAN,
    DATA_EXFIL,
    C2_APP_LAYER,
    ALL_TECHNIQUES,
    MitreTechnique,
    STLResult,
)
from caushap_nids.stl.classifier import (
    evaluate_all,
    predict,
    window_ground_truth,
)
from caushap_nids.stl.metrics import (
    evaluate_test_split,
    format_gate_result,
    W22GateResult,
    TechniqueMetrics,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _signals(n: int = 5, **overrides: float | list) -> dict[str, np.ndarray]:
    """Build a minimal signal dict, all zeros by default, with overrides."""
    cols = [
        "L4_DST_PORT", "L4_SRC_PORT", "PROTOCOL", "L7_PROTO",
        "FLOW_DURATION_MILLISECONDS", "IN_BYTES", "OUT_BYTES",
        "IN_PKTS", "OUT_PKTS", "TCP_FLAGS",
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "NUM_PKTS_UP_TO_128_BYTES", "FTP_COMMAND_RET_CODE",
        "MIN_TTL", "MAX_TTL",
    ]
    s: dict[str, np.ndarray] = {c: np.zeros(n, dtype=np.float64) for c in cols}
    for k, v in overrides.items():
        if isinstance(v, list):
            s[k] = np.array(v, dtype=np.float64)
        else:
            s[k] = np.full(n, v, dtype=np.float64)
    return s


def _window_df(n: int = 5, **col_values) -> pl.DataFrame:
    """Build a raw-feature Polars DataFrame with constant column values."""
    from caushap_nids.stl.signals import _REQUIRED_COLS
    data: dict[str, list] = {c: [0.0] * n for c in _REQUIRED_COLS}
    data["attack_family"] = ["Benign"] * n
    for k, v in col_values.items():
        if isinstance(v, list):
            data[k] = v
        else:
            data[k] = [v] * n
    return pl.DataFrame(data)


# ── evaluator: atomic propositions ────────────────────────────────────────────

class TestAtomics:
    def test_ge_satisfied(self):
        s = _signals(3, IN_PKTS=10.0)
        assert ge("IN_PKTS", 5.0).satisfied(s)

    def test_ge_violated(self):
        s = _signals(3, IN_PKTS=3.0)
        assert not ge("IN_PKTS", 5.0).satisfied(s)

    def test_le_satisfied(self):
        s = _signals(3, IN_PKTS=3.0)
        assert le("IN_PKTS", 5.0).satisfied(s)

    def test_le_violated(self):
        s = _signals(3, IN_PKTS=9.0)
        assert not le("IN_PKTS", 5.0).satisfied(s)

    def test_eq_satisfied(self):
        s = _signals(4, PROTOCOL=6.0)
        assert eq("PROTOCOL", 6.0).satisfied(s)

    def test_eq_violated(self):
        s = _signals(4, PROTOCOL=17.0)
        assert not eq("PROTOCOL", 6.0).satisfied(s)

    def test_eq_mixed_violated(self):
        s = _signals(n=3, PROTOCOL=0.0)
        s["PROTOCOL"] = np.array([6.0, 17.0, 6.0])
        assert not eq("PROTOCOL", 6.0).satisfied(s)

    def test_robustness_value(self):
        s = _signals(3, IN_PKTS=8.0)
        rho = ge("IN_PKTS", 5.0).global_robustness(s)
        assert pytest.approx(rho) == 3.0


# ── evaluator: connectives ────────────────────────────────────────────────────

class TestConnectives:
    def test_and_all_satisfied(self):
        s = _signals(3, PROTOCOL=6.0, L4_DST_PORT=21.0)
        assert and_(eq("PROTOCOL", 6.0), eq("L4_DST_PORT", 21.0)).satisfied(s)

    def test_and_one_violated(self):
        s = _signals(3, PROTOCOL=17.0, L4_DST_PORT=21.0)
        assert not and_(eq("PROTOCOL", 6.0), eq("L4_DST_PORT", 21.0)).satisfied(s)

    def test_or_one_satisfied(self):
        s = _signals(3, PROTOCOL=17.0)
        assert or_(eq("PROTOCOL", 6.0), eq("PROTOCOL", 17.0)).satisfied(s)

    def test_or_all_violated(self):
        s = _signals(3, PROTOCOL=1.0)
        assert not or_(eq("PROTOCOL", 6.0), eq("PROTOCOL", 17.0)).satisfied(s)

    def test_neg(self):
        s = _signals(3, PROTOCOL=17.0)
        assert not eq("PROTOCOL", 6.0).satisfied(s)

    def test_and_requires_two_children(self):
        with pytest.raises(ValueError):
            and_(eq("PROTOCOL", 6.0))

    def test_or_requires_two_children(self):
        with pytest.raises(ValueError):
            or_(eq("PROTOCOL", 6.0))

    def test_and_three_children(self):
        s = _signals(3, PROTOCOL=6.0, L4_DST_PORT=80.0, MIN_TTL=63.0)
        assert and_(eq("PROTOCOL", 6.0), eq("L4_DST_PORT", 80.0),
                    eq("MIN_TTL", 63.0)).satisfied(s)


# ── evaluator: temporal operators ────────────────────────────────────────────

class TestTemporalOperators:
    def test_globally_all_satisfy(self):
        s = _signals(5, PROTOCOL=6.0)
        assert globally(eq("PROTOCOL", 6.0)).satisfied(s)

    def test_globally_one_violates(self):
        s = _signals(n=5, PROTOCOL=0.0)
        s["PROTOCOL"] = np.array([6.0, 6.0, 17.0, 6.0, 6.0])
        assert not globally(eq("PROTOCOL", 6.0)).satisfied(s)

    def test_eventually_one_satisfies(self):
        s = _signals(n=5, IN_PKTS=0.0)
        s["IN_PKTS"] = np.array([5.0, 5.0, 963.0, 5.0, 5.0])
        assert eventually(ge("IN_PKTS", 963.0)).satisfied(s)

    def test_eventually_none_satisfies(self):
        s = _signals(5, IN_PKTS=5.0)
        assert not eventually(ge("IN_PKTS", 963.0)).satisfied(s)

    def test_globally_empty_window(self):
        s = _signals(0)
        rho = globally(eq("PROTOCOL", 6.0)).global_robustness(s)
        assert rho == -np.inf

    def test_globally_robustness_is_minimum(self):
        s = _signals(n=4, IN_PKTS=0.0)
        s["IN_PKTS"] = np.array([10.0, 5.0, 20.0, 8.0])
        rho = globally(ge("IN_PKTS", 3.0)).global_robustness(s)
        assert pytest.approx(rho) == 2.0

    def test_eventually_robustness_is_maximum(self):
        s = _signals(n=4, IN_PKTS=0.0)
        s["IN_PKTS"] = np.array([10.0, 5.0, 20.0, 8.0])
        rho = eventually(ge("IN_PKTS", 3.0)).global_robustness(s)
        assert pytest.approx(rho) == 17.0


# ── signal extraction ─────────────────────────────────────────────────────────

class TestSignalExtraction:
    def test_all_required_cols_present(self):
        from caushap_nids.stl.signals import _REQUIRED_COLS
        df = _window_df(10)
        s = extract_signals(df)
        for col in _REQUIRED_COLS:
            assert col in s, f"Missing signal: {col}"
            assert s[col].shape == (10,)
            assert s[col].dtype == np.float64

    def test_missing_col_fills_zeros(self):
        df = pl.DataFrame({"L4_DST_PORT": [21.0] * 5})
        s = extract_signals(df)
        assert "PROTOCOL" in s
        assert np.all(s["PROTOCOL"] == 0.0)

    def test_values_match_dataframe(self):
        df = _window_df(4, L4_DST_PORT=22.0, IN_PKTS=10.0)
        s = extract_signals(df)
        assert np.all(s["L4_DST_PORT"] == 22.0)
        assert np.all(s["IN_PKTS"] == 10.0)

    def test_empty_window(self):
        df = _window_df(0)
        s = extract_signals(df)
        for arr in s.values():
            assert arr.shape == (0,)


# ── technique formulae: T1 FTP Brute Force ───────────────────────────────────
#
# Formula: G(port=21 ∧ TCP ∧ MIN_TTL=63 ∧ IN_PKTS∈[13,14])
# Discriminators grounded in NF-v2: FTP attack IN_PKTS always 13–14;
# benign FTP IN_PKTS max=11; SlowHTTPTest (same port) has IN_PKTS=7–8.

class TestFTPBruteForce:
    def _ftp_window(self, n: int = 10, in_pkts: float = 13.0) -> pl.DataFrame:
        return _window_df(
            n,
            L4_DST_PORT=21.0,
            PROTOCOL=6.0,
            MIN_TTL=63.0,
            MAX_TTL=63.0,
            IN_PKTS=in_pkts,
            attack_family="FTP-BruteForce",
        )

    def test_true_positive_pkts13(self):
        assert FTP_BRUTE_FORCE.evaluate(self._ftp_window(in_pkts=13.0)).satisfied

    def test_true_positive_pkts14(self):
        assert FTP_BRUTE_FORCE.evaluate(self._ftp_window(in_pkts=14.0)).satisfied

    def test_low_pkts_rejected(self):
        # Benign FTP has IN_PKTS ≤ 11
        r = FTP_BRUTE_FORCE.evaluate(self._ftp_window(in_pkts=3.0))
        assert not r.satisfied

    def test_high_pkts_rejected(self):
        # IN_PKTS = 20 exceeds the ≤14 upper bound
        r = FTP_BRUTE_FORCE.evaluate(self._ftp_window(in_pkts=20.0))
        assert not r.satisfied

    def test_wrong_port_rejected(self):
        df = _window_df(10, L4_DST_PORT=22.0, PROTOCOL=6.0,
                        MIN_TTL=63.0, IN_PKTS=13.0)
        assert not FTP_BRUTE_FORCE.evaluate(df).satisfied

    def test_udp_rejected(self):
        df = _window_df(10, L4_DST_PORT=21.0, PROTOCOL=17.0,
                        MIN_TTL=63.0, IN_PKTS=13.0)
        assert not FTP_BRUTE_FORCE.evaluate(df).satisfied

    def test_wrong_ttl_rejected(self):
        # Different MIN_TTL (not the attacker's 63) → not satisfied
        df = _window_df(10, L4_DST_PORT=21.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, IN_PKTS=13.0)
        assert not FTP_BRUTE_FORCE.evaluate(df).satisfied

    def test_slow_http_test_not_confused(self):
        # SlowHTTPTest: same port/proto/TTL but IN_PKTS=7-8 → not FTP attack
        df = _window_df(10, L4_DST_PORT=21.0, PROTOCOL=6.0,
                        MIN_TTL=63.0, IN_PKTS=7.0)
        assert not FTP_BRUTE_FORCE.evaluate(df).satisfied

    def test_below_min_window_size(self):
        df = _window_df(2, L4_DST_PORT=21.0, PROTOCOL=6.0,
                        MIN_TTL=63.0, IN_PKTS=13.0)
        assert not FTP_BRUTE_FORCE.evaluate(df).satisfied

    def test_result_type(self):
        r = FTP_BRUTE_FORCE.evaluate(self._ftp_window())
        assert isinstance(r, STLResult)
        assert r.mitre_id == "T1110.001"


# ── technique formulae: T2 SSH Brute Force ───────────────────────────────────
#
# Formula: G(port=22 ∧ TCP ∧ IN_PKTS≥20)
# Brute-force SSH DH+auth exchanges require ≥20 pkts; benign SSH p95=13 pkts.
# Only 10 benign port-22 flows (of 2.27 M) have IN_PKTS≥20.

class TestSSHBruteForce:
    def _ssh_window(self, n: int = 10, in_pkts: float = 23.0) -> pl.DataFrame:
        return _window_df(
            n,
            L4_DST_PORT=22.0,
            PROTOCOL=6.0,
            IN_PKTS=in_pkts,
            attack_family="SSH-Bruteforce",
        )

    def test_true_positive(self):
        assert SSH_BRUTE_FORCE.evaluate(self._ssh_window()).satisfied

    def test_low_pkts_rejected(self):
        # Benign SSH: p95 = 13 pkts
        r = SSH_BRUTE_FORCE.evaluate(self._ssh_window(in_pkts=5.0))
        assert not r.satisfied

    def test_boundary_pkts_rejected(self):
        # 19 pkts — just below the ≥20 threshold
        r = SSH_BRUTE_FORCE.evaluate(self._ssh_window(in_pkts=19.0))
        assert not r.satisfied

    def test_wrong_port_rejected(self):
        df = _window_df(10, L4_DST_PORT=21.0, PROTOCOL=6.0, IN_PKTS=23.0)
        assert not SSH_BRUTE_FORCE.evaluate(df).satisfied

    def test_udp_rejected(self):
        df = _window_df(10, L4_DST_PORT=22.0, PROTOCOL=17.0, IN_PKTS=23.0)
        assert not SSH_BRUTE_FORCE.evaluate(df).satisfied

    def test_mitre_id(self):
        assert SSH_BRUTE_FORCE.mitre_id == "T1110.004"


# ── technique formulae: T3 DDoS Volumetric ───────────────────────────────────
#
# Formula: F(LOIC-UDP ∨ HOIC ∨ Hulk ∨ LOIC-HTTP)
# Each sub-condition uses TTL + TCP_FLAGS or protocol invariants measured from
# real attack flows; all have 0 or near-0 benign overlap.

class TestDDoSVolumetric:
    def _loic_udp_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, PROTOCOL=17.0, IN_PKTS=5000.0,
                          attack_family="DDOS attack-LOIC-UDP")

    def _hoic_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, TCP_FLAGS=219.0, MIN_TTL=127.0,
                          attack_family="DDOS attack-HOIC")

    def _hulk_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, TCP_FLAGS=27.0, MIN_TTL=63.0, L4_DST_PORT=80.0,
                          PROTOCOL=6.0, attack_family="DoS attacks-Hulk")

    def _loic_http_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, TCP_FLAGS=223.0, MIN_TTL=127.0,
                          attack_family="DDoS attacks-LOIC-HTTP")

    def test_loic_udp_true_positive(self):
        assert DDOS_VOLUMETRIC.evaluate(self._loic_udp_window()).satisfied

    def test_hoic_true_positive(self):
        assert DDOS_VOLUMETRIC.evaluate(self._hoic_window()).satisfied

    def test_hulk_true_positive(self):
        assert DDOS_VOLUMETRIC.evaluate(self._hulk_window()).satisfied

    def test_loic_http_true_positive(self):
        assert DDOS_VOLUMETRIC.evaluate(self._loic_http_window()).satisfied

    def test_hoic_window_tolerates_off_pattern_flows(self):
        df = _window_df(
            10,
            L4_DST_PORT=80.0,
            PROTOCOL=6.0,
            TCP_FLAGS=[219.0] * 9 + [0.0],
            MIN_TTL=[127.0] * 9 + [0.0],
            attack_family="DDOS attack-HOIC",
        )
        assert DDOS_VOLUMETRIC.evaluate(df).satisfied

    def test_low_udp_pkts_rejected(self):
        # UDP with < 963 pkts — not a flood
        df = _window_df(10, PROTOCOL=17.0, IN_PKTS=50.0)
        assert not DDOS_VOLUMETRIC.evaluate(df).satisfied

    def test_benign_http_rejected(self):
        # Benign HTTP: port=80, TCP, but different MIN_TTL and TCP_FLAGS
        df = _window_df(10, L4_DST_PORT=80.0, PROTOCOL=6.0,
                        TCP_FLAGS=223.0, MIN_TTL=64.0)
        assert not DDOS_VOLUMETRIC.evaluate(df).satisfied

    def test_wrong_flags_rejected(self):
        # port=80, TCP, MIN_TTL=63, but TCP_FLAGS≠27 — not Hulk
        df = _window_df(10, L4_DST_PORT=80.0, PROTOCOL=6.0,
                        TCP_FLAGS=31.0, MIN_TTL=63.0)
        assert not DDOS_VOLUMETRIC.evaluate(df).satisfied

    def test_hulk_needs_port80(self):
        # TCP_FLAGS=27 + MIN_TTL=63 but wrong port — guards against SSH overlap
        df = _window_df(10, L4_DST_PORT=22.0, PROTOCOL=6.0,
                        TCP_FLAGS=27.0, MIN_TTL=63.0)
        assert not DDOS_VOLUMETRIC.evaluate(df).satisfied

    def test_mitre_id(self):
        assert DDOS_VOLUMETRIC.mitre_id == "T1498"


# ── technique formulae: T4 Endpoint DoS (slow) ───────────────────────────────
#
# Formula: G(Slowloris_or_GE ∨ SlowHTTPTest)
# Slowloris/GoldenEye: port=80, TCP, MIN_TTL=63 — 0 benign overlap.
# SlowHTTPTest: port=21, TCP, MIN_TTL=63, IN_PKTS≤10 — separates from FTP (IN_PKTS=13-14).

class TestEndpointDoS:
    def _slowloris_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, L4_DST_PORT=80.0, PROTOCOL=6.0,
                          MIN_TTL=63.0, IN_PKTS=15.0,
                          attack_family="DoS attacks-Slowloris")

    def _slow_http_test_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, L4_DST_PORT=21.0, PROTOCOL=6.0,
                          MIN_TTL=63.0, IN_PKTS=7.0,
                          attack_family="DoS attacks-SlowHTTPTest")

    def _goldeneye_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, L4_DST_PORT=80.0, PROTOCOL=6.0,
                          MIN_TTL=63.0, TCP_FLAGS=31.0,
                          attack_family="DoS attacks-GoldenEye")

    def test_slowloris_true_positive(self):
        assert ENDPOINT_DOS.evaluate(self._slowloris_window()).satisfied

    def test_slow_http_test_true_positive(self):
        assert ENDPOINT_DOS.evaluate(self._slow_http_test_window()).satisfied

    def test_goldeneye_true_positive(self):
        assert ENDPOINT_DOS.evaluate(self._goldeneye_window()).satisfied

    def test_wrong_ttl_rejected(self):
        # MIN_TTL=128 (not 63) → no sub-condition fires
        df = _window_df(10, L4_DST_PORT=80.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, IN_PKTS=15.0)
        assert not ENDPOINT_DOS.evaluate(df).satisfied

    def test_wrong_port_rejected(self):
        df = _window_df(10, L4_DST_PORT=443.0, PROTOCOL=6.0,
                        MIN_TTL=63.0, IN_PKTS=15.0)
        assert not ENDPOINT_DOS.evaluate(df).satisfied

    def test_udp_rejected(self):
        df = _window_df(10, L4_DST_PORT=80.0, PROTOCOL=17.0,
                        MIN_TTL=63.0, IN_PKTS=15.0)
        assert not ENDPOINT_DOS.evaluate(df).satisfied

    def test_ftp_attack_not_confused_with_slow_http_test(self):
        # FTP-BruteForce: same port/proto/TTL but IN_PKTS=13 > 10
        df = _window_df(10, L4_DST_PORT=21.0, PROTOCOL=6.0,
                        MIN_TTL=63.0, IN_PKTS=13.0)
        assert not ENDPOINT_DOS.evaluate(df).satisfied

    def test_mitre_id(self):
        assert ENDPOINT_DOS.mitre_id == "T1499"


# ── classifier ────────────────────────────────────────────────────────────────

class TestClassifier:
    def _ftp_window(self, n: int = 10) -> pl.DataFrame:
        return _window_df(n, L4_DST_PORT=21.0, PROTOCOL=6.0,
                          MIN_TTL=63.0, IN_PKTS=13.0,
                          attack_family="FTP-BruteForce")

    def test_evaluate_all_returns_seven_results(self):
        results = evaluate_all(self._ftp_window())
        assert len(results) == 7
        assert all(isinstance(r, STLResult) for r in results)

    def test_predict_ftp_window(self):
        pred = predict(self._ftp_window())
        assert pred == "T1110.001"

    def test_predict_benign_returns_none(self):
        # port=12345: no technique condition fires
        df = _window_df(10, L4_DST_PORT=12345.0, PROTOCOL=6.0, IN_PKTS=10.0)
        pred = predict(df)
        assert pred is None

    def test_predict_ssh_window(self):
        df = _window_df(10, L4_DST_PORT=22.0, PROTOCOL=6.0, IN_PKTS=23.0)
        assert predict(df) == "T1110.004"

    def test_predict_hoic_window(self):
        df = _window_df(10, TCP_FLAGS=219.0, MIN_TTL=127.0)
        assert predict(df) == "T1498"

    def test_predict_slowloris_window(self):
        df = _window_df(10, L4_DST_PORT=80.0, PROTOCOL=6.0, MIN_TTL=63.0)
        assert predict(df) == "T1499"


class TestWindowGroundTruth:
    def test_majority_family(self):
        df = pl.DataFrame({
            "attack_family": ["FTP-BruteForce"] * 8 + ["Benign"] * 2
        })
        assert window_ground_truth(df) == "FTP-BruteForce"

    def test_benign_majority(self):
        df = pl.DataFrame({
            "attack_family": ["Benign"] * 9 + ["FTP-BruteForce"] * 1
        })
        assert window_ground_truth(df) == "Benign"

    def test_no_clear_majority_returns_none(self):
        df = pl.DataFrame({
            "attack_family": ["A"] * 4 + ["B"] * 4 + ["C"] * 2
        })
        result = window_ground_truth(df, majority_threshold=0.5)
        assert result is None

    def test_empty_window_returns_none(self):
        df = pl.DataFrame({"attack_family": []})
        assert window_ground_truth(df) is None

    def test_missing_col_returns_none(self):
        df = pl.DataFrame({"L4_DST_PORT": [21.0]})
        assert window_ground_truth(df) is None


# ── metrics / W22 gate ────────────────────────────────────────────────────────

def _make_test_df(
    n_ftp: int = 200,
    n_ssh: int = 200,
    n_benign: int = 200,
    n_bot: int = 0,
    n_infil: int = 0,
) -> pl.DataFrame:
    """Synthetic DataFrame with attack blocks matching the calibrated formulae.

    evaluate_test_split(sort_by_family=True) will group these into pure-family
    windows regardless of insertion order.
    """
    from caushap_nids.stl.signals import _REQUIRED_COLS

    rows: dict[str, list] = {c: [] for c in _REQUIRED_COLS}
    rows["attack_family"] = []

    def _add(n: int, family: str, **vals):
        for col in _REQUIRED_COLS:
            rows[col].extend([vals.get(col, 0.0)] * n)
        rows["attack_family"].extend([family] * n)

    # FTP brute-force: port=21, TCP, MIN_TTL=63, IN_PKTS=13
    _add(n_ftp, "FTP-BruteForce",
         L4_DST_PORT=21.0, PROTOCOL=6.0, MIN_TTL=63.0,
         MAX_TTL=63.0, IN_PKTS=13.0)

    # SSH brute-force: port=22, TCP, IN_PKTS=23
    _add(n_ssh, "SSH-Bruteforce",
         L4_DST_PORT=22.0, PROTOCOL=6.0, IN_PKTS=23.0)

    # Bot C2: port=8080, TCP, MIN/MAX_TTL=128, TCP_FLAGS=219, IN_PKTS=5
    _add(n_bot, "Bot",
         L4_DST_PORT=8080.0, PROTOCOL=6.0, MIN_TTL=128.0,
         MAX_TTL=128.0, TCP_FLAGS=219.0, IN_PKTS=5.0)

    # Infilteration RDP-recon sub-fingerprint: port=3389, TCP, MIN_TTL=105
    _add(n_infil, "Infilteration",
         L4_DST_PORT=3389.0, PROTOCOL=6.0, MIN_TTL=105.0, MAX_TTL=105.0)

    # Benign: port=443, TCP — no formula fires
    _add(n_benign, "Benign",
         L4_DST_PORT=443.0, PROTOCOL=6.0, IN_PKTS=5.0)

    return pl.DataFrame(rows)


class TestMetrics:
    def test_evaluate_returns_seven_technique_metrics(self):
        df = _make_test_df()
        result = evaluate_test_split(df, window_size=50)
        assert isinstance(result, W22GateResult)
        # 7 techniques after the v2 repair pass: FTP, SSH, DDoS, EndpointDoS,
        # plus T1046 (Port Scan), T1041 (Exfil), T1071 (C2).
        assert len(result.technique_metrics) == 7

    def test_required_passing_scales_with_count(self):
        df = _make_test_df()
        result = evaluate_test_split(df, window_size=50)
        # 7 techniques registered ⇒ majority threshold is 4.
        assert result.required_passing == 4

    def test_ftp_and_ssh_pass_gate(self):
        df = _make_test_df(n_ftp=250, n_ssh=250, n_benign=100)
        result = evaluate_test_split(df, window_size=50)
        ftp = next(m for m in result.technique_metrics if m.mitre_id == "T1110.001")
        ssh = next(m for m in result.technique_metrics if m.mitre_id == "T1110.004")
        assert ftp.precision == pytest.approx(1.0)
        assert ssh.precision == pytest.approx(1.0)

    def test_gate_fails_with_only_two_techniques_passing(self):
        # With 7 techniques the gate now requires 4 to pass — two attacks alone
        # are insufficient.
        df = _make_test_df(n_ftp=250, n_ssh=250, n_benign=100)
        result = evaluate_test_split(df, window_size=50)
        assert not result.passed
        assert result.techniques_passing < result.required_passing

    def test_gate_passes_with_four_techniques(self):
        # FTP + SSH + Bot (T1071) + Infilteration-RDP (T1046) → 4 techniques
        # at precision 1.0 (synthetic windows have no benign contamination
        # because evaluate_test_split sorts by family before windowing).
        df = _make_test_df(
            n_ftp=250, n_ssh=250, n_bot=250, n_infil=250, n_benign=100,
        )
        result = evaluate_test_split(df, window_size=50)
        assert result.passed
        passing = {m.mitre_id for m in result.technique_metrics if m.passes_gate}
        assert {"T1110.001", "T1110.004", "T1071"}.issubset(passing)
        # T1046 must clear precision ≥ 0.95 on the synthetic Infilteration block.
        t1046 = next(m for m in result.technique_metrics if m.mitre_id == "T1046")
        assert t1046.precision >= 0.95

    def test_gate_fails_when_no_attacks(self):
        # All benign zeros: no technique fires → precision=0 → gate fails
        from caushap_nids.stl.signals import _REQUIRED_COLS
        rows = {c: [0.0] * 300 for c in _REQUIRED_COLS}
        rows["attack_family"] = ["Benign"] * 300
        df = pl.DataFrame(rows)
        result = evaluate_test_split(df, window_size=50)
        assert not result.passed

    def test_included_windows_leq_total(self):
        df = _make_test_df()
        result = evaluate_test_split(df, window_size=50)
        assert result.included_windows <= result.total_windows

    def test_sort_by_family_false_supported(self):
        # sort_by_family=False must not error
        df = _make_test_df(n_ftp=100, n_ssh=100, n_benign=100)
        result = evaluate_test_split(df, window_size=50, sort_by_family=False)
        assert isinstance(result, W22GateResult)

    def test_precision_formula(self):
        m = TechniqueMetrics(
            mitre_id="T1110.001", name="FTP Brute Force",
            tp=10, fp=2, fn=3, tn=100,
            windows_predicted=12, windows_true=13,
            precision=10 / 12, recall=10 / 13,
            gate_precision_threshold=0.95,
            passes_gate=(10 / 12 >= 0.95),
        )
        assert pytest.approx(m.precision, rel=1e-6) == 10 / 12
        assert not m.passes_gate

    def test_format_gate_result_contains_status(self):
        df = _make_test_df(n_ftp=250, n_ssh=250, n_benign=100)
        result = evaluate_test_split(df, window_size=50)
        text = format_gate_result(result)
        assert "PASS" in text or "FAIL" in text

    def test_gate_result_to_dict(self):
        df = _make_test_df()
        result = evaluate_test_split(df, window_size=50)
        d = result.to_dict()
        assert "passed" in d
        assert "status" in d
        assert d["status"] in ("PASS", "FAIL")

    def test_save_gate_result(self, tmp_path):
        import json
        df = _make_test_df()
        result = evaluate_test_split(df, window_size=50)
        out = tmp_path / "gate.json"
        from caushap_nids.stl.metrics import save_gate_result
        save_gate_result(result, out)
        loaded = json.loads(out.read_text())
        assert "passed" in loaded


# ── ALL_TECHNIQUES registry ───────────────────────────────────────────────────

class TestRegistry:
    def test_seven_techniques_registered(self):
        # 4 frozen pre-v2-repair + 3 added in the repair pass (T1046/T1041/T1071)
        assert len(ALL_TECHNIQUES) == 7

    def test_all_have_unique_mitre_ids(self):
        ids = [t.mitre_id for t in ALL_TECHNIQUES]
        assert len(set(ids)) == len(ids)

    def test_all_target_families_non_empty(self):
        for t in ALL_TECHNIQUES:
            assert len(t.target_families) >= 1

    def test_t1046_and_t1041_share_infilteration_target(self):
        # By design: T1046 (recon) and T1041 (exfil) both partition the
        # Infilteration family. predict() picks the higher-robustness winner
        # per window. The metrics module's TP/FP arithmetic is unchanged by
        # this overlap because TP/FP are evaluated per technique independently.
        t1046 = next(t for t in ALL_TECHNIQUES if t.mitre_id == "T1046")
        t1041 = next(t for t in ALL_TECHNIQUES if t.mitre_id == "T1041")
        assert "Infilteration" in t1046.target_families
        assert "Infilteration" in t1041.target_families

    def test_disjoint_target_families_outside_infilteration(self):
        # Every family except Infilteration belongs to exactly one technique.
        seen: dict[str, str] = {}
        for t in ALL_TECHNIQUES:
            for fam in t.target_families:
                if fam == "Infilteration":
                    continue
                assert fam not in seen, (
                    f"{fam} is claimed by both {seen[fam]} and {t.mitre_id}"
                )
                seen[fam] = t.mitre_id

    def test_goldeneye_in_t1499_not_t1498(self):
        t1498 = next(t for t in ALL_TECHNIQUES if t.mitre_id == "T1498")
        t1499 = next(t for t in ALL_TECHNIQUES if t.mitre_id == "T1499")
        assert "DoS attacks-GoldenEye" not in t1498.target_families
        assert "DoS attacks-GoldenEye" in t1499.target_families


# ── YAML library round-trip (artifacts/stl_library.yaml) ────────────────────


class TestLibraryYaml:
    def test_round_trip_preserves_all_techniques(self, tmp_path):
        from caushap_nids.stl import dump_library, load_library
        out = tmp_path / "stl_library.yaml"
        dump_library(ALL_TECHNIQUES, out)
        reloaded = load_library(out)
        assert len(reloaded) == len(ALL_TECHNIQUES)
        for orig, rt in zip(ALL_TECHNIQUES, reloaded):
            assert orig.mitre_id == rt.mitre_id
            assert orig.name == rt.name
            assert orig.target_families == rt.target_families
            assert orig.min_window_size == rt.min_window_size
            assert orig.formula == rt.formula

    def test_round_trip_preserves_formula_ast(self, tmp_path):
        # Pick a technique with deeply nested AST (T1499 has G(or(and(...))))
        from caushap_nids.stl import dump_library, load_library
        out = tmp_path / "stl_library.yaml"
        dump_library(ALL_TECHNIQUES, out)
        reloaded = load_library(out)
        t1499_orig = next(t for t in ALL_TECHNIQUES if t.mitre_id == "T1499")
        t1499_rt   = next(t for t in reloaded         if t.mitre_id == "T1499")
        # Equality on frozen dataclasses recurses into formula children.
        assert t1499_orig.formula == t1499_rt.formula

    def test_dump_returns_yaml_text(self):
        from caushap_nids.stl import dump_library
        text = dump_library(ALL_TECHNIQUES)
        assert "techniques" in text
        assert "T1046" in text and "T1041" in text and "T1071" in text
        assert "version" in text

    def test_load_unknown_node_type_raises(self, tmp_path):
        from caushap_nids.stl.library import load_library
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: '1.0'\n"
            "techniques:\n"
            "  - mitre_id: T9999\n"
            "    name: bogus\n"
            "    target_families: []\n"
            "    min_window_size: 3\n"
            "    formula: {type: not_a_node}\n"
        )
        with pytest.raises(ValueError):
            load_library(bad)


# ── new techniques: T1046 / T1041 / T1071 (v2 repair pass) ──────────────────


class TestPortScanT1046:
    def test_rdp_recon_satisfied(self):
        df = _window_df(10, L4_DST_PORT=3389.0, PROTOCOL=6.0, MIN_TTL=105.0)
        assert PORT_SCAN.evaluate(df).satisfied

    def test_smb_445_recon_satisfied(self):
        df = _window_df(10, L4_DST_PORT=445.0, PROTOCOL=6.0,
                        IN_PKTS=1.0, OUT_BYTES=80.0)
        assert PORT_SCAN.evaluate(df).satisfied

    def test_dns_recon_satisfied(self):
        df = _window_df(10, L4_DST_PORT=53.0, PROTOCOL=17.0,
                        MIN_TTL=0.0, IN_PKTS=1.0)
        assert PORT_SCAN.evaluate(df).satisfied

    def test_benign_rdp_with_standard_ttl_rejected(self):
        # Benign RDP at port 3389 uses MIN_TTL=128, not 105
        df = _window_df(10, L4_DST_PORT=3389.0, PROTOCOL=6.0, MIN_TTL=128.0)
        assert not PORT_SCAN.evaluate(df).satisfied

    def test_benign_dns_rejected(self):
        # Benign DNS uses MIN_TTL=100/128, not 0
        df = _window_df(10, L4_DST_PORT=53.0, PROTOCOL=17.0,
                        MIN_TTL=128.0, IN_PKTS=1.0)
        assert not PORT_SCAN.evaluate(df).satisfied

    def test_long_smb_session_rejected(self):
        # Benign SMB sessions have IN_PKTS > 5
        df = _window_df(10, L4_DST_PORT=445.0, PROTOCOL=6.0,
                        IN_PKTS=20.0, OUT_BYTES=80.0)
        assert not PORT_SCAN.evaluate(df).satisfied

    def test_mitre_id(self):
        df = _window_df(10, L4_DST_PORT=3389.0, PROTOCOL=6.0, MIN_TTL=105.0)
        assert PORT_SCAN.evaluate(df).mitre_id == "T1046"


class TestDataExfilT1041:
    def test_https_exfil_satisfied(self):
        df = _window_df(10, L4_DST_PORT=443.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, OUT_BYTES=5000.0,
                        DST_TO_SRC_AVG_THROUGHPUT=20_000_000.0)
        assert DATA_EXFIL.evaluate(df).satisfied

    def test_http_exfil_satisfied(self):
        df = _window_df(10, L4_DST_PORT=80.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, OUT_BYTES=5000.0, OUT_PKTS=10.0)
        assert DATA_EXFIL.evaluate(df).satisfied

    def test_benign_browsing_rejected(self):
        # Typical benign HTTPS browsing has small client uploads
        df = _window_df(10, L4_DST_PORT=443.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, OUT_BYTES=500.0,
                        DST_TO_SRC_AVG_THROUGHPUT=20_000_000.0)
        assert not DATA_EXFIL.evaluate(df).satisfied

    def test_low_throughput_rejected(self):
        # OUT_BYTES high but slow → not a real exfil
        df = _window_df(10, L4_DST_PORT=443.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, OUT_BYTES=5000.0,
                        DST_TO_SRC_AVG_THROUGHPUT=1_000_000.0)
        assert not DATA_EXFIL.evaluate(df).satisfied

    def test_mitre_id(self):
        df = _window_df(10, L4_DST_PORT=443.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, OUT_BYTES=5000.0,
                        DST_TO_SRC_AVG_THROUGHPUT=20_000_000.0)
        assert DATA_EXFIL.evaluate(df).mitre_id == "T1041"


class TestC2AppLayerT1071:
    def _bot_window(self, n: int = 10, tcp_flags: float = 219.0,
                    in_pkts: float = 5.0) -> pl.DataFrame:
        return _window_df(
            n,
            L4_DST_PORT=8080.0,
            PROTOCOL=6.0,
            MIN_TTL=128.0,
            MAX_TTL=128.0,
            TCP_FLAGS=tcp_flags,
            IN_PKTS=in_pkts,
            attack_family="Bot",
        )

    def test_true_positive_flags_219(self):
        assert C2_APP_LAYER.evaluate(self._bot_window()).satisfied

    def test_true_positive_flags_223(self):
        assert C2_APP_LAYER.evaluate(self._bot_window(tcp_flags=223.0)).satisfied

    def test_wrong_port_rejected(self):
        df = _window_df(10, L4_DST_PORT=443.0, PROTOCOL=6.0,
                        MIN_TTL=128.0, MAX_TTL=128.0, TCP_FLAGS=219.0,
                        IN_PKTS=5.0)
        assert not C2_APP_LAYER.evaluate(df).satisfied

    def test_long_session_rejected(self):
        # Benign HTTP/8080 long-running session: IN_PKTS > 10
        df = self._bot_window(in_pkts=50.0)
        assert not C2_APP_LAYER.evaluate(df).satisfied

    def test_wrong_ttl_rejected(self):
        df = _window_df(10, L4_DST_PORT=8080.0, PROTOCOL=6.0,
                        MIN_TTL=64.0, MAX_TTL=64.0, TCP_FLAGS=219.0,
                        IN_PKTS=5.0)
        assert not C2_APP_LAYER.evaluate(df).satisfied

    def test_mitre_id(self):
        assert C2_APP_LAYER.evaluate(self._bot_window()).mitre_id == "T1071"
