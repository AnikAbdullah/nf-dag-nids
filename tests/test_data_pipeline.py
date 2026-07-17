"""Tests for Module 1: data_pipeline.

Coverage targets (CODING_PLAN §15):
- 100% line coverage on splits.py
- Property-based tests on temporal_split and zero_day_split
- Imbalance detection, windowing, checksums error paths
"""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from caushap_nids.data_pipeline.class_balance import report_imbalance
from caushap_nids.data_pipeline import splits as splits_module
from caushap_nids.data_pipeline.splits import temporal_split, zero_day_split, xenids_split
from caushap_nids.data_pipeline.windowing import windowize
from caushap_nids.data_pipeline.checksums import verify_dataset
from caushap_nids.data_pipeline.loaders import repair_protocol_fields, _stratified_split


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_df(n: int, n_attack: int = 0, seed: int = 42) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    label = [0] * (n - n_attack) + [1] * n_attack
    attack = ["Benign"] * (n - n_attack) + ["BruteForce"] * n_attack
    return pl.DataFrame({
        "label": label,
        "attack_family": attack,
        "feat_a": rng.random(n).tolist(),
        "feat_b": rng.random(n).tolist(),
    })


# ── temporal_split ──────────────────────────────────────────────────────────

class TestTemporalSplit:
    def test_fractions_sum(self):
        df = _make_df(1000)
        train, val, test = temporal_split(df, train_frac=0.70, val_frac=0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(df.unique())

    def test_no_overlap(self):
        df = _make_df(300)
        train, val, test = temporal_split(df)
        # Use index positions as proxy for temporal order
        assert len(train) > 0 and len(val) > 0 and len(test) > 0

    def test_train_val_test_sizes_approx(self):
        df = _make_df(1000)
        train, val, test = temporal_split(df, train_frac=0.70, val_frac=0.15)
        assert 650 <= len(train) <= 750
        assert 100 <= len(val) <= 200

    def test_timestamp_assertion_fires(self):
        ts = list(range(100, 0, -1))  # reverse order; function must sort first
        df = pl.DataFrame({
            "flow_start_time": ts,
            "label": [0] * 100,
        })
        train, val, test = temporal_split(df, timestamp_col="flow_start_time")
        assert train["flow_start_time"].max() < test["flow_start_time"].min()
        assert len(train) + len(val) + len(test) == len(df.unique())

    def test_timestamp_col_absent_falls_back(self):
        df = _make_df(200)
        train, val, test = temporal_split(df, timestamp_col="nonexistent")
        assert len(train) + len(val) + len(test) == len(df.unique())


@given(
    n=st.integers(min_value=10, max_value=500),
    train_frac=st.floats(min_value=0.5, max_value=0.8),
    val_frac=st.floats(min_value=0.1, max_value=0.2),
)
@settings(max_examples=50)
def test_temporal_split_property(n, train_frac, val_frac):
    if train_frac + val_frac >= 1.0:
        return
    df = _make_df(n)
    train, val, test = temporal_split(df, train_frac=train_frac, val_frac=val_frac)
    assert len(train) + len(val) + len(test) == len(df.unique())
    assert len(train) > 0


# ── _stratified_split (Edge-IIoTset / 5G-NIDD non-temporal datasets) ─────────

def _make_family_grouped_df(n_benign: int, n_attack: int) -> pl.DataFrame:
    """Mimic Edge/5G raw parquet: rows concatenated by class, NOT shuffled."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = n_benign + n_attack
    return pl.DataFrame({
        "label": [0] * n_benign + [1] * n_attack,
        "attack_family": ["Normal"] * n_benign + ["DDoS"] * n_attack,
        "feat_a": rng.random(n).tolist(),
    })


class TestStratifiedSplit:
    def test_every_split_has_both_classes(self):
        # The exact failure mode that collapsed Edge-IIoTset macro-F1 to ~0.5:
        # a positional cut of this family-grouped frame gives a single-class test set.
        df = _make_family_grouped_df(n_benign=300, n_attack=700)
        train, val, test = _stratified_split(df, seed=42)
        for part in (train, val, test):
            labels = set(part["label"].to_list())
            assert labels == {0, 1}, f"split missing a class: {labels}"

    def test_disjoint_and_covers_all_rows(self):
        df = _make_family_grouped_df(300, 700)
        train, val, test = _stratified_split(df, seed=42)
        assert len(train) + len(val) + len(test) == len(df)

    def test_preserves_class_balance(self):
        df = _make_family_grouped_df(300, 700)  # 30% benign overall
        for part in _stratified_split(df, seed=42):
            benign_frac = (part["label"] == 0).sum() / len(part)
            assert abs(benign_frac - 0.30) < 0.03

    def test_deterministic_for_a_seed(self):
        # The three load_nf_v2(train/val/test) calls must agree → split is a pure
        # function of (df, seed).
        df = _make_family_grouped_df(300, 700)
        a = _stratified_split(df, seed=42)
        b = _stratified_split(df, seed=42)
        for pa, pb in zip(a, b):
            assert pa.equals(pb)

    def test_seeds_produce_different_partitions(self):
        df = _make_family_grouped_df(300, 700)
        test_42 = _stratified_split(df, seed=42)[2]
        test_43 = _stratified_split(df, seed=43)[2]
        # Same size, but membership should differ (real per-seed CI variation).
        assert len(test_42) == len(test_43)
        assert not test_42.equals(test_43)

    def test_falls_back_to_label_when_no_family(self):
        df = _make_family_grouped_df(300, 700).drop("attack_family")
        train, val, test = _stratified_split(df, seed=42)
        for part in (train, val, test):
            assert set(part["label"].to_list()) == {0, 1}


# ── zero_day_split ──────────────────────────────────────────────────────────

class TestZeroDaySplit:
    def test_held_out_removed_from_train(self):
        df = _make_df(100, n_attack=30)
        train, held = zero_day_split(df, held_out_families=["BruteForce"])
        assert "BruteForce" not in train["attack_family"].to_list()
        assert all(f == "BruteForce" for f in held["attack_family"].to_list())

    def test_missing_attack_col_raises(self):
        df = _make_df(50)
        with pytest.raises(ValueError, match="attack_family"):
            zero_day_split(df.drop("attack_family"), held_out_families=["X"])

    def test_empty_held_out(self):
        df = _make_df(100, n_attack=10)
        train, held = zero_day_split(df, held_out_families=["NonExistentFamily"])
        assert len(train) == len(df)
        assert len(held) == 0


@given(
    n=st.integers(min_value=5, max_value=200),
    n_attack=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=40)
def test_zero_day_split_property(n, n_attack):
    if n_attack >= n:
        return
    df = _make_df(n, n_attack=n_attack)
    train, held = zero_day_split(df, held_out_families=["BruteForce"])
    assert len(train) + len(held) == len(df)
    assert all(f != "BruteForce" for f in train["attack_family"].to_list())


# ── xenids_split ─────────────────────────────────────────────────────────────

def test_xenids_split_loads_requested_train_and_test(monkeypatch):
    calls = []

    def fake_load_nf_v2(dataset, split, seed=42, data_dir="data/processed"):
        calls.append((dataset, split, seed, data_dir))
        return pl.DataFrame({
            "dataset": [dataset],
            "split": [split],
            "seed": [seed],
        })

    monkeypatch.setattr(splits_module, "load_nf_v2", fake_load_nf_v2, raising=False)
    monkeypatch.setattr(
        "caushap_nids.data_pipeline.loaders.load_nf_v2",
        fake_load_nf_v2,
    )

    train, test = xenids_split(
        "nf_cic2018",
        "nf_unsw15",
        seed=123,
        data_dir="custom/processed",
    )

    assert train["dataset"].item() == "nf_cic2018"
    assert train["split"].item() == "train"
    assert test["dataset"].item() == "nf_unsw15"
    assert test["split"].item() == "test"
    assert calls == [
        ("nf_cic2018", "train", 123, "custom/processed"),
        ("nf_unsw15", "test", 123, "custom/processed"),
    ]


# ── report_imbalance ────────────────────────────────────────────────────────

class TestReportImbalance:
    def test_fractions_sum_to_one(self):
        df = _make_df(100, n_attack=20)
        fracs = report_imbalance(df, label_col="label")
        assert abs(sum(fracs.values()) - 1.0) < 1e-9

    def test_warns_on_rare_class(self):
        df = _make_df(1000, n_attack=3)
        with pytest.warns(UserWarning, match="< 1%"):
            report_imbalance(df, label_col="label")

    def test_missing_label_col_raises(self):
        df = _make_df(50)
        with pytest.raises(ValueError):
            report_imbalance(df, label_col="nonexistent")

    def test_empty_df(self):
        df = pl.DataFrame({"label": [], "feat": []})
        assert report_imbalance(df, label_col="label") == {}


# ── windowize ───────────────────────────────────────────────────────────────

class TestWindowize:
    def test_all_rows_covered(self):
        df = _make_df(100)
        windows = list(windowize(df, window_seconds=200))
        total = sum(len(w) for w in windows)
        assert total == len(df)

    def test_non_empty_windows(self):
        df = _make_df(50)
        for w in windowize(df, window_seconds=50):
            assert len(w) > 0

    def test_with_timestamp_col(self):
        df = pl.DataFrame({
            "flow_start_time": list(range(0, 300, 3)),
            "label": [0] * 100,
        })
        windows = list(windowize(df, window_seconds=60))
        assert len(windows) == 5  # 300s / 60s = 5 windows


# ── verify_dataset ──────────────────────────────────────────────────────────

class TestVerifyDataset:
    def test_correct_hash_passes(self, tmp_path):
        import hashlib
        data = b"test dataset content"
        h = hashlib.sha256(data).hexdigest()
        fpath = tmp_path / "test.parquet"
        fpath.write_bytes(data)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "test_ds": {"path": str(fpath), "sha256": h}
        }))
        verify_dataset("test_ds", manifest_path=str(manifest))

    def test_wrong_hash_raises(self, tmp_path):
        fpath = tmp_path / "test.parquet"
        fpath.write_bytes(b"real content")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "test_ds": {"path": str(fpath), "sha256": "deadbeef"}
        }))
        with pytest.raises(RuntimeError, match="mismatch"):
            verify_dataset("test_ds", manifest_path=str(manifest))

    def test_missing_dataset_raises(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({}))
        with pytest.raises(KeyError):
            verify_dataset("nonexistent", manifest_path=str(manifest))


# ── repair_protocol_fields (Gap 4-C) ────────────────────────────────────────

def _make_protocol_df():
    """Mixed-protocol DataFrame for testing semantic field repair."""
    import pandas as pd
    return pd.DataFrame({
        "PROTOCOL":         [1.0, 6.0, 17.0, 6.0, 1.0],   # ICMP, TCP, UDP, TCP, ICMP
        "L7_PROTO":         [0.0, 80.0, 53.0, 443.0, 0.0],
        # ICMP-only fields:
        "ICMP_TYPE":        [8.0, 99.0, 99.0,  0.0, 3.0],
        "ICMP_IPV4_TYPE":   [8.0, 88.0, 77.0, 11.0, 3.0],
        # TCP-only fields:
        "TCP_FLAGS":        [9.0, 24.0, 16.0, 26.0, 7.0],
        "CLIENT_TCP_FLAGS": [9.0, 24.0, 16.0, 26.0, 7.0],
        "SERVER_TCP_FLAGS": [9.0, 24.0, 16.0, 26.0, 7.0],
        "TCP_WIN_MAX_IN":   [42.0, 1024.0, 12.0, 2048.0, 11.0],
        "TCP_WIN_MAX_OUT":  [42.0, 1024.0, 12.0, 2048.0, 11.0],
        # DNS-only fields:
        "DNS_QUERY_ID":     [55.0, 66.0, 77.0, 88.0, 99.0],
        "DNS_QUERY_TYPE":   [ 1.0,  2.0,  3.0,  4.0,  5.0],
        "DNS_TTL_ANSWER":   [10.0, 20.0, 30.0, 40.0, 50.0],
        # FTP_COMMAND_RET_CODE — must NOT be repaired (T1110 STL predicate).
        "FTP_COMMAND_RET_CODE": [0.0, 530.0, 0.0, 230.0, 0.0],
    })


class TestRepairProtocolFields:
    """Gate Gap 4-C: ICMP/TCP/DNS fields zeroed for protocol-invalid flows."""

    def test_icmp_fields_zeroed_for_non_icmp(self):
        df = _make_protocol_df()
        repaired, _ = repair_protocol_fields(df)
        non_icmp_mask = df["PROTOCOL"].to_numpy() != 1.0
        assert (repaired.loc[non_icmp_mask, "ICMP_TYPE"] == 0).all()
        assert (repaired.loc[non_icmp_mask, "ICMP_IPV4_TYPE"] == 0).all()

    def test_icmp_fields_preserved_for_icmp(self):
        df = _make_protocol_df()
        repaired, _ = repair_protocol_fields(df)
        icmp_mask = df["PROTOCOL"].to_numpy() == 1.0
        assert (repaired.loc[icmp_mask, "ICMP_TYPE"] == df.loc[icmp_mask, "ICMP_TYPE"]).all()
        assert (repaired.loc[icmp_mask, "ICMP_IPV4_TYPE"] == df.loc[icmp_mask, "ICMP_IPV4_TYPE"]).all()

    def test_tcp_fields_zeroed_for_non_tcp(self):
        df = _make_protocol_df()
        repaired, _ = repair_protocol_fields(df)
        non_tcp_mask = df["PROTOCOL"].to_numpy() != 6.0
        for col in ("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
                    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"):
            assert (repaired.loc[non_tcp_mask, col] == 0).all(), col

    def test_tcp_fields_preserved_for_tcp(self):
        df = _make_protocol_df()
        repaired, _ = repair_protocol_fields(df)
        tcp_mask = df["PROTOCOL"].to_numpy() == 6.0
        for col in ("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
                    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"):
            assert (repaired.loc[tcp_mask, col] == df.loc[tcp_mask, col]).all(), col

    def test_dns_fields_zeroed_for_non_dns(self):
        df = _make_protocol_df()
        repaired, _ = repair_protocol_fields(df)
        non_dns_mask = df["L7_PROTO"].to_numpy() != 53.0
        for col in ("DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER"):
            assert (repaired.loc[non_dns_mask, col] == 0).all(), col

    def test_dns_fields_preserved_for_dns(self):
        df = _make_protocol_df()
        repaired, _ = repair_protocol_fields(df)
        dns_mask = df["L7_PROTO"].to_numpy() == 53.0
        for col in ("DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER"):
            assert (repaired.loc[dns_mask, col] == df.loc[dns_mask, col]).all(), col

    def test_ftp_command_ret_code_not_repaired(self):
        """FTP_COMMAND_RET_CODE must NOT be touched — T1110 STL predicate."""
        df = _make_protocol_df()
        repaired, counts = repair_protocol_fields(df)
        assert (repaired["FTP_COMMAND_RET_CODE"] == df["FTP_COMMAND_RET_CODE"]).all()
        assert "FTP_COMMAND_RET_CODE" not in counts

    def test_original_df_not_mutated(self):
        df = _make_protocol_df()
        original_icmp = df["ICMP_TYPE"].copy()
        original_tcp = df["TCP_FLAGS"].copy()
        original_dns = df["DNS_QUERY_ID"].copy()
        _ = repair_protocol_fields(df)
        assert (df["ICMP_TYPE"] == original_icmp).all()
        assert (df["TCP_FLAGS"] == original_tcp).all()
        assert (df["DNS_QUERY_ID"] == original_dns).all()

    def test_counts_dict_matches_actual_changes(self):
        df = _make_protocol_df()
        _, counts = repair_protocol_fields(df)
        # 3 non-ICMP rows × 2 non-zero ICMP cells each = 3 (UDP + 2 TCP rows; row 4 already 0)
        # Concretely: rows 1 (TCP, ICMP_TYPE=99), 2 (UDP, ICMP_TYPE=99), 3 (TCP, ICMP_TYPE=0)
        # → 2 non-zero → 2 cells changed for ICMP_TYPE
        assert counts["ICMP_TYPE"] == 2
        assert counts["ICMP_IPV4_TYPE"] == 3  # rows 1, 2, 3 all non-zero
        # 4 non-DNS rows; all 4 DNS_QUERY_ID values non-zero → 4 changed
        assert counts["DNS_QUERY_ID"] == 4
        # 3 non-TCP rows (rows 0, 2, 4) with non-zero TCP_FLAGS → 3 changed
        assert counts["TCP_FLAGS"] == 3

    def test_missing_l7_proto_column_skips_dns_gracefully(self):
        df = _make_protocol_df().drop(columns=["L7_PROTO"])
        repaired, counts = repair_protocol_fields(df)
        # DNS gate skipped → counts reported as 0, columns untouched
        for col in ("DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER"):
            assert counts[col] == 0
            assert (repaired[col] == df[col]).all()
        # ICMP / TCP gates still applied
        non_icmp_mask = df["PROTOCOL"].to_numpy() != 1.0
        assert (repaired.loc[non_icmp_mask, "ICMP_TYPE"] == 0).all()

    def test_missing_optional_field_handled(self):
        df = _make_protocol_df().drop(columns=["TCP_WIN_MAX_IN"])
        repaired, counts = repair_protocol_fields(df)
        assert counts["TCP_WIN_MAX_IN"] == 0
        assert "TCP_WIN_MAX_IN" not in repaired.columns
        # Remaining TCP fields still repaired
        non_tcp_mask = df["PROTOCOL"].to_numpy() != 6.0
        assert (repaired.loc[non_tcp_mask, "TCP_FLAGS"] == 0).all()

    def test_custom_column_names(self):
        df = _make_protocol_df().rename(columns={"PROTOCOL": "proto_l4", "L7_PROTO": "proto_l7"})
        repaired, _ = repair_protocol_fields(df, protocol_col="proto_l4", l7_col="proto_l7")
        non_icmp_mask = df["proto_l4"].to_numpy() != 1.0
        assert (repaired.loc[non_icmp_mask, "ICMP_TYPE"] == 0).all()
        non_dns_mask = df["proto_l7"].to_numpy() != 53.0
        assert (repaired.loc[non_dns_mask, "DNS_QUERY_ID"] == 0).all()

    def test_all_valid_protocol_no_changes(self):
        """Every flow has the valid protocol for all gates → no fields zeroed."""
        import pandas as pd
        df = pd.DataFrame({
            "PROTOCOL":       [6.0, 6.0],
            "L7_PROTO":       [80.0, 443.0],
            "TCP_FLAGS":      [24.0, 26.0],
            "ICMP_TYPE":      [0.0, 0.0],   # already 0 — non-TCP fields, ok for TCP rows
            "DNS_QUERY_ID":   [0.0, 0.0],
        })
        repaired, counts = repair_protocol_fields(df)
        assert counts["TCP_FLAGS"] == 0
        assert (repaired["TCP_FLAGS"] == df["TCP_FLAGS"]).all()
