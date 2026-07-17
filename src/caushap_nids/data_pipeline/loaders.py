from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd

NF_V2_FEATURE_COLS: list[str] = [
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

_DATASET_FILES: dict[str, str] = {
    "nf_cic2018":   "NF-CSE-CIC-IDS2018-V2.parquet",
    "nf_unsw15":    "NF-UNSW-NB15-v2.parquet",
    "5g_nidd":      "5G-NIDD.parquet",
    "edge_iiotset": "Edge-IIoTset.parquet",
}

_SPLIT_FILES: dict[str, str] = {
    "train": "train.parquet",
    "val":   "val.parquet",
    "test":  "test.parquet",
}

# Datasets whose raw parquet is NOT in packet/flow capture order and therefore
# cannot be split temporally. Edge-IIoTset and 5G-NIDD ship as ML-curated CSVs
# concatenated by attack family with no usable capture timestamp, so a positional
# 70/15/15 cut yields single-class val/test splits (e.g. Edge test = 100% attack),
# which collapses macro-F1 to ~0.5 and makes ROC-AUC undefined. For these we use a
# seeded, family-stratified split instead (see _stratified_split). nf_cic2018 and
# nf_unsw15 are genuine capture-ordered NetFlow exports and keep the temporal
# (positional) split — never shuffle them (no-leakage, Arp et al. 2022).
_STRATIFIED_SPLIT_DATASETS: frozenset[str] = frozenset({"edge_iiotset", "5g_nidd"})


def load_nf_v2(
    dataset: Literal["nf_cic2018", "nf_unsw15", "5g_nidd", "edge_iiotset"],
    split: Literal["train", "val", "test"],
    seed: int = 42,
    data_dir: str = "data/processed",
) -> pl.DataFrame:
    """Load a pre-processed NF-v2 dataset split.

    Output always has exactly the available NF-v2 feature columns + 'label' + 'attack_family'.
    Raises FileNotFoundError if dataset not found.
    Raises ValueError if schema does not match NF-v2 standard.
    """
    base = Path(data_dir)
    split_path = base / dataset / _SPLIT_FILES[split]
    if not split_path.exists():
        # Fallback: single raw parquet not yet split
        raw_path = base / _DATASET_FILES[dataset]
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {split_path}\n"
                f"Also tried: {raw_path}\n"
                f"Download from Sarhan 2022 Kaggle release and place in {base}/"
            )
        df = pl.read_parquet(raw_path)
        df = _normalise_schema(df, dataset)
        if dataset in _STRATIFIED_SPLIT_DATASETS:
            train_df, val_df, test_df = _stratified_split(df, seed=seed)
        else:
            train_df, val_df, test_df = _positional_split(df)
        splits = {"train": train_df, "val": val_df, "test": test_df}
        return splits[split]

    df = pl.read_parquet(split_path)
    return _normalise_schema(df, dataset)


def _normalise_schema(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Rename label column to standard names and validate NF-v2 features are present."""
    rename_map: dict[str, str] = {}
    if "Label" in df.columns and "label" not in df.columns:
        rename_map["Label"] = "label"
    if "Attack" in df.columns and "attack_family" not in df.columns:
        rename_map["Attack"] = "attack_family"
    if rename_map:
        df = df.rename(rename_map)

    present = set(df.columns)
    missing = [f for f in NF_V2_FEATURE_COLS if f not in present]
    if missing:
        raise ValueError(
            f"Dataset '{dataset}' is missing NF-v2 features: {missing}\n"
            f"Expected all features from NF_V2_FEATURE_COLS."
        )
    return df


def _positional_split(
    df: pl.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = len(df)
    t = int(train_frac * n)
    v = int((train_frac + val_frac) * n)
    return df[:t], df[t:v], df[v:]


def _stratified_split(
    df: pl.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Seeded, class-stratified 70/15/15 split for non-temporal datasets.

    Used only for datasets in ``_STRATIFIED_SPLIT_DATASETS`` (Edge-IIoTset,
    5G-NIDD), whose raw parquet is grouped by attack family with no usable
    capture timestamp — a positional cut there produces single-class splits.

    Each stratum (attack_family if present, else label) is shuffled with a
    seed-derived RNG and sliced proportionally, so every split keeps the global
    class/family balance and contains both benign and attack rows. The result is
    a deterministic function of ``(df, seed)``: the three ``load_nf_v2`` calls
    (train/val/test) for one seed agree, while seeds 42/43/44 yield genuinely
    different partitions (real per-seed CIs).
    """
    n = len(df)
    strat_col = "attack_family" if "attack_family" in df.columns else "label"
    strata = np.asarray(df[strat_col].to_numpy())
    rng = np.random.default_rng(seed)

    # 0 = train, 1 = val, 2 = test
    assign = np.empty(n, dtype=np.int8)
    for value in sorted(np.unique(strata).tolist()):  # fixed order → reproducible
        rows = np.flatnonzero(strata == value)
        rng.shuffle(rows)
        m = len(rows)
        t = int(round(train_frac * m))
        v = int(round((train_frac + val_frac) * m))
        assign[rows[:t]] = 0
        assign[rows[t:v]] = 1
        assign[rows[v:]] = 2

    df = df.with_columns(pl.Series("__split_assign", assign))
    train_df = df.filter(pl.col("__split_assign") == 0).drop("__split_assign")
    val_df   = df.filter(pl.col("__split_assign") == 1).drop("__split_assign")
    test_df  = df.filter(pl.col("__split_assign") == 2).drop("__split_assign")
    return train_df, val_df, test_df


# ── Protocol-semantic repair ──────────────────────────────────────────────────
# Maps (canonical_indicator_column, valid_L4/L7_value) -> NF-v2 fields to zero
# when the flow's indicator does NOT equal that value.
#
# Reading guide:
#   ("PROTOCOL", 1.0, (...))  → ICMP_TYPE and ICMP_IPV4_TYPE are only
#                               meaningful when PROTOCOL == 1 (ICMP).
#   ("PROTOCOL", 6.0, (...))  → TCP flag and window fields are only meaningful
#                               when PROTOCOL == 6 (TCP).
#   ("L7_PROTO", 53.0, (...)) → DNS meta-fields are only meaningful when the
#                               detected application layer is DNS (L7 code 53).
#
# FTP_COMMAND_RET_CODE is intentionally excluded: L7_PROTO=21 detection in
# NF-CSE-CIC-IDS2018-v2 is unreliable (many brute-force flows show L7_PROTO=0
# even when FTP credentials are being attacked).  Zeroing FTP_COMMAND_RET_CODE
# on the basis of L7_PROTO would silently discard the key T1110 STL predicate.
_PROTOCOL_GATES: list[tuple[str, float, tuple[str, ...]]] = [
    (
        "PROTOCOL", 1.0,
        ("ICMP_TYPE", "ICMP_IPV4_TYPE"),
    ),
    (
        "PROTOCOL", 6.0,
        ("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
         "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"),
    ),
    (
        "L7_PROTO", 53.0,
        ("DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER"),
    ),
]


def repair_protocol_fields(
    df: pd.DataFrame,
    *,
    protocol_col: str = "PROTOCOL",
    l7_col: str = "L7_PROTO",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Zero NF-v2 fields that are semantically invalid for a flow's L4/L7 protocol.

    Raw NF-v2 exports sometimes store non-zero ICMP, TCP-flag, or DNS values on
    flows whose L4/L7 protocol does not support those fields.  Leaving them in
    place creates spurious causal attributions: the DAG edge PROTOCOL → ICMP_TYPE
    correctly encodes the dependency, yet a TCP flow with ICMP_TYPE != 0 produces
    non-zero Shapley mass on a feature that is causally disconnected for that flow.
    The same artefact inflates counterfactual infeasibility scores in Module 5b
    and concept-activation false positives in Module 5c.

    Three repair rules are applied, all fully vectorised:

    * ICMP  (PROTOCOL == 1): zero ICMP_TYPE, ICMP_IPV4_TYPE for every
                             non-ICMP flow.
    * TCP   (PROTOCOL == 6): zero TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS,
                             TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT for every non-TCP flow.
    * DNS   (L7_PROTO == 53): zero DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER
                              for every non-DNS flow.

    FTP_COMMAND_RET_CODE is intentionally *not* repaired — see module-level
    comment on ``_PROTOCOL_GATES`` for the rationale.

    Parameters
    ----------
    df:
        Pandas DataFrame of raw NF-v2 flow records.  Must contain ``PROTOCOL``
        (or the column named by ``protocol_col``).  ``L7_PROTO`` (or ``l7_col``)
        is optional — DNS repair is skipped when the column is absent.
    protocol_col:
        Name of the L4-protocol column.  Default ``"PROTOCOL"``.
    l7_col:
        Name of the L7-protocol column.  Default ``"L7_PROTO"``.

    Returns
    -------
    repaired : pd.DataFrame
        Full copy of ``df`` with protocol-invalid fields zeroed.
        The original DataFrame is never mutated.
    counts : dict[str, int]
        ``{field_name: n_cells_changed}`` — number of previously non-zero cells
        set to zero per field.  Use for paper diagnostics and audit logging.
    """
    import pandas as pd  # local to keep polars-only module-level imports clean

    repaired: pd.DataFrame = df.copy()
    counts: dict[str, int] = {}

    # Resolve custom column names against the canonical gate definitions.
    col_alias: dict[str, str] = {"PROTOCOL": protocol_col, "L7_PROTO": l7_col}

    for canonical, valid_val, fields in _PROTOCOL_GATES:
        indicator = col_alias.get(canonical, canonical)
        if indicator not in repaired.columns:
            # Whole gate skipped — column absent (e.g. L7_PROTO missing).
            for f in fields:
                counts[f] = 0
            continue

        # Boolean mask: True where the protocol is NOT the valid one for these fields.
        not_valid: np.ndarray = (
            repaired[indicator].to_numpy(dtype=np.float64) != valid_val
        )
        if not not_valid.any():
            for f in fields:
                counts[f] = 0
            continue

        for field in fields:
            if field not in repaired.columns:
                counts[field] = 0
                continue

            arr: np.ndarray = repaired[field].to_numpy(dtype=np.float64)
            # Count only cells that were non-zero before we zero them.
            n_changed = int(np.count_nonzero(arr[not_valid]))
            counts[field] = n_changed

            if n_changed:
                # Copy before mutation so we never alias the original array.
                arr = arr.copy()
                arr[not_valid] = 0.0
                repaired[field] = arr
            # If n_changed == 0 all invalid-protocol cells are already 0;
            # skip the copy+assign for performance.

    return repaired, counts
