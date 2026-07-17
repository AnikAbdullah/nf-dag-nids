from typing import Literal

import polars as pl


def temporal_split(
    df: pl.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    timestamp_col: str = "flow_start_time",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split by time order. Never shuffles — mandatory per Arp et al. (2022).

    If timestamp_col is present, sorts by it first and asserts temporal ordering.
    If absent, assumes the DataFrame is already in temporal order.
    """
    if timestamp_col in df.columns:
        df = df.sort(timestamp_col)

    n = len(df)
    t = int(train_frac * n)
    v = int((train_frac + val_frac) * n)

    train = df[:t].unique(maintain_order=True)
    val   = df[t:v].unique(maintain_order=True)
    test  = df[v:].unique(maintain_order=True)

    if timestamp_col in df.columns:
        train_max = train[timestamp_col].max()
        test_min  = test[timestamp_col].min()
        assert train_max < test_min, (
            f"Temporal split invariant violated: "
            f"train max ({train_max}) >= test min ({test_min})"
        )

    return train, val, test


def zero_day_split(
    df: pl.DataFrame,
    held_out_families: list[str],
    attack_col: str = "attack_family",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Remove held_out_families from training set.

    Returns (train_without_held_out, held_out_test).
    Used for zero-day recall evaluation.
    """
    if attack_col not in df.columns:
        raise ValueError(f"Column '{attack_col}' not found. Available: {df.columns}")

    mask_held = df[attack_col].is_in(held_out_families)
    return df.filter(~mask_held), df.filter(mask_held)


def xenids_split(
    train_dataset: Literal["nf_cic2018", "nf_unsw15", "5g_nidd", "edge_iiotset"],
    test_dataset: Literal["nf_cic2018", "nf_unsw15", "5g_nidd", "edge_iiotset"],
    seed: int = 42,
    data_dir: str = "data/processed",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cross-dataset evaluation split following Apruzzese (2022) XeNIDS protocol.

    Trains on train_dataset, tests on test_dataset.
    Returns (train_df, test_df).
    """
    from .loaders import load_nf_v2

    train_df = load_nf_v2(train_dataset, "train", seed=seed, data_dir=data_dir)
    test_df  = load_nf_v2(test_dataset,  "test",  seed=seed, data_dir=data_dir)
    return train_df, test_df
