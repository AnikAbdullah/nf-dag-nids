import warnings

import polars as pl


def report_imbalance(
    df: pl.DataFrame,
    label_col: str = "label",
) -> dict[str, float]:
    """Return {class_name: fraction} dict. Warns if any class < 1% of total."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {df.columns}")

    total = len(df)
    if total == 0:
        return {}

    counts = (
        df.group_by(label_col)
        .agg(pl.len().alias("count"))
        .sort(label_col)
    )

    fractions: dict[str, float] = {}
    for row in counts.iter_rows(named=True):
        cls  = str(row[label_col])
        frac = row["count"] / total
        fractions[cls] = frac
        if frac < 0.01:
            warnings.warn(
                f"Class '{cls}' represents {frac:.2%} of data (< 1%). "
                "Consider impact on evaluation metrics.",
                stacklevel=2,
            )

    return fractions
