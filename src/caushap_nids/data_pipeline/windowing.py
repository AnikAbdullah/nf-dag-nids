from collections.abc import Iterator

import polars as pl


def windowize(
    df: pl.DataFrame,
    window_seconds: int = 60,
    group_cols: list[str] | None = None,
    timestamp_col: str = "flow_start_time",
) -> Iterator[pl.DataFrame]:
    """Group flows into non-overlapping time windows for STL evaluation.

    Each yielded DataFrame contains flows within window_seconds of the window start.
    Temporal order within each window is preserved.

    If timestamp_col is absent, falls back to positional chunking (for datasets
    without explicit timestamps — flow order assumed to be temporal).
    """
    if group_cols is None:
        group_cols = []

    if timestamp_col not in df.columns:
        yield from _positional_windows(df, window_seconds)
        return

    df = df.sort(timestamp_col)

    if group_cols:
        for _, group_df in df.group_by(group_cols, maintain_order=True):
            yield from _time_windows(group_df, window_seconds, timestamp_col)
    else:
        yield from _time_windows(df, window_seconds, timestamp_col)


def _time_windows(
    df: pl.DataFrame,
    window_seconds: int,
    timestamp_col: str,
) -> Iterator[pl.DataFrame]:
    if len(df) == 0:
        return

    ts = df[timestamp_col]
    t_start = ts.min()
    t_end   = ts.max()
    t_cur   = t_start

    while t_cur <= t_end:
        t_next = t_cur + window_seconds
        mask   = (ts >= t_cur) & (ts < t_next)
        chunk  = df.filter(mask)
        if len(chunk) > 0:
            yield chunk
        t_cur = t_next


def _positional_windows(df: pl.DataFrame, rows_per_window: int) -> Iterator[pl.DataFrame]:
    for start in range(0, len(df), rows_per_window):
        chunk = df[start : start + rows_per_window]
        if len(chunk) > 0:
            yield chunk
