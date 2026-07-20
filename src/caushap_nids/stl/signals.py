"""Extract numpy signal arrays from a Polars window DataFrame.

Works on RAW NF-v2 feature values (not preprocessed).  Every value returned
is a float64 array of shape (n,) where n = len(window), preserving row order.
Missing columns (unlikely but safe) produce a column of zeros.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .evaluator import Signals

# Columns we need for STL evaluation.  The list is exhaustive across all 4
# MITRE techniques; individual formulae pick what they need by signal name.
_REQUIRED_COLS: tuple[str, ...] = (
    "L4_DST_PORT",
    "L4_SRC_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "FLOW_DURATION_MILLISECONDS",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "FTP_COMMAND_RET_CODE",
    "MIN_TTL",
    "MAX_TTL",
)


def extract_signals(window: pl.DataFrame) -> Signals:
    """Return a Signals dict from a raw NF-v2 window DataFrame.

    All values are float64; absent columns are filled with 0.0.
    Empty windows return empty arrays of shape (0,).
    """
    n = len(window)
    signals: Signals = {}
    for col in _REQUIRED_COLS:
        if col in window.columns:
            signals[col] = window[col].to_numpy(allow_copy=True).astype(np.float64)
        else:
            signals[col] = np.zeros(n, dtype=np.float64)
    return signals
