"""W22 gate CLI — run STL typing evaluation on the test split.

Usage
─────
  PYTHONPATH=src python -m caushap_nids.stl.gate \\
      --data-dir data/processed \\
      --window-size 50 \\
      --out artifacts/stl_gate_result.json

Exit codes
──────────
  0 — gate PASSED (precision >= 0.95 on >= 2 / 4 techniques)
  1 — gate FAILED
  2 — dataset not found (missing data, skip gracefully)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .metrics import evaluate_test_split, format_gate_result, save_gate_result
from .formulae import ALL_TECHNIQUES


def _load_test_split(data_dir: str) -> "pl.DataFrame":
    """Attempt to load the NF-CSE-CIC-IDS2018 test split."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from caushap_nids.data_pipeline.loaders import load_nf_v2
        return load_nf_v2("nf_cic2018", "test", data_dir=data_dir)
    except FileNotFoundError as exc:
        print(f"[W22] Dataset not found: {exc}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W22 STL attack-typing gate for caushap_nids (Module 4)."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Root directory containing processed dataset splits.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Number of flows per positional evaluation window (default: 50).",
    )
    parser.add_argument(
        "--majority-threshold",
        type=float,
        default=0.50,
        help="Fraction of flows needed for an unambiguous window label (default: 0.5).",
    )
    parser.add_argument(
        "--out",
        default="artifacts/stl_gate_result.json",
        help="Path to write the gate result JSON.",
    )
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=0.95,
        help="Per-technique precision required to count as passing (default: 0.95).",
    )
    args = parser.parse_args()

    try:
        df = _load_test_split(args.data_dir)
    except FileNotFoundError:
        raise SystemExit(2)

    result = evaluate_test_split(
        df,
        techniques=ALL_TECHNIQUES,
        window_size=args.window_size,
        majority_threshold=args.majority_threshold,
    )
    print(format_gate_result(result))

    save_gate_result(result, args.out)
    print(f"\nResult written to: {args.out}")

    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
