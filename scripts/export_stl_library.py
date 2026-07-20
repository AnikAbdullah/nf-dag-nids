"""Export the registered MITRE STL library to artifacts/stl_library.yaml.

Required by v2 plan §3 (reproducibility artifacts) and §7 (acceptance criteria
"All formulas serialisable to/from YAML without loss").  The output is a
machine-readable AST that round-trips through
``caushap_nids.stl.library.{dump_library, load_library}``.

Run from project root:
    .venv/bin/python scripts/export_stl_library.py
"""
from __future__ import annotations

from pathlib import Path

from caushap_nids.stl import (
    ALL_TECHNIQUES,
    dump_library,
    load_library,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "artifacts" / "stl_library.yaml"


def main() -> None:
    text = dump_library(ALL_TECHNIQUES, OUT_PATH)
    n_bytes = len(text.encode("utf-8"))

    # Round-trip sanity check on the just-written file
    reloaded = load_library(OUT_PATH)
    assert len(reloaded) == len(ALL_TECHNIQUES), (
        f"round-trip lost techniques: {len(reloaded)} vs {len(ALL_TECHNIQUES)}"
    )
    for orig, rt in zip(ALL_TECHNIQUES, reloaded):
        assert orig.mitre_id == rt.mitre_id
        assert orig.name == rt.name
        assert orig.target_families == rt.target_families
        assert orig.min_window_size == rt.min_window_size
        assert orig.formula == rt.formula, (
            f"formula AST mismatch for {orig.mitre_id}"
        )

    print(f"✓ wrote {OUT_PATH.relative_to(ROOT)}  ({n_bytes} bytes, "
          f"{len(ALL_TECHNIQUES)} techniques)")
    for t in ALL_TECHNIQUES:
        print(f"    {t.mitre_id:12s} {t.name}")


if __name__ == "__main__":
    main()
