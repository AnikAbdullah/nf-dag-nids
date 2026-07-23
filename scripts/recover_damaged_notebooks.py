"""Recover damaged notebooks (one-shot).

Restoration sources:
- 03_stl_typing.ipynb        ← VS Code History (May 17 03:02, same 18 cells)
- 04_layer_a_causal_shapley  ← VS Code History (May 17 03:45, same 20 cells)
- 06_layer_c_concept_abduction ← VS Code History (May 17 05:36, same 23 cells)
- 05_layer_b_multi_obj_cf    ← VS Code History (May 17 04:05, 17 cells) for
                                cells 0..16; KEEP damaged cur cells 17..18
                                (w17_md_01, w17_code_01) — recently added
- 02_dag_construction        ← notebooks_old (May 15 14:58, 22 cells) for
                                cells 0..21; KEEP damaged cur cells 22..25
                                (Gap 2-A and Gap 2-B additions)

Damaged-cell repair for cells that we must keep from current (02:23, 02:25,
05:18, plus any others) is handled in a SEPARATE script after this one runs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
VS_HIST = Path("/Users/winterfell/Library/Application Support/Code/User/History")
OLD_DIR = Path(
    "/Users/winterfell/Education/Academic/Research Projects/Data Mining "
    "Project/causal-shap-nids/notebooks_old"
)

RECOVERY_PLAN: list[dict] = [
    {
        "target": NB_DIR / "03_stl_typing.ipynb",
        "source": VS_HIST / "30b5980b/F9b5.ipynb",
        "strategy": "full",
        "keep_cur_range": None,
    },
    {
        "target": NB_DIR / "04_layer_a_causal_shapley.ipynb",
        "source": VS_HIST / "5014456f/DV9z.ipynb",
        "strategy": "full",
        "keep_cur_range": None,
    },
    {
        "target": NB_DIR / "06_layer_c_concept_abduction.ipynb",
        "source": VS_HIST / "1516497/NiVG.ipynb",
        "strategy": "full",
        "keep_cur_range": None,
    },
    {
        "target": NB_DIR / "05_layer_b_multi_obj_cf.ipynb",
        "source": VS_HIST / "-336c6a26/9w7I.ipynb",
        "strategy": "splice",
        # restore source[0..16], append cur[17..18]
        "restore_from_source_range": (0, 17),
        "keep_cur_range": (17, 19),
    },
    {
        "target": NB_DIR / "02_dag_construction.ipynb",
        "source": OLD_DIR / "02_dag_construction.ipynb",
        "strategy": "splice",
        # restore source[0..21], append cur[22..25]
        "restore_from_source_range": (0, 22),
        "keep_cur_range": (22, 26),
    },
]


def recover_full(target: Path, source: Path) -> None:
    """Copy source verbatim to target."""
    shutil.copy2(source, target)


def recover_splice(target: Path, source: Path, src_range: tuple[int, int],
                   cur_range: tuple[int, int]) -> None:
    """Combine source[src_range] + current_damaged[cur_range].

    Preserves the notebook's top-level metadata (nbformat, kernelspec) from
    the SOURCE (clean, pristine notebook).
    """
    src_nb = json.loads(source.read_text())
    cur_nb = json.loads(target.read_text())

    new_cells = list(src_nb["cells"][src_range[0]:src_range[1]])
    new_cells.extend(cur_nb["cells"][cur_range[0]:cur_range[1]])

    # Keep src_nb's top-level metadata (was the pristine state)
    out = dict(src_nb)
    out["cells"] = new_cells

    target.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    for plan in RECOVERY_PLAN:
        target = plan["target"]
        source = plan["source"]
        print(f"\n[recover] {target.name}")
        print(f"  source: {source}")
        print(f"  strategy: {plan['strategy']}")
        if not source.exists():
            print(f"  ERROR: source not found, skipping")
            continue
        if plan["strategy"] == "full":
            recover_full(target, source)
            n = len(json.loads(target.read_text())["cells"])
            print(f"  -> wrote {target.name} ({n} cells)")
        elif plan["strategy"] == "splice":
            recover_splice(
                target, source,
                plan["restore_from_source_range"],
                plan["keep_cur_range"],
            )
            n = len(json.loads(target.read_text())["cells"])
            print(f"  -> wrote {target.name} ({n} cells, spliced)")


if __name__ == "__main__":
    main()
