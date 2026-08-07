"""Strip decoration glyphs (PASS-stickers / emoji) from frozen notebook cells.

SAFE version — only does literal string replacement. Does NOT collapse
whitespace, does NOT touch Python indentation. The earlier version of this
script collapsed any 2+ space run to a single space and damaged Python
indentation across five notebooks; recovery from VS Code History
snapshots + notebooks_old backups was needed. This rewrite is restricted
to the minimal glyph substitutions described below.

Substitutions applied to:
- code cell `source`
- code cell text outputs (`outputs[i].text` and `outputs[i].data["text/plain"]`)
- markdown cell `source`

Substitutions (literal, in order):
- "PASS ✓"   → "PASS"
- "FAIL ✗"   → "FAIL"
- "PASSED ✓" → "PASSED"
- "FAILED ✗" → "FAILED"
- "[PASS ✓]" → "[PASS]"
- "[FAIL ✗]" → "[FAIL]"
- " ✓"       → ""      (leading space + glyph → empty, when adjacent to a label)
- " ✗"       → ""
- "  ✓"      → ""
- "  ✗"      → ""
- "✅"       → "PASS"
- "❌"       → "FAIL"
- "✓ "       → ""      (status glyph followed by space)
- "✗ "       → ""

The bare "✓" and "✗" alone are LEFT in place to avoid accidentally hitting
non-status uses; the patterns above always include a delimiter so they only
target status decorations. `⚠️` admonitions in markdown are NOT touched.

Run via `python scripts/strip_emojis_for_screenshot.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
TARGETS = [
    "02_dag_construction.ipynb",
    "03_stl_typing.ipynb",
    "04_layer_a_causal_shapley.ipynb",
    "05_layer_b_multi_obj_cf.ipynb",
    "06_layer_c_concept_abduction.ipynb",
]

REPLACEMENTS = [
    ("PASS ✓",   "PASS"),
    ("FAIL ✗",   "FAIL"),
    ("PASSED ✓", "PASSED"),
    ("FAILED ✗", "FAILED"),
    ("[PASS ✓]", "[PASS]"),
    ("[FAIL ✗]", "[FAIL]"),
    ("'PASS ✓'", "'PASS'"),
    ("'FAIL ✗'", "'FAIL'"),
    ('"PASS ✓"', '"PASS"'),
    ('"FAIL ✗"', '"FAIL"'),
    ("✓ FROZEN", "FROZEN"),
    ("  ✓",      ""),
    (" ✓",       ""),
    ("  ✗",      ""),
    (" ✗",       ""),
    ("✓ ",       ""),
    ("✗ ",       ""),
    ("✅",       "PASS"),
    ("❌",       "FAIL"),
]


def _apply(s: str) -> str:
    out = s
    for old, new in REPLACEMENTS:
        out = out.replace(old, new)
    return out


def _process_text(value):
    if isinstance(value, list):
        joined = "".join(value)
        new = _apply(joined)
        if new == joined:
            return value, False
        # keep trailing newline behaviour by splitting again
        # but a single-string list is fine for ipynb format
        return [new], True
    new = _apply(value)
    return new, new != value


def main() -> None:
    audit_lines: list[str] = [
        "# Screenshot cleanup audit trail (safe version)\n",
        (
            "Task 3 of `plans/go-and-beat-if-optimized-hopper.md`. "
            "Removes PASS-sticker decoration glyphs (✓ / ✗ / ✅ / ❌) "
            "next to status labels so the user can screenshot notebook "
            "outputs for the Q1 paper without emoji noise. Cells are NOT "
            "re-executed; only source strings and text outputs are edited. "
            "This is the SAFE version — whitespace is preserved.\n"
        ),
    ]
    total = 0
    for nb_name in TARGETS:
        nb_path = NB_DIR / nb_name
        nb = json.loads(nb_path.read_text())
        n_changed = 0
        per_cell: list[str] = []
        for i, cell in enumerate(nb["cells"]):
            touched = False
            if "source" in cell:
                new_src, did = _process_text(cell["source"])
                if did:
                    cell["source"] = new_src
                    touched = True
            if cell["cell_type"] == "code":
                for out in cell.get("outputs", []) or []:
                    if "text" in out:
                        new_text, did = _process_text(out["text"])
                        if did:
                            out["text"] = new_text
                            touched = True
                    if "data" in out:
                        for k in list(out["data"].keys()):
                            if k.startswith("text/"):
                                new_val, did = _process_text(out["data"][k])
                                if did:
                                    out["data"][k] = new_val
                                    touched = True
            if touched:
                n_changed += 1
                per_cell.append(f"  - cell {i} ({cell['cell_type']})")
        if n_changed:
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
            audit_lines.append(f"\n## `{nb_name}` — {n_changed} cell(s) cleaned\n")
            audit_lines.extend(per_cell)
            total += n_changed
        else:
            audit_lines.append(f"\n## `{nb_name}` — no changes needed\n")
        print(f"  {nb_name}: {n_changed} cell(s) cleaned")
    audit_path = ROOT / "artifacts" / "SCREENSHOT_CLEANUP.md"
    audit_path.write_text("\n".join(audit_lines) + "\n")
    print(f"\nTotal cells touched: {total}")
    print(f"Audit written to {audit_path}")


if __name__ == "__main__":
    main()
