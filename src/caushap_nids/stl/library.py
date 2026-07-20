"""YAML serialization for the MITRE STL library.

The released `artifacts/stl_library.yaml` is a faithful, machine-readable AST
of every registered :class:`MitreTechnique`.  Each formula is encoded as a
nested dict with `type` and either `sig/thr/val` (leaves) or `children/phi`
(inner nodes), mirroring the dataclass structure of
:mod:`caushap_nids.stl.evaluator`.

Round-trip guarantee
────────────────────
``load_library(dump_library(ALL_TECHNIQUES))`` returns objects equal-by-value
to the originals; ``test_stl::TestLibraryYaml`` enforces this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .evaluator import (
    STLNode,
    Ge, Le, Eq, Neg, And, Or, Globally, Eventually,
)
from .formulae import MitreTechnique, ALL_TECHNIQUES


# ── STLNode AST <-> dict ──────────────────────────────────────────────────────

def node_to_dict(node: STLNode) -> dict[str, Any]:
    if isinstance(node, Ge):
        return {"type": "ge", "sig": node.sig, "thr": float(node.thr)}
    if isinstance(node, Le):
        return {"type": "le", "sig": node.sig, "thr": float(node.thr)}
    if isinstance(node, Eq):
        return {"type": "eq", "sig": node.sig, "val": float(node.val)}
    if isinstance(node, Neg):
        return {"type": "neg", "phi": node_to_dict(node.phi)}
    if isinstance(node, And):
        return {"type": "and", "children": [node_to_dict(c) for c in node.children]}
    if isinstance(node, Or):
        return {"type": "or", "children": [node_to_dict(c) for c in node.children]}
    if isinstance(node, Globally):
        return {"type": "globally", "phi": node_to_dict(node.phi)}
    if isinstance(node, Eventually):
        return {"type": "eventually", "phi": node_to_dict(node.phi)}
    raise TypeError(f"unsupported STLNode: {type(node).__name__}")


def node_from_dict(d: dict[str, Any]) -> STLNode:
    t = d["type"]
    if t == "ge":
        return Ge(sig=d["sig"], thr=float(d["thr"]))
    if t == "le":
        return Le(sig=d["sig"], thr=float(d["thr"]))
    if t == "eq":
        return Eq(sig=d["sig"], val=float(d["val"]))
    if t == "neg":
        return Neg(phi=node_from_dict(d["phi"]))
    if t == "and":
        return And(children=tuple(node_from_dict(c) for c in d["children"]))
    if t == "or":
        return Or(children=tuple(node_from_dict(c) for c in d["children"]))
    if t == "globally":
        return Globally(phi=node_from_dict(d["phi"]))
    if t == "eventually":
        return Eventually(phi=node_from_dict(d["phi"]))
    raise ValueError(f"unknown STL node type: {t!r}")


# ── MitreTechnique <-> dict ───────────────────────────────────────────────────

def technique_to_dict(t: MitreTechnique) -> dict[str, Any]:
    return {
        "mitre_id": t.mitre_id,
        "name": t.name,
        "target_families": sorted(t.target_families),
        "min_window_size": int(t.min_window_size),
        "formula": node_to_dict(t.formula),
    }


def technique_from_dict(d: dict[str, Any]) -> MitreTechnique:
    return MitreTechnique(
        mitre_id=d["mitre_id"],
        name=d["name"],
        target_families=frozenset(d["target_families"]),
        min_window_size=int(d.get("min_window_size", 3)),
        formula=node_from_dict(d["formula"]),
    )


# ── Library-level YAML I/O ────────────────────────────────────────────────────

LIBRARY_VERSION = "1.0"


def dump_library(
    techniques: tuple[MitreTechnique, ...] = ALL_TECHNIQUES,
    path: str | Path | None = None,
) -> str:
    payload = {
        "version": LIBRARY_VERSION,
        "techniques": [technique_to_dict(t) for t in techniques],
    }
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


def load_library(path: str | Path) -> tuple[MitreTechnique, ...]:
    payload = yaml.safe_load(Path(path).read_text())
    return tuple(technique_from_dict(t) for t in payload["techniques"])
