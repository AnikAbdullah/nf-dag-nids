"""Markdown + JSON renderers for the campaign-report supplement.

The markdown form is what the paper supplement table consumes; JSON is the
deterministic reproducibility artifact (round-trips through ``json.loads``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schema import CampaignReport


_HEADER = (
    "| Campaign | Window | Flows | STL technique(s) | Concept(s) | "
    "Top causal features | Max score | Summary |\n"
    "|---|---|---:|---|---|---|---:|---|"
)


def _markdown_row(r: CampaignReport) -> str:
    techs = ", ".join(r.dominant_stl_techniques) if r.dominant_stl_techniques else "—"
    concs = ", ".join(r.dominant_concepts) if r.dominant_concepts else "—"
    feats = ", ".join(r.top_causal_features) if r.top_causal_features else "—"
    summary = r.summary.replace("|", "\\|")
    return (
        f"| {r.campaign_id} | {r.window_start} → {r.window_end} "
        f"| {r.flow_count} | {techs} | {concs} | {feats} "
        f"| {r.max_anomaly_score:.4f} | {summary} |"
    )


def campaign_reports_to_markdown(reports: Iterable[CampaignReport]) -> str:
    """Render a markdown table for the paper supplement.

    The output starts with the canonical header even when there are zero
    rows, so downstream tooling can render a stable placeholder.
    """
    lines = [_HEADER]
    rows = sorted(reports, key=lambda r: (r.window_start, r.src_ip, r.dst_ip))
    lines.extend(_markdown_row(r) for r in rows)
    return "\n".join(lines)


def campaign_reports_to_json(
    reports: Iterable[CampaignReport],
    output_path: str | Path,
) -> None:
    """Write a deterministic JSON file (sorted keys, stable ordering)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(reports, key=lambda r: (r.window_start, r.src_ip, r.dst_ip))
    payload = {"campaigns": [r.to_dict() for r in rows]}
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
