"""Group per-flow explanations into campaign-level summaries.

A "campaign" is the set of explained flows that share a (src_ip, dst_ip) pair
and fall within the same time bucket (default 10 minutes).  Apruzzese et al.
(SoK, 2026) argue that SOC analysts triage *reports*, not isolated alerts, so
this module exists to consolidate the per-flow Causal Shapley + STL + Concept
outputs into a paper-supplement-ready table.

Pure post-processing: no model retraining, no new third-party dependency
beyond stdlib + the existing :class:`FlowExplanationRecord` shape.

Determinism
───────────
For a fixed input order the output campaign list is deterministic — bucket
order follows ``sorted(group_keys)`` and within a campaign the dominant
techniques / concepts / features are sorted by descending count with ties
broken alphabetically.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from .schema import FlowExplanationRecord, CampaignReport


def _parse_ts(s: str) -> datetime:
    """Best-effort ISO-8601 parse; falls back to a stable hash bucket on failure.

    The reporting layer must not crash on minor timestamp-format drift between
    datasets, so unparseable strings collapse into the epoch — they still
    aggregate deterministically, just into a single shared bucket.
    """
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _bucket(ts: datetime, minutes: int) -> datetime:
    floor = ts.replace(second=0, microsecond=0)
    floor -= timedelta(minutes=floor.minute % minutes)
    return floor


def _top_k_sorted(counter: Counter[str], k: int) -> list[str]:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k_ for k_, _ in items[:k]]


def _campaign_summary(
    src_ip: str,
    dst_ip: str,
    flow_count: int,
    dominant_techniques: list[str],
    dominant_concepts: list[str],
    top_features: list[str],
) -> str:
    """Short natural-language summary line for the markdown supplement."""
    tech_str = ", ".join(dominant_techniques) if dominant_techniques else "no STL match"
    concept_str = " + ".join(dominant_concepts) if dominant_concepts else "no concept fired"
    feat_str = ", ".join(top_features[:3]) if top_features else "—"
    return (
        f"Host {src_ip} → {dst_ip} shows {tech_str} over {flow_count} flows; "
        f"concepts: {concept_str}; top causal features: {feat_str}."
    )


def aggregate_campaigns(
    records: Iterable[FlowExplanationRecord],
    *,
    window_minutes: int = 10,
    min_flows: int = 3,
    top_k_features: int = 5,
    top_k_techniques: int = 3,
    top_k_concepts: int = 3,
) -> list[CampaignReport]:
    """Bucket records by (src_ip, dst_ip, time_window) and emit a report each.

    Parameters
    ──────────
    records:
        Iterable of per-flow explanation records.  No global state is shared
        across calls; iteration is single-pass.
    window_minutes:
        Time-bucket width.  10 min is the v2 plan default for the paper
        supplement; SOC analysts can adjust at call-time.
    min_flows:
        Drop campaigns smaller than this — they are isolated alerts, not
        campaigns.  Default 3 matches plan §11A.
    top_k_features:
        Number of top causal features carried into the report.
    top_k_techniques / top_k_concepts:
        Limit on the technique / concept lists shown in the dominant fields.

    Returns
    ──────
    Deterministic list of :class:`CampaignReport` sorted by
    (window_start, src_ip, dst_ip).
    """
    buckets: dict[
        tuple[str, str, datetime],
        list[FlowExplanationRecord],
    ] = {}
    for rec in records:
        ts = _parse_ts(rec.window_start)
        key = (rec.src_ip, rec.dst_ip, _bucket(ts, window_minutes))
        buckets.setdefault(key, []).append(rec)

    reports: list[CampaignReport] = []
    for (src_ip, dst_ip, bucket_start), bucket_recs in sorted(buckets.items()):
        if len(bucket_recs) < min_flows:
            continue

        tech_counter   = Counter(r.stl_technique for r in bucket_recs if r.stl_technique)
        concept_counter = Counter(
            c for r in bucket_recs for c in (r.present_concepts or [])
        )
        feat_counter   = Counter(
            f for r in bucket_recs for f in (r.top_causal_features or [])
        )

        dominant_techniques = _top_k_sorted(tech_counter, top_k_techniques)
        dominant_concepts   = _top_k_sorted(concept_counter, top_k_concepts)
        top_features        = _top_k_sorted(feat_counter, top_k_features)

        scores = [r.anomaly_score for r in bucket_recs]
        flow_ids = sorted({r.flow_id for r in bucket_recs})

        window_end = bucket_start + timedelta(minutes=window_minutes)
        campaign_id = (
            f"{src_ip}--{dst_ip}@{bucket_start.isoformat()}"
        )
        reports.append(CampaignReport(
            campaign_id=campaign_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            window_start=bucket_start.isoformat(),
            window_end=window_end.isoformat(),
            flow_count=len(bucket_recs),
            dominant_stl_techniques=dominant_techniques,
            dominant_concepts=dominant_concepts,
            top_causal_features=top_features,
            max_anomaly_score=max(scores),
            mean_anomaly_score=sum(scores) / len(scores),
            contributing_flow_ids=flow_ids,
            summary=_campaign_summary(
                src_ip, dst_ip, len(bucket_recs),
                dominant_techniques, dominant_concepts, top_features,
            ),
        ))

    reports.sort(key=lambda r: (r.window_start, r.src_ip, r.dst_ip))
    return reports
