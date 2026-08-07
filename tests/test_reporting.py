"""Tests for Module 6b: reporting/.

Covers:
  - Schema dataclasses (FlowExplanationRecord, CampaignReport) serialise cleanly
  - aggregate_campaigns groups by (src_ip, dst_ip, time-bucket) deterministically
  - Campaigns smaller than min_flows are dropped
  - Dominant techniques / concepts / features are top-k sorted (desc count, alpha)
  - JSON round-trip preserves all fields
  - Markdown output is well-formed and stable
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from caushap_nids.reporting import (
    FlowExplanationRecord,
    CampaignReport,
    aggregate_campaigns,
    campaign_reports_to_markdown,
    campaign_reports_to_json,
    NarrativeReport,
    narrate_campaign,
    narrate_flow,
    narrate_batch,
)


def _rec(
    flow_id: str,
    src: str,
    dst: str,
    ts: str,
    technique: str | None = "T1110.001",
    concepts: list[str] | None = None,
    features: list[str] | None = None,
    score: float = 0.5,
) -> FlowExplanationRecord:
    return FlowExplanationRecord(
        flow_id=flow_id,
        src_ip=src,
        dst_ip=dst,
        window_start=ts,
        stl_technique=technique,
        top_causal_features=features or ["IN_BYTES", "TCP_FLAGS"],
        present_concepts=concepts or ["BruteForce"],
        anomaly_score=score,
    )


class TestSchema:
    def test_flow_record_to_dict_round_trip(self):
        r = _rec("f1", "10.0.0.1", "10.0.0.2", "2026-05-15T14:00:00+00:00")
        d = r.to_dict()
        assert d["flow_id"] == "f1"
        assert d["src_ip"] == "10.0.0.1"
        assert d["stl_technique"] == "T1110.001"

    def test_flow_record_allows_null_technique(self):
        r = _rec("f1", "a", "b", "2026-05-15T14:00:00+00:00", technique=None)
        assert r.to_dict()["stl_technique"] is None

    def test_campaign_report_to_dict(self):
        c = CampaignReport(
            campaign_id="c1",
            src_ip="10.0.0.1", dst_ip="10.0.0.2",
            window_start="2026-05-15T14:00:00",
            window_end="2026-05-15T14:10:00",
            flow_count=5,
            dominant_stl_techniques=["T1110.001"],
            dominant_concepts=["BruteForce"],
            top_causal_features=["IN_BYTES"],
            summary="...",
        )
        d = c.to_dict()
        assert d["flow_count"] == 5


class TestAggregate:
    def test_single_campaign_grouping(self):
        recs = [
            _rec(f"f{i}", "10.0.0.1", "10.0.0.2",
                 "2026-05-15T14:00:00+00:00", score=0.5 + 0.01 * i)
            for i in range(5)
        ]
        reports = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        assert len(reports) == 1
        r = reports[0]
        assert r.src_ip == "10.0.0.1"
        assert r.dst_ip == "10.0.0.2"
        assert r.flow_count == 5
        assert r.dominant_stl_techniques == ["T1110.001"]
        assert r.dominant_concepts == ["BruteForce"]
        # max ≈ 0.5 + 0.01*4 = 0.54
        assert r.max_anomaly_score == pytest.approx(0.54)

    def test_below_min_flows_dropped(self):
        recs = [
            _rec(f"f{i}", "a", "b", "2026-05-15T14:00:00+00:00")
            for i in range(2)
        ]
        reports = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        assert reports == []

    def test_separate_pairs_become_separate_campaigns(self):
        recs = (
            [_rec(f"a{i}", "10.0.0.1", "10.0.0.2",
                  "2026-05-15T14:00:00+00:00") for i in range(3)]
            + [_rec(f"b{i}", "10.0.0.3", "10.0.0.4",
                    "2026-05-15T14:00:00+00:00") for i in range(3)]
        )
        reports = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        assert len(reports) == 2
        pairs = {(r.src_ip, r.dst_ip) for r in reports}
        assert pairs == {("10.0.0.1", "10.0.0.2"), ("10.0.0.3", "10.0.0.4")}

    def test_separate_time_buckets_become_separate_campaigns(self):
        recs = (
            [_rec(f"e{i}", "a", "b",
                  "2026-05-15T14:02:00+00:00") for i in range(3)]
            + [_rec(f"l{i}", "a", "b",
                    "2026-05-15T14:31:00+00:00") for i in range(3)]
        )
        reports = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        assert len(reports) == 2
        assert reports[0].window_start < reports[1].window_start

    def test_dominant_techniques_ranked_by_count(self):
        recs = (
            [_rec(f"a{i}", "x", "y", "2026-05-15T14:00:00+00:00",
                  technique="T1110.001") for i in range(5)]
            + [_rec(f"b{i}", "x", "y", "2026-05-15T14:00:00+00:00",
                    technique="T1498") for i in range(2)]
        )
        reports = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        assert reports[0].dominant_stl_techniques == ["T1110.001", "T1498"]

    def test_none_technique_excluded_from_dominant(self):
        recs = (
            [_rec(f"a{i}", "x", "y", "2026-05-15T14:00:00+00:00",
                  technique=None) for i in range(2)]
            + [_rec(f"b{i}", "x", "y", "2026-05-15T14:00:00+00:00",
                    technique="T1071") for i in range(3)]
        )
        reports = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        # Both None and T1071 contribute flows, but only T1071 surfaces as dominant.
        assert reports[0].dominant_stl_techniques == ["T1071"]

    def test_deterministic_output_order(self):
        # Same input in different order → same output.
        recs_a = [_rec(f"f{i}", "10.0.0.1", "10.0.0.2",
                       "2026-05-15T14:00:00+00:00") for i in range(5)]
        recs_b = list(reversed(recs_a))
        reports_a = aggregate_campaigns(recs_a, window_minutes=10, min_flows=3)
        reports_b = aggregate_campaigns(recs_b, window_minutes=10, min_flows=3)
        assert reports_a[0].to_dict() == reports_b[0].to_dict()

    def test_unparseable_timestamp_does_not_crash(self):
        recs = [
            _rec(f"f{i}", "a", "b", "not-a-real-timestamp")
            for i in range(3)
        ]
        reports = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        # Still groups by (src, dst, fallback-epoch); one campaign emerges.
        assert len(reports) == 1


class TestRender:
    def _reports(self) -> list[CampaignReport]:
        recs = (
            [_rec(f"a{i}", "10.0.0.1", "10.0.0.2",
                  "2026-05-15T14:00:00+00:00") for i in range(5)]
            + [_rec(f"b{i}", "10.0.0.3", "10.0.0.4",
                    "2026-05-15T14:00:00+00:00",
                    technique="T1071", concepts=["DataExfiltration"]) for i in range(3)]
        )
        return aggregate_campaigns(recs, window_minutes=10, min_flows=3)

    def test_markdown_has_header(self):
        md = campaign_reports_to_markdown(self._reports())
        assert "| Campaign |" in md
        assert "Top causal features" in md

    def test_markdown_has_one_row_per_campaign(self):
        md = campaign_reports_to_markdown(self._reports())
        assert md.count("\n") >= 3  # header + separator + 2 data rows

    def test_markdown_empty_input_renders_header_only(self):
        md = campaign_reports_to_markdown([])
        assert md.count("\n") == 1  # just the header

    def test_markdown_escapes_pipe_in_summary(self):
        c = CampaignReport(
            campaign_id="c", src_ip="x", dst_ip="y",
            window_start="2026-05-15T14:00:00",
            window_end="2026-05-15T14:10:00",
            flow_count=3,
            dominant_stl_techniques=["T1071"],
            dominant_concepts=[],
            top_causal_features=[],
            summary="contains | pipe character",
        )
        md = campaign_reports_to_markdown([c])
        assert "contains \\| pipe character" in md

    def test_json_round_trip(self, tmp_path: Path):
        out = tmp_path / "campaigns.json"
        campaign_reports_to_json(self._reports(), out)
        payload = json.loads(out.read_text())
        assert "campaigns" in payload
        assert len(payload["campaigns"]) == 2
        first = payload["campaigns"][0]
        for key in (
            "campaign_id", "src_ip", "dst_ip", "window_start", "window_end",
            "flow_count", "dominant_stl_techniques", "dominant_concepts",
            "top_causal_features", "summary",
            "max_anomaly_score", "mean_anomaly_score", "contributing_flow_ids",
        ):
            assert key in first, f"missing key: {key}"

    def test_json_is_deterministic(self, tmp_path: Path):
        out1 = tmp_path / "a.json"
        out2 = tmp_path / "b.json"
        reports = self._reports()
        campaign_reports_to_json(reports, out1)
        campaign_reports_to_json(reports, out2)
        assert out1.read_bytes() == out2.read_bytes()


class TestNarrate:
    def _campaign(self, techniques=None, concepts=None, score=0.92) -> CampaignReport:
        if techniques is None:
            techniques = ["T1110.001"]
        if concepts is None:
            concepts = ["BruteForce"]
        return CampaignReport(
            campaign_id="c1",
            src_ip="10.0.0.1",
            dst_ip="192.168.1.5",
            window_start="2026-05-15T14:00:00",
            window_end="2026-05-15T14:10:00",
            flow_count=7,
            dominant_stl_techniques=techniques,
            dominant_concepts=concepts,
            top_causal_features=["IN_BYTES", "TCP_FLAGS", "MIN_TTL"],
            summary="...",
            max_anomaly_score=score,
            mean_anomaly_score=score - 0.05,
        )

    def _flow(self) -> FlowExplanationRecord:
        return FlowExplanationRecord(
            flow_id="f42",
            src_ip="10.0.0.1",
            dst_ip="192.168.1.5",
            window_start="2026-05-15T14:00:00+00:00",
            stl_technique="T1046",
            top_causal_features=["L4_DST_PORT", "FLOW_DURATION_MILLISECONDS"],
            present_concepts=["ScanBehaviour"],
            anomaly_score=0.81,
        )

    # ── NarrativeReport structure ────────────────────────────────────────────

    def test_plain_text_has_three_sections(self):
        n = narrate_campaign(self._campaign())
        assert "[Detection]" in n.plain_text
        assert "[Reasoning]" in n.plain_text
        assert "[Suggested Action]" in n.plain_text

    def test_to_dict_has_required_keys(self):
        n = narrate_campaign(self._campaign())
        d = n.to_dict()
        assert set(d) == {"detection", "reasoning", "suggested_action"}
        for v in d.values():
            assert isinstance(v, str) and len(v) > 0

    # ── narrate_campaign ────────────────────────────────────────────────────

    def test_detection_contains_ips_and_score(self):
        n = narrate_campaign(self._campaign(score=0.92))
        assert "10.0.0.1" in n.detection
        assert "192.168.1.5" in n.detection
        assert "0.9200" in n.detection

    def test_detection_very_high_confidence_label(self):
        n = narrate_campaign(self._campaign(score=0.95))
        assert "very high confidence" in n.detection

    def test_detection_moderate_confidence_label(self):
        n = narrate_campaign(self._campaign(score=0.65))
        assert "moderate confidence" in n.detection

    def test_reasoning_contains_mitre_id(self):
        n = narrate_campaign(self._campaign(techniques=["T1110.001"]))
        assert "T1110.001" in n.reasoning

    def test_reasoning_human_feature_names(self):
        n = narrate_campaign(self._campaign())
        # IN_BYTES → "inbound byte volume"
        assert "inbound byte volume" in n.reasoning
        # TCP_FLAGS → "TCP flag pattern (tool signature)"
        assert "TCP flag pattern" in n.reasoning

    def test_reasoning_concept_phrase(self):
        n = narrate_campaign(self._campaign(concepts=["BruteForce"]))
        assert "credential brute-force pattern" in n.reasoning

    def test_reasoning_no_concept(self):
        n = narrate_campaign(self._campaign(concepts=[]))
        assert "no behavioural concept activated" in n.reasoning

    def test_suggested_action_ssh_technique(self):
        n = narrate_campaign(self._campaign(techniques=["T1110.004"]))
        assert "SSH" in n.suggested_action

    def test_suggested_action_dos_technique(self):
        n = narrate_campaign(self._campaign(techniques=["T1498"]))
        assert "scrubbing" in n.suggested_action or "DDoS" in n.suggested_action

    def test_suggested_action_unknown_technique_fallback(self):
        n = narrate_campaign(self._campaign(techniques=[]))
        assert "Investigate" in n.suggested_action

    def test_parent_technique_fallback_for_action(self):
        # T1110.001 → FTP action; parent T1110 also has an entry
        n = narrate_campaign(self._campaign(techniques=["T1110.001"]))
        assert "FTP" in n.suggested_action or "credential" in n.suggested_action.lower()

    def test_deterministic(self):
        c = self._campaign()
        assert narrate_campaign(c).plain_text == narrate_campaign(c).plain_text

    # ── narrate_flow ────────────────────────────────────────────────────────

    def test_flow_detection_has_flow_id(self):
        n = narrate_flow(self._flow())
        assert "f42" in n.detection

    def test_flow_reasoning_scan_behaviour(self):
        n = narrate_flow(self._flow())
        assert "port-scanning behaviour" in n.reasoning
        assert "destination port" in n.reasoning

    def test_flow_no_technique(self):
        rec = FlowExplanationRecord(
            flow_id="f0",
            src_ip="a", dst_ip="b",
            window_start="2026-05-15T14:00:00+00:00",
            stl_technique=None,
            top_causal_features=["OUT_BYTES"],
            present_concepts=[],
            anomaly_score=0.4,
        )
        n = narrate_flow(rec)
        assert "no MITRE technique matched" in n.reasoning

    # ── narrate_batch ───────────────────────────────────────────────────────

    def test_batch_length(self):
        recs = [_rec(f"f{i}", "a", "b", "2026-05-15T14:00:00+00:00") for i in range(5)]
        campaigns = aggregate_campaigns(recs, window_minutes=10, min_flows=3)
        pairs = narrate_batch(campaigns)
        assert len(pairs) == len(campaigns)
        for report, narrative in pairs:
            assert isinstance(report, CampaignReport)
            assert isinstance(narrative, NarrativeReport)
