"""Human-readable narrative generation for XAI outputs.

Produces three-part analyst reports (Detection / Reasoning / Suggested Action)
from :class:`CampaignReport` and :class:`FlowExplanationRecord` objects.

Design
──────
Template-based NLG only — fully deterministic and auditable.  No external API
is called; every sentence is generated from domain mappings and the structured
XAI outputs already in the dataclasses.

This approach is intentional for reproducibility: an LLM API (e.g. Gemini)
would make the narrative non-deterministic and introduce a third-party
dependency that reviewers cannot audit or reproduce.  If an LLM-enriched
variant is ever wanted for an interactive demo, it must use a local model
(e.g. Ollama) and must never sit in the paper-experiment path.

Three-part report structure
───────────────────────────
  [Detection]        What was flagged and how confident the model is.
  [Reasoning]        Why — top causal features, fired concepts, STL match.
  [Suggested Action] Analyst next-step keyed to the MITRE technique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .schema import CampaignReport, FlowExplanationRecord


# ── Domain knowledge maps ──────────────────────────────────────────────────────

_MITRE_DESCRIPTIONS: dict[str, str] = {
    "T1046":     "Network Service Discovery (port scan)",
    "T1110":     "Credential Brute Force",
    "T1110.001": "FTP Credential Brute Force",
    "T1110.004": "SSH Credential Brute Force",
    "T1498":     "Network Denial-of-Service (volumetric flood)",
    "T1499":     "Endpoint Denial-of-Service (application-layer slow attack)",
    "T1041":     "Exfiltration Over C2 Channel",
    "T1071":     "Application Layer Protocol — Command & Control",
}

_MITRE_ACTIONS: dict[str, str] = {
    "T1046": (
        "Isolate the scanning host and review firewall rules; "
        "check whether any subsequently contacted ports are exposed services."
    ),
    "T1110": (
        "Block the source IP at the perimeter and force credential rotation "
        "on the targeted service; audit recent successful logins for lateral movement."
    ),
    "T1110.001": (
        "Disable or restrict FTP on the target host; rotate FTP credentials "
        "and audit recent successful FTP sessions."
    ),
    "T1110.004": (
        "Rate-limit or block SSH from the source subnet; "
        "rotate SSH keys and review ~/.authorized_keys on the target."
    ),
    "T1498": (
        "Engage upstream ISP for traffic scrubbing; "
        "activate DDoS mitigation rules and monitor service availability."
    ),
    "T1499": (
        "Identify and drop slow-session connections from the attacker range; "
        "review application-layer timeouts and connection-limit settings."
    ),
    "T1041": (
        "Capture full-packet data from the flagged flows for forensic review; "
        "check whether the destination IP is a known C2 indicator."
    ),
    "T1071": (
        "Inspect the application-layer payload for encoded commands; "
        "correlate the destination domain/IP against threat-intelligence feeds."
    ),
}

_FEATURE_EXPLANATIONS: dict[str, str] = {
    "L4_DST_PORT":                "destination port",
    "L4_SRC_PORT":                "source port",
    "FLOW_DURATION_MILLISECONDS": "flow duration (ms)",
    "IN_PKTS":                    "inbound packet count",
    "OUT_PKTS":                   "outbound packet count",
    "IN_BYTES":                   "inbound byte volume",
    "OUT_BYTES":                  "outbound byte volume",
    "MIN_TTL":                    "minimum TTL (OS fingerprint)",
    "MAX_TTL":                    "maximum TTL",
    "TCP_FLAGS":                  "TCP flag pattern (tool signature)",
    "FTP_COMMAND_RET_CODE":       "FTP return code",
    "PROTOCOL":                   "layer-4 protocol",
    "RETRANSMITTED_IN_BYTES":     "retransmitted inbound bytes",
    "RETRANSMITTED_OUT_BYTES":    "retransmitted outbound bytes",
    "SRC_TO_DST_AVG_THROUGHPUT":  "src→dst average throughput",
    "DST_TO_SRC_AVG_THROUGHPUT":  "dst→src average throughput",
    "NUM_PKTS_UP_TO_128_BYTES":   "small packet count (≤128 B)",
    "LONGEST_FLOW_PKT":           "longest packet in flow",
    "SHORTEST_FLOW_PKT":          "shortest packet in flow",
    "MIN_IP_PKT_LEN":             "minimum IP packet length",
    "MAX_IP_PKT_LEN":             "maximum IP packet length",
    "DURATION_IN":                "inbound flow duration",
    "DURATION_OUT":               "outbound flow duration",
    "SRC_TO_DST_SECOND_BYTES":    "src→dst bytes per second",
    "DST_TO_SRC_SECOND_BYTES":    "dst→src bytes per second",
}

_CONCEPT_EXPLANATIONS: dict[str, str] = {
    "ScanBehaviour":    "port-scanning behaviour",
    "BruteForce":       "credential brute-force pattern",
    "DataExfiltration": "data exfiltration pattern",
}

_SCORE_BANDS: list[tuple[float, str]] = [
    (0.9,  "very high confidence"),
    (0.75, "high confidence"),
    (0.6,  "moderate confidence"),
    (0.0,  "low confidence"),
]


def _score_label(score: float) -> str:
    for threshold, label in _SCORE_BANDS:
        if score >= threshold:
            return label
    return "low confidence"


def _feature_phrase(features: Sequence[str]) -> str:
    explained = [_FEATURE_EXPLANATIONS.get(f, f) for f in features]
    if not explained:
        return "no dominant features identified"
    if len(explained) == 1:
        return explained[0]
    return ", ".join(explained[:-1]) + f" and {explained[-1]}"


def _concept_phrase(concepts: Sequence[str]) -> str:
    explained = [_CONCEPT_EXPLANATIONS.get(c, c) for c in concepts]
    if not explained:
        return "no behavioural concept activated"
    return " + ".join(explained)


def _technique_phrase(techniques: Sequence[str]) -> str:
    if not techniques:
        return "no MITRE technique matched"
    parts = [f"{t} ({_MITRE_DESCRIPTIONS.get(t, 'unknown')})" for t in techniques]
    return "; ".join(parts)


def _action_for(techniques: Sequence[str]) -> str:
    for t in techniques:
        if t in _MITRE_ACTIONS:
            return _MITRE_ACTIONS[t]
    if techniques:
        parent = techniques[0].split(".")[0]
        if parent in _MITRE_ACTIONS:
            return _MITRE_ACTIONS[parent]
    return (
        "Investigate the flagged flows manually; "
        "correlate src/dst IPs against threat-intelligence feeds."
    )


# ── Public dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NarrativeReport:
    """Three-part human-readable explanation for a campaign or flow.

    Fields
    ──────
    detection        One-sentence verdict with confidence label.
    reasoning        Why the model raised the alert (features + concepts + STL).
    suggested_action Analyst next-step keyed to the dominant MITRE technique.
    plain_text       Concatenated paragraphs, ready for display or export.
    """

    detection: str
    reasoning: str
    suggested_action: str

    @property
    def plain_text(self) -> str:
        return (
            f"[Detection]\n{self.detection}\n\n"
            f"[Reasoning]\n{self.reasoning}\n\n"
            f"[Suggested Action]\n{self.suggested_action}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "detection": self.detection,
            "reasoning": self.reasoning,
            "suggested_action": self.suggested_action,
        }


# ── Narration entry points ─────────────────────────────────────────────────────

def narrate_campaign(report: CampaignReport) -> NarrativeReport:
    """Generate a three-part analyst narrative from a :class:`CampaignReport`.

    Deterministic — identical inputs always produce identical output.
    """
    confidence = _score_label(report.max_anomaly_score)
    tech_str   = _technique_phrase(report.dominant_stl_techniques)
    concept_str = _concept_phrase(report.dominant_concepts)
    feat_str   = _feature_phrase(report.top_causal_features)

    detection = (
        f"The model detected a {confidence} anomaly from "
        f"{report.src_ip} → {report.dst_ip} "
        f"over {report.flow_count} flows in the window "
        f"{report.window_start} – {report.window_end} "
        f"(peak anomaly score: {report.max_anomaly_score:.4f})."
    )

    reasoning = (
        f"The alert was triggered by {tech_str}. "
        f"The causal Shapley layer identified the most influential network "
        f"signals as: {feat_str}. "
        f"The concept abduction layer activated: {concept_str}."
    )

    action = _action_for(report.dominant_stl_techniques)
    suggested_action = (
        f"Based on the detected technique ({tech_str}), "
        f"the recommended analyst action is: {action}"
    )

    return NarrativeReport(
        detection=detection,
        reasoning=reasoning,
        suggested_action=suggested_action,
    )


def narrate_flow(record: FlowExplanationRecord) -> NarrativeReport:
    """Generate a three-part analyst narrative from a single :class:`FlowExplanationRecord`."""
    confidence = _score_label(record.anomaly_score)
    tech_str   = (
        f"{record.stl_technique} ({_MITRE_DESCRIPTIONS.get(record.stl_technique or '', 'unknown')})"
        if record.stl_technique
        else "no MITRE technique matched"
    )
    concept_str = _concept_phrase(record.present_concepts)
    feat_str    = _feature_phrase(record.top_causal_features)

    detection = (
        f"Flow {record.flow_id} ({record.src_ip} → {record.dst_ip}) "
        f"flagged with {confidence} (anomaly score: {record.anomaly_score:.4f})."
    )

    reasoning = (
        f"The STL classifier matched: {tech_str}. "
        f"Causal Shapley identified the dominant features as: {feat_str}. "
        f"Concept abduction activated: {concept_str}."
    )

    action = _action_for([record.stl_technique] if record.stl_technique else [])
    suggested_action = (
        f"Based on the detected technique ({tech_str}), "
        f"the recommended analyst action is: {action}"
    )

    return NarrativeReport(
        detection=detection,
        reasoning=reasoning,
        suggested_action=suggested_action,
    )


def narrate_batch(
    reports: list[CampaignReport],
) -> list[tuple[CampaignReport, NarrativeReport]]:
    """Narrate every campaign in a list; returns paired (report, narrative) tuples."""
    return [(r, narrate_campaign(r)) for r in reports]
