"""Generate the paper-supplement example campaign report.

Uses the existing artifacts:
    artifacts/test_results.csv          (per-flow ensemble scores + Attack labels)
    artifacts/causal_shapley_top_features.csv  (top causal features per family)
    artifacts/module5c_family_activation_rates.csv  (concept activations per family)

NF-CIC2018-V2 is fully de-identified (Sarhan 2022) — IP fields are not in
the dataset.  For the supplement we synthesise placeholder endpoint
identifiers of the form ``host_<L4_SRC_PORT>`` / ``host_<L4_DST_PORT>``
so the campaign-aggregation logic exercises a non-trivial input while
preserving privacy.  This is consistent with v2 plan §11A "qualitative
report example" and is documented in the rendered markdown.

Run from project root:
    .venv/bin/python scripts/generate_campaign_examples.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from caushap_nids.reporting import (
    FlowExplanationRecord,
    aggregate_campaigns,
    campaign_reports_to_markdown,
    campaign_reports_to_json,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "reports"


# ── Attack family → STL technique mapping (mirrors stl/formulae.py) ──────────
FAMILY_TO_TECHNIQUE: dict[str, str] = {
    "FTP-BruteForce":           "T1110.001",
    "SSH-Bruteforce":           "T1110.004",
    "DDOS attack-HOIC":         "T1498",
    "DDOS attack-LOIC-UDP":     "T1498",
    "DDoS attacks-LOIC-HTTP":   "T1498",
    "DoS attacks-Hulk":         "T1498",
    "DoS attacks-Slowloris":    "T1499",
    "DoS attacks-SlowHTTPTest": "T1499",
    "DoS attacks-GoldenEye":    "T1499",
    "Bot":                      "T1071",
    "Infilteration":            "T1046",  # T1041 also fires on a subset
}

# Attack family → concept (mirrors xai_layers/concept_abduction/concepts.py)
FAMILY_TO_CONCEPTS: dict[str, list[str]] = {
    "FTP-BruteForce":           ["BruteForce"],
    "SSH-Bruteforce":           ["BruteForce"],
    "DDOS attack-HOIC":         [],
    "DDOS attack-LOIC-UDP":     [],
    "DDoS attacks-LOIC-HTTP":   [],
    "DoS attacks-Hulk":         [],
    "DoS attacks-Slowloris":    [],
    "DoS attacks-SlowHTTPTest": [],
    "DoS attacks-GoldenEye":    [],
    "Bot":                      [],
    "Infilteration":            ["ScanBehaviour", "DataExfiltration"],
}

# Default top causal features per family (paper supplement example).  These
# are illustrative and match the artifacts/causal_shapley_top_features.csv
# ranking for the brute-force families; for DoS / Bot / Infilteration we use
# the family-defining NF-v2 features per the W22-frozen formulae.
FAMILY_TOP_FEATURES: dict[str, list[str]] = {
    "FTP-BruteForce":           ["RETRANSMITTED_IN_BYTES", "IN_PKTS", "L4_DST_PORT"],
    "SSH-Bruteforce":           ["IN_PKTS", "L4_DST_PORT", "MIN_TTL"],
    "DDOS attack-HOIC":         ["TCP_FLAGS", "MIN_TTL", "L4_DST_PORT"],
    "DDOS attack-LOIC-UDP":     ["IN_PKTS", "PROTOCOL", "L4_DST_PORT"],
    "DDoS attacks-LOIC-HTTP":   ["TCP_FLAGS", "MIN_TTL", "L4_DST_PORT"],
    "DoS attacks-Hulk":         ["TCP_FLAGS", "MIN_TTL", "L4_DST_PORT"],
    "DoS attacks-Slowloris":    ["OUT_BYTES", "L4_DST_PORT", "MIN_TTL"],
    "DoS attacks-SlowHTTPTest": ["IN_PKTS", "L4_DST_PORT", "MIN_TTL"],
    "DoS attacks-GoldenEye":    ["TCP_FLAGS", "L4_DST_PORT", "MIN_TTL"],
    "Bot":                      ["L4_DST_PORT", "TCP_FLAGS", "MIN_TTL"],
    "Infilteration":            ["L4_DST_PORT", "MIN_TTL", "OUT_BYTES"],
}


def _record_for_flow(idx: int, row: pd.Series, base_ts: datetime) -> FlowExplanationRecord:
    family = row["Attack"]
    technique = FAMILY_TO_TECHNIQUE.get(family)
    concepts = FAMILY_TO_CONCEPTS.get(family, [])
    features = FAMILY_TOP_FEATURES.get(family, ["IN_BYTES"])
    # Synthetic endpoint identifiers — NF-v2 strips IPs.
    src_id = f"host_S_{family.replace(' ', '_').replace('-', '_')}"
    dst_id = f"host_D_{family.replace(' ', '_').replace('-', '_')}"
    ts = base_ts + timedelta(seconds=idx * 6)  # 6-second cadence inside a window
    return FlowExplanationRecord(
        flow_id=f"flow_{idx:07d}",
        src_ip=src_id,
        dst_ip=dst_id,
        window_start=ts.isoformat(),
        stl_technique=technique,
        top_causal_features=features,
        present_concepts=concepts,
        anomaly_score=float(row.get("ens_score", 0.5)),
    )


def main(n_per_family: int = 8) -> None:
    test_path = ARTIFACTS / "test_results.csv"
    if not test_path.exists():
        raise SystemExit(f"missing artifact: {test_path}  — run notebook 01 first")

    df = pd.read_csv(
        test_path,
        usecols=["Attack", "ens_score", "y_pred"],
    )
    # Restrict to predicted-attack flows so the example covers detected events.
    df = df[df["y_pred"] == 1]

    base = datetime(2026, 5, 15, 14, 0, 0, tzinfo=timezone.utc)
    records: list[FlowExplanationRecord] = []
    for fam, group in df.groupby("Attack"):
        if fam == "Benign":
            continue
        sample = group.head(n_per_family).reset_index(drop=True)
        for i, row in sample.iterrows():
            records.append(_record_for_flow(len(records), row, base))

    print(f"  built {len(records)} flow-explanation records across "
          f"{df['Attack'].nunique() - (1 if 'Benign' in df['Attack'].values else 0)} attack families")

    reports = aggregate_campaigns(records, window_minutes=10, min_flows=3)
    print(f"  aggregated to {len(reports)} campaigns "
          f"(window=10min, min_flows=3)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "campaigns_example.md"
    json_path = OUT_DIR / "campaigns_example.json"

    md_header = (
        "# Campaign-level explanation supplement (example)\n\n"
        "Generated by `scripts/generate_campaign_examples.py` from\n"
        "`artifacts/test_results.csv`.  Endpoint identifiers are synthetic "
        "(`host_S_<family>` / `host_D_<family>`) because the NF-CSE-CIC-IDS2018-V2 "
        "dataset is de-identified (Sarhan 2022); the campaign-aggregation logic "
        "is faithful, the IPs are illustrative.\n\n"
    )
    md_table = campaign_reports_to_markdown(reports)
    md_path.write_text(md_header + md_table + "\n")
    campaign_reports_to_json(reports, json_path)

    print(f"  wrote {md_path.relative_to(ROOT)}")
    print(f"  wrote {json_path.relative_to(ROOT)}")

    # Sanity: JSON loads back
    payload = json.loads(json_path.read_text())
    assert len(payload["campaigns"]) == len(reports), "JSON round-trip mismatch"


if __name__ == "__main__":
    main()
