# Module 6b — Campaign-level reporting (Phase P3C/P4)
# Owner: Md. Abdullah
# Depends on: Module 4 (STL technique IDs), Module 5a (causal SHAP top features),
#             Module 5c (concept abduction outputs).

from .schema import FlowExplanationRecord, CampaignReport
from .campaign import aggregate_campaigns
from .render import (
    campaign_reports_to_markdown,
    campaign_reports_to_json,
)
from .narrate import NarrativeReport, narrate_campaign, narrate_flow, narrate_batch

__all__ = [
    "FlowExplanationRecord",
    "CampaignReport",
    "aggregate_campaigns",
    "campaign_reports_to_markdown",
    "campaign_reports_to_json",
    "NarrativeReport",
    "narrate_campaign",
    "narrate_flow",
    "narrate_batch",
]
