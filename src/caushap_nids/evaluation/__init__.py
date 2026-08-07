from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
from caushap_nids.evaluation.faithfulness import (
    sufficiency,
    comprehensiveness,
    lipschitz_stability,
)
from caushap_nids.evaluation.cf_metrics import (
    aggregate_cf_metrics,
    cf_hypervolume,
    compare_cf_sets,
)
from caushap_nids.evaluation.concept_metrics import (
    batch_concept_fidelity,
    batch_concept_rank_stability,
)
from caushap_nids.evaluation.statistical import wilcoxon_bonferroni, cohens_d
from caushap_nids.evaluation.table_writer import write_latex_table, write_all_paper_tables
from caushap_nids.evaluation.quantus_adapter import eraser_sufficiency, eraser_comprehensiveness
from caushap_nids.evaluation.criteria import (
    CriteriaCheck,
    evaluate_ablation_criteria,
    criteria_to_rows,
    all_criteria_pass,
)

__all__ = [
    "compute_detection_metrics",
    "sufficiency",
    "comprehensiveness",
    "lipschitz_stability",
    "aggregate_cf_metrics",
    "cf_hypervolume",
    "compare_cf_sets",
    "batch_concept_fidelity",
    "batch_concept_rank_stability",
    "wilcoxon_bonferroni",
    "cohens_d",
    "write_latex_table",
    "write_all_paper_tables",
    "eraser_sufficiency",
    "eraser_comprehensiveness",
    "CriteriaCheck",
    "evaluate_ablation_criteria",
    "criteria_to_rows",
    "all_criteria_pass",
]
