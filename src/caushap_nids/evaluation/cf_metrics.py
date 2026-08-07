from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from caushap_nids.xai_layers.multi_obj_cf.nsga import CounterfactualExplanation


def aggregate_cf_metrics(cfs: list[CounterfactualExplanation]) -> dict[str, float]:
    """Mean validity, proximity, sparsity, feasibility_rate over a CF list."""
    if not cfs:
        return {
            "validity": 0.0,
            "proximity": float("nan"),
            "sparsity": float("nan"),
            "feasibility_rate": 0.0,
        }
    return {
        "validity": float(np.mean([cf.validity for cf in cfs])),
        "proximity": float(np.mean([cf.proximity for cf in cfs])),
        "sparsity": float(np.mean([cf.sparsity for cf in cfs])),
        "feasibility_rate": float(np.mean([cf.feasibility_rate for cf in cfs])),
    }


def cf_hypervolume(cfs: list[CounterfactualExplanation]) -> float:
    """Pareto hypervolume of the CF set. Delegates to multi_obj_cf.pareto."""
    from caushap_nids.xai_layers.multi_obj_cf.pareto import hypervolume
    return hypervolume(cfs)


def compare_cf_sets(
    method_a: list[CounterfactualExplanation],
    method_b: list[CounterfactualExplanation],
) -> dict[str, dict]:
    """
    Compare two CF methods on all metrics.
    Returns {metric: {a, b, delta, a_better}} for each metric.
    Higher is better for: validity, feasibility_rate, hypervolume.
    Lower is better for: proximity, sparsity.
    """
    ma = aggregate_cf_metrics(method_a)
    mb = aggregate_cf_metrics(method_b)
    ma["hypervolume"] = cf_hypervolume(method_a)
    mb["hypervolume"] = cf_hypervolume(method_b)

    higher_better = {"validity", "feasibility_rate", "hypervolume"}
    result: dict[str, dict] = {}
    for metric in ma:
        val_a, val_b = ma[metric], mb[metric]
        delta = val_a - val_b
        a_better = (delta > 0) if metric in higher_better else (delta < 0)
        result[metric] = {"a": val_a, "b": val_b, "delta": delta, "a_better": a_better}
    return result
