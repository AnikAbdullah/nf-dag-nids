from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV

from .nsga import CounterfactualExplanation

DEFAULT_REFERENCE_POINT = np.array([1.1, 20.0, 45.0, 1.1], dtype=np.float64)


def _obj_vector(cf: CounterfactualExplanation) -> np.ndarray:
    """Objective vector in minimisation form: [1-validity, proximity, sparsity, 1-feasibility_rate]."""
    return np.array([
        1.0 - cf.validity,
        cf.proximity,
        float(cf.sparsity),
        1.0 - cf.feasibility_rate,
    ])


def hypervolume(
    pareto_front: list[CounterfactualExplanation],
    reference_point: np.ndarray | None = None,
) -> float:
    """
    Hypervolume indicator of the Pareto front (all 4 objectives in minimisation form).
    Higher = better (front dominates more space relative to reference point).
    """
    if not pareto_front:
        return 0.0

    F = np.array([_obj_vector(cf) for cf in pareto_front])

    if reference_point is None:
        # Use a fixed, generous reference point so hypervolumes are comparable
        # across methods/runs. The old self-referenced HV rewarded methods that
        # merely produced a wider or worse front.
        reference_point = np.maximum(DEFAULT_REFERENCE_POINT, F.max(axis=0) * 1.05 + 1e-9)

    ind = HV(ref_point=reference_point)
    return float(ind(F))


def dominates(a: CounterfactualExplanation, b: CounterfactualExplanation) -> bool:
    """True if a Pareto-dominates b: a is at least as good on all 4 objectives and strictly better on one."""
    obj_a = _obj_vector(a)
    obj_b = _obj_vector(b)
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))
