"""Module 5a: causal Shapley explanations for anomaly scores."""

from .cc_shapley import cc_shapley_correction
from .interventional import interventional_expectation
from .kernel import CausalShapleyExplanation, causal_shapley, vanilla_kernel_shap
from .mediation import decompose_effects

causal_shapley_values = causal_shapley

__all__ = [
    "CausalShapleyExplanation",
    "causal_shapley",
    "causal_shapley_values",
    "vanilla_kernel_shap",
    "cc_shapley_correction",
    "decompose_effects",
    "interventional_expectation",
]
