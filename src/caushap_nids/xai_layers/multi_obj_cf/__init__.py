# Module 5b — Multi-Objective Causally-Constrained Counterfactuals (NSGA-II)
# Phase P3B. Owner: Abdullah Al Taieb.
# Depends on: Module 2 (detector), Module 3 (dag)
# Gate W17: Pareto hypervolume dominates scalarized DiCE on >= 3/4 datasets.

from .nsga import CounterfactualExplanation, generate_cf_pareto_front
from .pareto import dominates, hypervolume

__all__ = [
    "CounterfactualExplanation",
    "dominates",
    "generate_cf_pareto_front",
    "hypervolume",
]
