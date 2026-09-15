from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from .feasibility import dag_respecting_mutation
from .objectives import feasibility as _co_change_violation
from .objectives import proximity, sparsity
from .objectives import validity as _validity
from .structural import (
    StructuralEquations,
    structural_violation,
    structural_violation_batch,
)


def _feasibility(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    dag: nx.DiGraph,
    feature_names: list[str],
    sem: StructuralEquations | None = None,
) -> float:
    """Feasibility violation, minimised. Structural when a fitted SEM is supplied.

    Without a SEM this falls back to the co-change criterion, which is retained
    only to reproduce previously published runs: it is maximised both by
    changing nothing and by changing every feature, so a search optimising it
    drives sparsity up to the feature count. Pass ``sem`` for new work.
    """
    if sem is not None:
        return structural_violation(x_orig, x_cf, sem)
    return _co_change_violation(x_orig, x_cf, dag, feature_names)


@dataclass
class CounterfactualExplanation:
    x_orig: np.ndarray
    x_cf: np.ndarray
    validity: float
    proximity: float
    sparsity: int
    feasibility_rate: float     # fraction of DAG edges satisfied (1.0 = fully feasible)
    changed_features: list[str] = field(default_factory=list)


class _DagMutation(Mutation):
    """pymoo Mutation operator that propagates parent changes through the DAG."""

    def __init__(self, dag: nx.DiGraph, feature_names: list[str], prob: float = 0.1):
        super().__init__()
        self._dag = dag
        self._feature_names = feature_names
        self._prob = prob

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        rng = np.random.default_rng()
        for i in range(len(X)):
            X[i] = dag_respecting_mutation(
                X[i], self._dag, self._feature_names,
                mutation_rate=self._prob, rng=rng,
            )
        return X


def _soft_validity_obj(x_cf: np.ndarray, detector, threshold: float) -> float:
    """Continuous validity objective for NSGA-II optimisation.
    Returns 0 when benign (score < threshold), otherwise the normalised excess
    (score - threshold) / threshold so the optimiser gets gradient signal rather
    than a flat binary cliff.  The output CounterfactualExplanation.validity is
    always recomputed as the binary value via _validity().
    """
    score = float(detector.score(np.asarray(x_cf, dtype=np.float64).reshape(1, -1))[0])
    if score < threshold:
        return 0.0
    return (score - threshold) / max(threshold, 1e-9)


class _CFProblem(Problem):
    """4-objective NSGA-II problem: minimise [soft_validity, proximity, sparsity, feas_violations].

    Evaluated over the whole population at once.  Element-wise evaluation issued
    one detector call per individual -- pop_size * n_gen single-row calls per
    anchor -- and single-row scoring is where an Isolation Forest spends almost
    all of its time.  Batching leaves the objectives identical.
    """

    def __init__(self, x_orig, detector, dag, feature_names, threshold, xl, xu, sem=None):
        super().__init__(n_var=len(x_orig), n_obj=4, xl=xl, xu=xu)
        self.x_orig = x_orig
        self.detector = detector
        self.dag = dag
        self.feature_names = feature_names
        self.threshold = threshold
        self.sem = sem

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        diff = np.abs(X - self.x_orig)

        scores = np.asarray(self.detector.score(X), dtype=np.float64).ravel()
        v_soft = np.where(
            scores < self.threshold,
            0.0,
            (scores - self.threshold) / max(self.threshold, 1e-9),
        )
        p = diff.sum(axis=1)
        s = (diff > 1e-6).sum(axis=1).astype(np.float64)

        if self.sem is not None:
            f = structural_violation_batch(self.x_orig, X, self.sem)
        else:
            f = np.array([
                _co_change_violation(self.x_orig, row, self.dag, self.feature_names)
                for row in X
            ])

        out["F"] = np.column_stack([v_soft, p, s, f])


class _WarmStartSampling(Sampling):
    """
    Initialise warm_frac of the population from actual background rows (clipped to bounds).
    Gives NSGA-II a head-start inside the valid (benign) region of feature space so it
    can find valid CFs without having to navigate there from scratch.
    """

    def __init__(self, background: np.ndarray, xl: np.ndarray, xu: np.ndarray, warm_frac: float = 0.3):
        super().__init__()
        self._background = np.clip(background, xl, xu)
        self._xl = xl
        self._xu = xu
        self._warm_frac = warm_frac

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng()
        n_warm = min(int(n_samples * self._warm_frac), len(self._background))
        idx = rng.choice(len(self._background), size=n_warm, replace=False)
        warm = self._background[idx].copy()
        # Tiny noise so duplicates are eliminated cleanly.
        warm += rng.normal(scale=1e-4, size=warm.shape)
        warm = np.clip(warm, self._xl, self._xu)
        n_uniform = n_samples - n_warm
        uniform = rng.uniform(self._xl, self._xu, size=(n_uniform, problem.n_var))
        return np.vstack([warm, uniform])


def _select_representative(
    cfs: list[CounterfactualExplanation], n: int
) -> list[CounterfactualExplanation]:
    """Pick n knee-point counterfactuals: closest to the ideal corner of the front.

    Each objective is min-max normalised across the front before distances are
    taken, so no single objective dominates the choice.  The earlier lexicographic
    sort ranked feasibility ahead of sparsity, which — combined with the co-change
    criterion's preference for changing everything — meant the reported
    counterfactual was systematically the least actionable point on the front.
    """
    if len(cfs) <= n:
        return cfs

    # Objectives, all "lower is better".
    obj = np.array(
        [
            [1.0 - cf.validity, cf.proximity, float(cf.sparsity), 1.0 - cf.feasibility_rate]
            for cf in cfs
        ],
        dtype=np.float64,
    )
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    dist = np.linalg.norm((obj - lo) / span, axis=1)
    return [cfs[i] for i in np.argsort(dist)[:n]]


def _binary_dominates(a: CounterfactualExplanation, b: CounterfactualExplanation) -> bool:
    """Pareto dominance using binary validity (the externally-visible form)."""
    obj_a = np.array([1.0 - a.validity, a.proximity, float(a.sparsity), 1.0 - a.feasibility_rate])
    obj_b = np.array([1.0 - b.validity, b.proximity, float(b.sparsity), 1.0 - b.feasibility_rate])
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def _prune_dominated(cfs: list[CounterfactualExplanation]) -> list[CounterfactualExplanation]:
    """Remove solutions that are dominated in binary-validity space after soft-validity optimisation."""
    non_dom = [
        a for i, a in enumerate(cfs)
        if not any(_binary_dominates(b, a) for j, b in enumerate(cfs) if j != i)
    ]
    return non_dom if non_dom else cfs   # fallback: never return empty


def _as_cf_result(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    detector,
    dag: nx.DiGraph,
    feature_names: list[str],
    threshold: float,
    sem: StructuralEquations | None = None,
) -> CounterfactualExplanation:
    v = _validity(x_cf, detector, threshold)
    p = proximity(x_orig, x_cf)
    s = int(round(sparsity(x_orig, x_cf)))
    feas = 1.0 - _feasibility(x_orig, x_cf, dag, feature_names, sem)
    changed = [feature_names[i] for i in range(len(x_orig)) if abs(x_cf[i] - x_orig[i]) > 1e-6]
    return CounterfactualExplanation(
        x_orig=x_orig.copy(),
        x_cf=x_cf.copy(),
        validity=v,
        proximity=p,
        sparsity=s,
        feasibility_rate=feas,
        changed_features=changed,
    )


def _sparsify_valid_cf(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    detector,
    dag: nx.DiGraph,
    feature_names: list[str],
    threshold: float,
    sem: StructuralEquations | None = None,
) -> np.ndarray:
    """Greedily remove unnecessary changes from a valid CF without hurting feasibility."""
    current = np.asarray(x_cf, dtype=np.float64).copy()
    x_orig = np.asarray(x_orig, dtype=np.float64)
    if _validity(current, detector, threshold) < 1.0:
        return current

    max_violation = _feasibility(x_orig, current, dag, feature_names, sem)

    def _acceptable(candidate: np.ndarray) -> bool:
        return (
            _validity(candidate, detector, threshold) == 1.0
            and _feasibility(x_orig, candidate, dag, feature_names, sem) <= max_violation + 1e-12
        )

    changed = np.flatnonzero(np.abs(current - x_orig) > 1e-6)
    for idx in sorted(changed, key=lambda i: abs(current[i] - x_orig[i])):
        candidate = current.copy()
        candidate[idx] = x_orig[idx]
        if _acceptable(candidate):
            current = candidate
            max_violation = _feasibility(x_orig, current, dag, feature_names, sem)

    changed = np.flatnonzero(np.abs(current - x_orig) > 1e-6)
    for idx in changed:
        endpoint = float(current[idx])
        lo, hi = 0.0, 1.0
        best = current.copy()
        for _ in range(10):
            mid = (lo + hi) / 2.0
            candidate = current.copy()
            candidate[idx] = float(x_orig[idx]) + mid * (endpoint - float(x_orig[idx]))
            if _acceptable(candidate):
                best = candidate
                hi = mid
            else:
                lo = mid
        current = best

    return current


def generate_cf_pareto_front(
    detector,
    dag: nx.DiGraph,
    x: np.ndarray,
    feature_names: list[str],
    target_class: int = 0,
    *,
    population_size: int = 100,
    n_generations: int = 50,
    seed: int = 42,
    bounds: tuple[float, float] | tuple[np.ndarray, np.ndarray] = (-10.0, 10.0),
    threshold: float | None = None,
    background: np.ndarray | None = None,
    warm_frac: float = 0.3,
    return_valid_only: bool = False,
    n_cfs: int | None = None,
    sem: StructuralEquations | None = None,
    selection: str = "knee",
) -> list[CounterfactualExplanation]:
    """
    Main entry point for Layer B.
    Returns the Pareto-optimal counterfactuals trading off validity, proximity,
    sparsity, and DAG feasibility.
    threshold is mandatory — pass the detector's operating decision threshold.
    background: if provided, warm_frac of the initial population is seeded from
        actual benign rows so NSGA-II starts inside the valid region.
    sem: structural equations fitted on benign traffic (see ``structural.py``).
        When supplied, feasibility is the causal-recourse criterion; when None it
        falls back to the legacy co-change criterion, which is kept only for
        reproducing previously published runs.
    selection: how to reduce the front to ``n_cfs`` points. "knee" (default)
        takes the points nearest the ideal corner after normalising each
        objective; "feasibility_first" reproduces the legacy lexicographic sort.
    """
    if threshold is None:
        raise ValueError(
            "threshold must be provided. Use the detector's operating threshold "
            "(the score value above which a flow is classified as attack)."
        )

    x = np.asarray(x, dtype=np.float64)
    d = len(x)

    if isinstance(bounds[0], (int, float)):
        xl = np.full(d, float(bounds[0]))
        xu = np.full(d, float(bounds[1]))
    else:
        xl = np.asarray(bounds[0], dtype=np.float64)
        xu = np.asarray(bounds[1], dtype=np.float64)

    if background is not None:
        sampler = _WarmStartSampling(np.asarray(background, dtype=np.float64), xl, xu, warm_frac)
    else:
        sampler = FloatRandomSampling()

    problem = _CFProblem(x, detector, dag, feature_names, threshold, xl, xu, sem)
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=sampler,
        crossover=SBX(prob=0.9, eta=15),
        mutation=_DagMutation(dag, feature_names, prob=0.1),
        eliminate_duplicates=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in power",
            category=RuntimeWarning,
            module=r"pymoo\.operators\.crossover\.sbx",
        )
        res = minimize(problem, algorithm, ("n_gen", n_generations), seed=seed, verbose=False)

    if res.X is None:
        return []

    X_sols = np.atleast_2d(res.X)
    results: list[CounterfactualExplanation] = []

    for x_cf in X_sols:
        x_cf = _sparsify_valid_cf(x, x_cf, detector, dag, feature_names, threshold, sem)
        results.append(_as_cf_result(x, x_cf, detector, dag, feature_names, threshold, sem))

    if return_valid_only:
        valid = [cf for cf in results if cf.validity == 1.0]
        if valid:
            results = valid

    results = _prune_dominated(results)
    if n_cfs is not None and n_cfs > 0:
        if selection == "feasibility_first":
            results = sorted(
                results,
                key=lambda cf: (-cf.validity, -cf.feasibility_rate, cf.sparsity, cf.proximity),
            )[:n_cfs]
        else:
            valid = [cf for cf in results if cf.validity == 1.0] or results
            results = _select_representative(valid, n_cfs)
    return results
