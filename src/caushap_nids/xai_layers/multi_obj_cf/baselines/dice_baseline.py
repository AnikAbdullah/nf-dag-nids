from __future__ import annotations

import warnings

import numpy as np
import networkx as nx
import pandas as pd

try:
    import dice_ml
    _DICE_AVAILABLE = True
except ImportError:
    _DICE_AVAILABLE = False

from ..nsga import CounterfactualExplanation
from ..objectives import feasibility as _feasibility
from ..objectives import proximity, sparsity
from ..objectives import validity as _validity


# Temperature applied to the AE-score logit so the sigmoid is not flat near
# the operating threshold. Gives DiCE's genetic optimiser usable gradient.
_PROBA_TEMPERATURE = 5.0


class _SklearnDetectorWrapper:
    """Sklearn-compatible wrapper so DiCE can call predict / predict_proba.

    `predict_proba` rescales the AE-score logit by a temperature so the
    sigmoid is not flat near the operating threshold (which is typically
    O(1e-2) on this dataset).
    """

    def __init__(self, detector, threshold: float):
        self.detector = detector
        self.threshold = float(threshold)

    def predict(self, X) -> np.ndarray:
        scores = self.detector.score(np.asarray(X, dtype=np.float64))
        return (scores >= self.threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        scores = self.detector.score(np.asarray(X, dtype=np.float64))
        scale = max(abs(self.threshold), 1e-6)
        logit = (scores - self.threshold) / scale * _PROBA_TEMPERATURE
        p_attack = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - p_attack, p_attack])


def generate_scalarized_dice_cfs(
    detector,
    background: np.ndarray,
    x_orig: np.ndarray,
    feature_names: list[str],
    dag: nx.DiGraph,
    threshold: float,
    *,
    total_cfs: int = 10,
    proximity_weight: float = 0.5,
    seed: int = 42,
    dice_method: str | None = None,
) -> list[CounterfactualExplanation]:
    """
    Scalarized DiCE baseline (Action 5, P3B plan).

    DiCE's scalarized optimiser minimises a weighted sum of
    validity + proximity + sparsity (single scalar objective), in contrast
    to NSGA-II which optimises all four objectives simultaneously on the
    Pareto front.

    Tries ``method='genetic'`` first; on failure or empty result, retries
    with ``method='random'`` (Mothilal 2020 random-search DiCE — the
    established stable baseline). Set ``dice_method`` to force one method.
    """
    methods = [dice_method] if dice_method else ["genetic", "random"]
    last_warning: str | None = None

    for method in methods:
        cfs = generate_dice_cfs(
            detector,
            background,
            x_orig,
            feature_names,
            dag,
            threshold,
            total_cfs=total_cfs,
            method=method,
            seed=seed,
            allow_fallback=False,
        )
        if cfs:
            return cfs
        last_warning = f"DiCE method={method!r} returned 0 CFs"

    # Both genetic and random failed — use the deterministic safety-net.
    if last_warning:
        warnings.warn(
            f"{last_warning}; using deterministic DiCE-style fallback. "
            "This means the baseline is not real DiCE — investigate.",
            stacklevel=2,
        )
    return _fallback_dice_cfs(
        detector, background, np.asarray(x_orig, dtype=np.float64),
        feature_names, dag, threshold,
        total_cfs=total_cfs, seed=seed,
    )


def generate_dice_cfs(
    detector,
    background: np.ndarray,
    x_orig: np.ndarray,
    feature_names: list[str],
    dag: nx.DiGraph,
    threshold: float,
    *,
    total_cfs: int = 10,
    method: str = "random",
    seed: int = 42,
    allow_fallback: bool = True,
) -> list[CounterfactualExplanation]:
    """
    Vanilla DiCE counterfactuals (no DAG constraints).
    Serves as the comparison baseline for the Layer B NSGA-II approach.

    When ``allow_fallback`` is True (the default) and dice-ml fails or is
    unavailable, returns a deterministic DiCE-style fallback. When False,
    returns ``[]`` so the caller can decide what to do (used by
    ``generate_scalarized_dice_cfs`` to chain methods before falling back).
    """
    x_orig = np.asarray(x_orig, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if not _DICE_AVAILABLE:
        warnings.warn(
            "dice-ml is not installed — using deterministic DiCE-style fallback.",
            stacklevel=2,
        )
        if not allow_fallback:
            return []
        return _fallback_dice_cfs(
            detector, background, x_orig, feature_names, dag, threshold,
            total_cfs=total_cfs, seed=seed,
        )

    bg_df = pd.DataFrame(background, columns=feature_names)
    scores = detector.score(background)
    bg_df["label"] = (scores >= threshold).astype(int)

    # DiCE requires both classes in the background data.
    if bg_df["label"].nunique() < 2:
        extra = pd.DataFrame([x_orig], columns=feature_names)
        extra["label"] = 1
        bg_df = pd.concat([bg_df, extra], ignore_index=True)

    data = dice_ml.Data(
        dataframe=bg_df,
        continuous_features=feature_names,
        outcome_name="label",
    )
    model_wrapper = _SklearnDetectorWrapper(detector, threshold)
    model = dice_ml.Model(model=model_wrapper, backend="sklearn")
    exp = dice_ml.Dice(data, model, method=method)

    query = pd.DataFrame([x_orig], columns=feature_names)

    # Only DiceRandom accepts random_seed; DiceGenetic raises TypeError on it.
    gen_kwargs: dict = {"total_CFs": total_cfs, "desired_class": "opposite"}
    if method == "random":
        gen_kwargs["random_seed"] = seed
    else:
        np.random.seed(seed)   # genetic uses numpy's global RNG for reproducibility

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dice_result = exp.generate_counterfactuals(query, **gen_kwargs)
        cfs_df = dice_result.cf_examples_list[0].final_cfs_df
        if cfs_df is None or len(cfs_df) == 0:
            warnings.warn(
                f"DiCE method={method!r} produced 0 CFs (empty cfs_df).",
                stacklevel=2,
            )
            if not allow_fallback:
                return []
            return _fallback_dice_cfs(
                detector, background, x_orig, feature_names, dag, threshold,
                total_cfs=total_cfs, seed=seed,
            )
    except Exception as exc:
        warnings.warn(
            f"DiCE method={method!r} raised {type(exc).__name__}: {exc!s}. "
            f"{'Falling back to deterministic baseline.' if allow_fallback else 'Returning empty list.'}",
            stacklevel=2,
        )
        if not allow_fallback:
            return []
        return _fallback_dice_cfs(
            detector, background, x_orig, feature_names, dag, threshold,
            total_cfs=total_cfs, seed=seed,
        )

    results: list[CounterfactualExplanation] = []
    for _, row in cfs_df.iterrows():
        x_cf = row[feature_names].to_numpy(dtype=np.float64)
        v = _validity(x_cf, detector, threshold)
        p = proximity(x_orig, x_cf)
        s = int(sparsity(x_orig, x_cf))
        feas = 1.0 - _feasibility(x_orig, x_cf, dag, feature_names)
        changed = [feature_names[i] for i in range(len(x_orig)) if abs(x_cf[i] - x_orig[i]) > 1e-6]
        results.append(CounterfactualExplanation(
            x_orig=x_orig.copy(),
            x_cf=x_cf,
            validity=v,
            proximity=p,
            sparsity=s,
            feasibility_rate=feas,
            changed_features=changed,
        ))

    return results


def _fallback_dice_cfs(
    detector,
    background: np.ndarray,
    x_orig: np.ndarray,
    feature_names: list[str],
    dag: nx.DiGraph,
    threshold: float,
    *,
    total_cfs: int,
    seed: int,
) -> list[CounterfactualExplanation]:
    """Deterministic DiCE-style fallback when dice-ml is unavailable.

    It preserves the vanilla baseline's key property: validity/proximity/sparsity
    are optimized without using DAG feasibility as a generation constraint.
    Candidates are built by moving the anomalous flow toward benign-looking
    background prototypes, then ranked by DiCE's scalar objectives.
    """
    rng = np.random.default_rng(seed)
    scores = detector.score(background)
    benign_pool = background[scores < threshold]
    if len(benign_pool) == 0:
        benign_pool = background
    if len(benign_pool) == 0:
        return []

    sample_n = min(len(benign_pool), max(50, total_cfs * 20))
    sample_idx = rng.choice(len(benign_pool), size=sample_n, replace=False)
    prototypes = benign_pool[sample_idx]

    candidates: list[CounterfactualExplanation] = []
    for proto in prototypes:
        deltas = np.abs(proto - x_orig)
        ranked = np.argsort(-deltas)
        for k in range(1, min(len(feature_names), 12) + 1):
            x_cf = x_orig.copy()
            changed_idx = ranked[:k]
            x_cf[changed_idx] = proto[changed_idx]
            v = _validity(x_cf, detector, threshold)
            if v < 1.0:
                continue
            p = proximity(x_orig, x_cf)
            s = int(sparsity(x_orig, x_cf))
            feas = 1.0 - _feasibility(x_orig, x_cf, dag, feature_names)
            changed = [feature_names[i] for i in range(len(x_orig)) if abs(x_cf[i] - x_orig[i]) > 1e-6]
            candidates.append(CounterfactualExplanation(
                x_orig=x_orig.copy(),
                x_cf=x_cf,
                validity=v,
                proximity=p,
                sparsity=s,
                feasibility_rate=feas,
                changed_features=changed,
            ))
            break

    # Vanilla DiCE is deliberately DAG-agnostic: rank by the scalar DiCE-style
    # objectives only, then evaluate DAG feasibility afterwards.
    candidates.sort(key=lambda cf: (-cf.validity, cf.proximity, cf.sparsity))
    return candidates[:total_cfs]
