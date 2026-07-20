"""Precision / recall per technique and W22 gate evaluation.

W22 gate (binding): STL precision ≥ 0.95 on at least `required_passing`
registered MITRE ATT&CK techniques, evaluated on the test split only.

`required_passing` scales with the technique count so the gate keeps its
"majority of techniques" semantics as the library grows:
    - 4 techniques (frozen pre-v2-repair): 2 must pass.
    - 7 techniques (v2 repair pass, adds T1046/T1041/T1071): 4 must pass.
The four frozen techniques (T1110.001 / T1110.004 / T1498 / T1499) carried
W22 at freeze time; the added techniques may individually fall back to
typed IF-THEN evidence per the plan §7 acceptance criteria.

Evaluation protocol
───────────────────
1. Generate positional windows (default 50 flows each) over the test DataFrame.
2. For each window:
   a. Predict: MITRE ID of the formula with highest positive robustness, or None.
   b. Ground truth: majority attack_family (≥ 50 % of window flows), or None
      (ambiguous window — excluded from TP/FP/FN/TN counts entirely).
3. For each technique T:
   - TP: predicted=T  ∧  gt ∈ T.target_families
   - FP: predicted=T  ∧  gt ∉ T.target_families  (includes gt='Benign')
   - FN: predicted≠T  ∧  gt ∈ T.target_families
   - TN: predicted≠T  ∧  gt ∉ T.target_families
   - precision = TP / (TP + FP);  undefined → 0.0
   - recall    = TP / (TP + FN);  undefined → 0.0
4. Gate passes iff count(precision ≥ threshold) ≥ 2.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import polars as pl

from .classifier import evaluate_all, predict, window_ground_truth
from .formulae import MitreTechnique, ALL_TECHNIQUES


@dataclass
class TechniqueMetrics:
    mitre_id: str
    name: str
    tp: int
    fp: int
    fn: int
    tn: int
    windows_predicted: int
    windows_true: int
    precision: float
    recall: float
    gate_precision_threshold: float
    passes_gate: bool


@dataclass
class W22GateResult:
    passed: bool
    techniques_passing: int
    required_passing: int
    precision_threshold: float
    technique_metrics: list[TechniqueMetrics]
    total_windows: int
    included_windows: int   # those with unambiguous ground truth

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = "PASS" if self.passed else "FAIL"
        return d


def evaluate_test_split(
    df: pl.DataFrame,
    techniques: tuple[MitreTechnique, ...] = ALL_TECHNIQUES,
    window_size: int = 50,
    attack_family_col: str = "attack_family",
    majority_threshold: float = 0.50,
    sort_by_family: bool = True,
) -> W22GateResult:
    """Run the W22 gate evaluation over *df* (expected: raw test split).

    Parameters
    ----------
    df:
        Raw NF-v2 test-split DataFrame (do not pass preprocessed features).
    techniques:
        Ordered tuple of MitreTechnique objects to evaluate.
    window_size:
        Number of flows per positional window.
    attack_family_col:
        Column containing ground-truth attack family labels.
    majority_threshold:
        Fraction of flows that must belong to one family for the window to
        have an unambiguous ground-truth label.
    sort_by_family:
        If True (default), sort *df* by *attack_family_col* before windowing.
        Attack flows in NF-CIC2018-V2 are uniformly interleaved with benign
        traffic, so without sorting every 50-flow window has benign majority
        and no true-positive windows exist.  Sorting groups same-family flows
        into contiguous blocks, enabling majority-vote labelling to produce
        the unambiguous attack windows required for precision measurement.
    """
    if sort_by_family and attack_family_col in df.columns:
        df = df.sort(attack_family_col, maintain_order=True)
    # Counters: {mitre_id → {tp, fp, fn, tn}}
    counts: dict[str, dict[str, int]] = {
        t.mitre_id: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for t in techniques
    }
    target_map = {t.mitre_id: t.target_families for t in techniques}

    total_windows    = 0
    included_windows = 0
    n = len(df)

    for start in range(0, n, window_size):
        window = df[start : start + window_size]
        if len(window) < 3:
            continue
        total_windows += 1

        gt      = window_ground_truth(window, attack_family_col, majority_threshold)
        pred_id = predict(window, techniques)

        if gt is None:
            continue  # ambiguous window: excluded entirely
        included_windows += 1

        for t in techniques:
            mid = t.mitre_id
            gt_positive  = gt in target_map[mid]
            pred_positive = pred_id == mid

            if pred_positive and gt_positive:
                counts[mid]["tp"] += 1
            elif pred_positive and not gt_positive:
                counts[mid]["fp"] += 1
            elif not pred_positive and gt_positive:
                counts[mid]["fn"] += 1
            else:
                counts[mid]["tn"] += 1

    # Build per-technique metrics
    precision_threshold = 0.95
    metrics_list: list[TechniqueMetrics] = []
    for t in techniques:
        c  = counts[t.mitre_id]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        denom_p = tp + fp
        denom_r = tp + fn
        precision = tp / denom_p if denom_p > 0 else 0.0
        recall    = tp / denom_r if denom_r > 0 else 0.0
        metrics_list.append(TechniqueMetrics(
            mitre_id=t.mitre_id,
            name=t.name,
            tp=tp, fp=fp, fn=fn, tn=tn,
            windows_predicted=tp + fp,
            windows_true=tp + fn,
            precision=precision,
            recall=recall,
            gate_precision_threshold=precision_threshold,
            passes_gate=(precision >= precision_threshold and denom_p > 0),
        ))

    n_passing = sum(1 for m in metrics_list if m.passes_gate)
    required_passing = 4 if len(techniques) >= 7 else 2
    return W22GateResult(
        passed=(n_passing >= required_passing),
        techniques_passing=n_passing,
        required_passing=required_passing,
        precision_threshold=precision_threshold,
        technique_metrics=metrics_list,
        total_windows=total_windows,
        included_windows=included_windows,
    )


def format_gate_result(result: W22GateResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"W22 STL gate: {status}",
        f"Techniques with precision >= {result.precision_threshold:.0%}: "
        f"{result.techniques_passing}/{len(result.technique_metrics)} "
        f"(required: {result.required_passing})",
        f"Total windows: {result.total_windows} "
        f"(included: {result.included_windows})",
        "",
        f"{'Technique':<40} {'MITRE':<12} {'Prec':>6} {'Rec':>6} "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'Gate':>5}",
        "─" * 82,
    ]
    for m in result.technique_metrics:
        gate_str = "PASS" if m.passes_gate else "fail"
        lines.append(
            f"{m.name:<40} {m.mitre_id:<12} {m.precision:>6.3f} {m.recall:>6.3f} "
            f"{m.tp:>5} {m.fp:>5} {m.fn:>5} {gate_str:>5}"
        )
    return "\n".join(lines)


def save_gate_result(result: W22GateResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)
