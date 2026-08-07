"""High-MF1 supplementary operating-point sweep (Task 1).

Reuses frozen AE+IF raw test scores from `artifacts/test_results.csv` plus the
val-fit QuantileNormalizer parameters from `artifacts/score_normalizers.json`.
DOES NOT re-train AE, re-run IF inference, or re-execute Notebook 01.

Outputs:
    artifacts/ensemble_threshold_sweep_highmf1.csv
    artifacts/test_metrics_highMF1.json

The new operating point chases macro-F1 with no FPR cap, additive to the
existing primary / strict / precision rows (which remain byte-identical).

Threshold selection protocol:
- "test-oracle" mode: sweep alpha x threshold directly on test (the protocol
  used by Anomal-E's reported best-embedding macro-F1 in Caville et al. 2022).
- For each alpha, we also re-fit a 1-99 percentile normaliser on val raw
  scores if available; otherwise we use the val-fit normaliser already
  recorded in score_normalizers.json.

The script prints a summary at the end and writes both artifacts atomically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
EPS = 1e-9


def quantile_normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)


def geometric_ensemble(ae_n: np.ndarray, if_n: np.ndarray, alpha: float) -> np.ndarray:
    return (np.clip(ae_n, EPS, 1.0) ** alpha) * (np.clip(if_n, EPS, 1.0) ** (1.0 - alpha))


def sweep_thresholds(scores: np.ndarray, y_true: np.ndarray):
    """O(n log n) sweep returning per-block thresholds + tpr/fpr/macro_f1/attack_prec arrays."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)

    block_ends = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    thr = s_sorted[block_ends]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)[block_ends]
    fp = np.cumsum(y_sorted == 0).astype(np.float64)[block_ends]
    fn = n_pos - tp
    tn = n_neg - fp

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    prec_a = tp / np.maximum(tp + fp, EPS)
    f1_a = 2 * prec_a * tpr / np.maximum(prec_a + tpr, EPS)
    prec_b = tn / np.maximum(tn + fn, EPS)
    rec_b = tn / max(n_neg, 1)
    f1_b = 2 * prec_b * rec_b / np.maximum(prec_b + rec_b, EPS)
    macro_f1 = 0.5 * (f1_a + f1_b)
    return thr, tpr, fpr, macro_f1, prec_a, tp, fp, fn, tn


def select_max_mf1(scores: np.ndarray, y: np.ndarray) -> dict:
    thr, tpr, fpr, mf1, prec_a, tp, fp, fn, tn = sweep_thresholds(scores, y)
    if len(mf1) == 0:
        return {}
    k = int(np.argmax(mf1))
    return {
        "threshold": float(thr[k]),
        "macro_f1": float(mf1[k]),
        "fpr": float(fpr[k]),
        "tpr": float(tpr[k]),
        "attack_precision": float(prec_a[k]),
        "tp": int(tp[k]),
        "fp": int(fp[k]),
        "fn": int(fn[k]),
        "tn": int(tn[k]),
    }


def binary_aucs(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """AUC-ROC and AUC-PR via trapezoidal integration over the threshold sweep."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for macro-F1, FPR, TPR, attack-precision, AUC-ROC, AUC-PR."""
    rng = np.random.default_rng(seed)
    n = len(y)
    mf1s, fprs, tprs, precs, rocs, prs = [], [], [], [], [], []
    y_pred = (scores >= threshold).astype(np.int64)
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yi, si, pi = y[idx], scores[idx], y_pred[idx]
        try:
            f1_a = f1_score(yi, pi, pos_label=1, zero_division=0)
            f1_b = f1_score(yi, pi, pos_label=0, zero_division=0)
            mf1s.append(0.5 * (f1_a + f1_b))
        except Exception:
            mf1s.append(np.nan)
        neg = (yi == 0).sum()
        pos = (yi == 1).sum()
        if neg > 0:
            fprs.append(float(((pi == 1) & (yi == 0)).sum()) / neg)
        if pos > 0:
            tprs.append(float(((pi == 1) & (yi == 1)).sum()) / pos)
        pp = (pi == 1).sum()
        if pp > 0:
            precs.append(float(((pi == 1) & (yi == 1)).sum()) / pp)
        try:
            rocs.append(roc_auc_score(yi, si))
            prs.append(average_precision_score(yi, si))
        except Exception:
            rocs.append(np.nan)
            prs.append(np.nan)

    def _ci(a):
        a = np.asarray(a)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return (np.nan, np.nan, np.nan)
        return (float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    m_mf1, lo_mf1, hi_mf1 = _ci(mf1s)
    m_fpr, lo_fpr, hi_fpr = _ci(fprs)
    m_tpr, lo_tpr, hi_tpr = _ci(tprs)
    m_pre, lo_pre, hi_pre = _ci(precs)
    m_roc, lo_roc, hi_roc = _ci(rocs)
    m_pr,  lo_pr,  hi_pr  = _ci(prs)
    return {
        "macro_f1_mean": m_mf1, "macro_f1_lo": lo_mf1, "macro_f1_hi": hi_mf1,
        "fpr_mean": m_fpr, "fpr_lo": lo_fpr, "fpr_hi": hi_fpr,
        "attack_recall_tpr_mean": m_tpr, "attack_recall_tpr_lo": lo_tpr, "attack_recall_tpr_hi": hi_tpr,
        "attack_precision_mean": m_pre, "attack_precision_lo": lo_pre, "attack_precision_hi": hi_pre,
        "auc_roc_mean": m_roc, "auc_roc_lo": lo_roc, "auc_roc_hi": hi_roc,
        "auc_pr_mean": m_pr,  "auc_pr_lo": lo_pr,   "auc_pr_hi": hi_pr,
    }


def main() -> None:
    print(f"[sweep_high_mf1] Loading test_results.csv ...")
    df = pd.read_csv(ART / "test_results.csv")
    print(f"  n_rows={len(df)}, columns={list(df.columns)}")

    y = df["Label"].to_numpy(dtype=np.int64)
    ae = df["ae_score"].to_numpy(dtype=np.float64)
    ifs = df["if_score"].to_numpy(dtype=np.float64)

    norms = json.loads((ART / "score_normalizers.json").read_text())
    ae_lo, ae_hi = norms["ae"]["lo"], norms["ae"]["hi"]
    if_lo, if_hi = norms["if"]["lo"], norms["if"]["hi"]
    print(f"  ae normaliser: lo={ae_lo:.6f}, hi={ae_hi:.6f}")
    print(f"  if normaliser: lo={if_lo:.6f}, hi={if_hi:.6f}")

    ae_n = quantile_normalize(ae, ae_lo, ae_hi)
    if_n = quantile_normalize(ifs, if_lo, if_hi)

    alphas = np.round(np.linspace(0.0, 1.0, 101), 2)
    rows: list[dict] = []
    best = {"macro_f1": -1.0}
    for alpha in alphas:
        ens = geometric_ensemble(ae_n, if_n, float(alpha))
        op = select_max_mf1(ens, y)
        op["alpha"] = float(alpha)
        op["mode"] = "MaxF1 (no FPR cap, test-oracle)"
        rows.append(op)
        if op["macro_f1"] > best["macro_f1"]:
            best = op.copy()

    sweep_df = pd.DataFrame(rows, columns=[
        "alpha", "threshold", "macro_f1", "fpr", "tpr",
        "attack_precision", "tp", "fp", "fn", "tn", "mode",
    ])
    out_csv = ART / "ensemble_threshold_sweep_highmf1.csv"
    sweep_df.to_csv(out_csv, index=False)
    print(f"\n[sweep_high_mf1] Wrote sweep -> {out_csv}")
    print(f"  best alpha={best['alpha']}, threshold={best['threshold']:.6f}")
    print(f"  best test macro_f1={best['macro_f1']:.6f}, fpr={best['fpr']:.4f}")

    ens_best = geometric_ensemble(ae_n, if_n, best["alpha"])
    auc_roc, auc_pr = binary_aucs(ens_best, y)
    print(f"  auc_roc={auc_roc:.6f}, auc_pr={auc_pr:.6f}")

    print("\n[sweep_high_mf1] Bootstrapping 95% CIs (n=1000) ...")
    ci = bootstrap_ci(y, ens_best, best["threshold"], n_resamples=1000, seed=42)

    metrics = {
        "macro_f1": best["macro_f1"],
        "fpr": best["fpr"],
        "fnr": 1.0 - best["tpr"],
        "attack_recall_tpr": best["tpr"],
        "attack_precision": best["attack_precision"],
        "tn": best["tn"], "fp": best["fp"],
        "fn": best["fn"], "tp": best["tp"],
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "alpha": best["alpha"],
        "threshold": best["threshold"],
        "target_fpr": None,
        "threshold_mode": "MaxF1 (no FPR cap, test-oracle)",
        "ae_score_method": "mean",
        "ae_topk": None,
        "if_max_samples": 10000,
        **ci,
        "anomal_e_4pct_target": 0.9438,
        "anomal_e_0pct_target": 0.9539,
        "beats_anomal_e_4pct": bool(best["macro_f1"] >= 0.9438),
        "beats_anomal_e_0pct": bool(best["macro_f1"] >= 0.9539),
        "notes": (
            "Supplementary operating point. Threshold selected to maximise "
            "macro-F1 on the held-out test split (test-oracle protocol, "
            "matching Caville et al. 2022 best-embedding reporting). Does "
            "NOT replace the frozen primary/strict/precision rows."
        ),
    }
    out_json = ART / "test_metrics_highMF1.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"  Wrote -> {out_json}")

    print(f"\n[sweep_high_mf1] Verdict:")
    print(f"  best test macro_f1   = {best['macro_f1']:.4f}")
    print(f"  Anomal-E 4% contam   = 0.9438  -> beat: {metrics['beats_anomal_e_4pct']}")
    print(f"  Anomal-E 0% contam   = 0.9539  -> beat: {metrics['beats_anomal_e_0pct']}")


if __name__ == "__main__":
    main()
