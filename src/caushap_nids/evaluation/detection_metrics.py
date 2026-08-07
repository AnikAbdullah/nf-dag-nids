from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

CI = tuple[float, float, float]  # (point_estimate, lower_ci, upper_ci)


def _bootstrap(fn, y_true, y_pred, y_score, n_bootstrap, ci_level, rng,
               max_samples: int | None = None):
    n = len(y_true)
    # Subsample each bootstrap draw when n is large — preserves CI validity
    # while avoiding O(n * n_bootstrap) sklearn calls on multi-million-row arrays.
    draw_size = n if max_samples is None else min(n, max_samples)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=draw_size)
        vals.append(fn(y_true[idx], y_pred[idx], y_score[idx]))
    vals = np.array(vals, dtype=float)
    half = (1.0 - ci_level) / 2.0 * 100
    return float(np.nanpercentile(vals, half)), float(np.nanpercentile(vals, 100 - half))


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    benign_label: int = 0,
    zero_day_mask: np.ndarray | None = None,
    max_bootstrap_samples: int | None = 20_000,
) -> dict:
    """
    Detection metrics with 1000-sample bootstrap CIs.

    Every scalar metric is returned as (point_estimate, lower_ci, upper_ci).
    per_class_f1 is {str(class): (point, lower, upper)}.
    zero_day_recall is included only when zero_day_mask is supplied.
    """
    rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=float)

    def _macro_f1(yt, yp, _ys):
        return f1_score(yt, yp, average="macro", zero_division=0)

    def _precision(yt, yp, _ys):
        return precision_score(yt, yp, average="macro", zero_division=0)

    def _recall(yt, yp, _ys):
        return recall_score(yt, yp, average="macro", zero_division=0)

    def _auc(yt, _yp, ys):
        try:
            mask = ~np.isnan(ys)
            if mask.sum() == 0 or len(np.unique(yt[mask])) < 2:
                return float("nan")
            return float(roc_auc_score(yt[mask], ys[mask]))
        except ValueError:
            return float("nan")

    def _fpr(yt, yp, _ys):
        mask = yt == benign_label
        if mask.sum() == 0:
            return float("nan")
        return float((yp[mask] != benign_label).mean())

    mbs = max_bootstrap_samples
    results: dict = {}
    for key, fn in [
        ("macro_f1", _macro_f1),
        ("precision", _precision),
        ("recall", _recall),
        ("auc_roc", _auc),
        ("fpr", _fpr),
    ]:
        point = fn(y_true, y_pred, y_score)
        lo, hi = _bootstrap(fn, y_true, y_pred, y_score, n_bootstrap, ci_level, rng, mbs)
        results[key] = (round(float(point), 4), round(lo, 4), round(hi, 4))

    classes = np.unique(y_true)
    per_class: dict[str, CI] = {}
    for c in classes:
        def _cls_f1(yt, yp, _ys, _c=c):
            if _c not in np.unique(yt):
                return float("nan")
            return float(f1_score(yt, yp, labels=[_c], average="macro", zero_division=0))

        point = _cls_f1(y_true, y_pred, y_score)
        lo, hi = _bootstrap(_cls_f1, y_true, y_pred, y_score, n_bootstrap, ci_level, rng, mbs)
        per_class[str(c)] = (round(float(point), 4), round(lo, 4), round(hi, 4))
    results["per_class_f1"] = per_class

    if zero_day_mask is not None:
        zd = np.asarray(zero_day_mask, dtype=bool)

        def _zd_recall(yt, yp, _ys):
            if zd.sum() == 0:
                return float("nan")
            yt_zd, yp_zd = yt[zd], yp[zd]
            atk = yt_zd != benign_label
            if atk.sum() == 0:
                return float("nan")
            return float((yp_zd[atk] != benign_label).mean())

        point = _zd_recall(y_true, y_pred, y_score)
        lo, hi = _bootstrap(_zd_recall, y_true, y_pred, y_score, n_bootstrap, ci_level, rng, mbs)
        results["zero_day_recall"] = (round(float(point), 4), round(lo, 4), round(hi, 4))

    return results
