import numpy as np


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Select threshold maximising Macro-F1 subject to FPR <= target_fpr.

    Ties broken by higher attack TPR, then higher attack precision.
    Falls back to unconstrained best Macro-F1 if target_fpr cannot be achieved.
    Returns scalar threshold value.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if not 0.0 <= float(target_fpr) <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    if len(scores) == 0:
        return 0.0

    order    = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order]

    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float(s_sorted[0]) if len(s_sorted) else 0.0

    eps = 1e-12
    # Evaluate only at score-block ends.  Returning a raw score threshold means
    # predict(score >= threshold), so all rows tied at that score are included.
    # Computing FPR inside a tie block can accidentally violate target_fpr.
    block_ends = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    thresholds = s_sorted[block_ends]

    tp = np.cumsum(y_sorted == 1).astype(np.float64)[block_ends]
    fp = np.cumsum(y_sorted == 0).astype(np.float64)[block_ends]
    fn = n_pos - tp
    tn = n_neg - fp

    tpr  = tp / n_pos
    fpr  = fp / n_neg
    prec_a = tp / np.maximum(tp + fp, eps)
    f1_a   = 2 * prec_a * tpr / np.maximum(prec_a + tpr, eps)
    prec_b = tn / np.maximum(tn + fn, eps)
    rec_b  = tn / n_neg
    f1_b   = 2 * prec_b * rec_b / np.maximum(prec_b + rec_b, eps)
    macro_f1 = 0.5 * (f1_a + f1_b)

    # Include the all-benign operating point.  It is rarely chosen, but it makes
    # the FPR constraint always satisfiable for extremely strict targets.
    threshold_none = np.nextafter(float(s_sorted[0]), np.inf)
    prec_b0 = n_neg / max(n_pos + n_neg, eps)
    f1_b0 = 2.0 * prec_b0 / max(prec_b0 + 1.0, eps)
    thresholds = np.r_[threshold_none, thresholds]
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    prec_a = np.r_[0.0, prec_a]
    macro_f1 = np.r_[0.5 * f1_b0, macro_f1]

    mask = fpr <= target_fpr
    if mask.any():
        pool      = np.where(mask)[0]
        composite = macro_f1[pool] + 1e-6 * tpr[pool] + 1e-9 * prec_a[pool]
        best      = pool[np.argmax(composite)]
    else:
        best = int(np.argmax(macro_f1))

    return float(thresholds[best])
