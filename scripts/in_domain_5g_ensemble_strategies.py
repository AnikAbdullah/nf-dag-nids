"""Try multiple ensemble strategies for 5G-NIDD to beat the per-seed mean (0.5137).

The default FPR<=0.01 threshold target is wrong for 5G-NIDD (60% attacks in val).
Try:
  (a) MaxF1 threshold on val (no FPR cap)
  (b) Geometric-mean of normalised per-seed scores (current default)
  (c) Rank-based ensemble (Borda-style; robust to score-distribution shift)
  (d) Per-seed predict + majority vote
  (e) Per-seed predict at MaxF1 + majority vote

Also report per-seed-best-AUC alpha sweep for reference.

Reference points:
  Per-seed in-domain matrix mean: 0.5137
  Per-seed best (seed44):          0.5597
  XeNIDS DA cross-domain (full):   0.6839
  XeNIDS DA non-volumetric subset: 0.9563
  Bayes-optimal ceiling (full):    0.7572 (unreachable by unsupervised on full)
  W20 bar:                          0.7837 (mathematically impossible)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from caushap_nids.experiments.runner import (
    _load_dataset, _train_or_load_detector, load_run_config, _EnsembleScoreAdapter,
)
from caushap_nids.experiments import DEFAULT_SEEDS

ART = ROOT / "artifacts"
OUT_JSON = ART / "in_domain_5g_ensemble_strategies.json"
DATASET = "5g_nidd"


def _macro_f1(y, pred):
    f1_a = f1_score(y, pred, pos_label=1, zero_division=0)
    f1_b = f1_score(y, pred, pos_label=0, zero_division=0)
    return 0.5 * (f1_a + f1_b)


def _metrics_at(scores, y, threshold):
    pred = (scores >= threshold).astype(np.int64)
    fpr = float(((pred == 1) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
    tpr = float(((pred == 1) & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
    prec = float(((pred == 1) & (y == 1)).sum()) / max(int((pred == 1).sum()), 1)
    return dict(
        macro_f1=float(_macro_f1(y, pred)),
        fpr=fpr, tpr=tpr, attack_precision=prec,
        auc_roc=float(roc_auc_score(y, scores)),
        threshold=float(threshold),
    )


def _calibrate_fpr(scores, y, target_fpr=0.01):
    order = np.argsort(-scores, kind="mergesort")
    sord = scores[order]; yord = y[order]
    n_neg = int((yord == 0).sum())
    if n_neg == 0:
        return float(sord[0])
    cum_fp = np.cumsum(yord == 0)
    fpr = cum_fp / n_neg
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float(sord[0]) + 1e-9
    return float(sord[int(valid[-1])])


def _calibrate_maxf1(scores, y):
    """Pick threshold that maximises macro-F1 on the calibration set."""
    order = np.argsort(-scores, kind="mergesort")
    sord = scores[order]; yord = y[order]
    n_pos = int(yord.sum()); n_neg = int(len(yord) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float(sord[0])
    # Distinct-threshold sweep
    block_ends = np.flatnonzero(np.r_[sord[1:] != sord[:-1], True])
    thr = sord[block_ends]
    tp = np.cumsum(yord == 1).astype(np.float64)[block_ends]
    fp = np.cumsum(yord == 0).astype(np.float64)[block_ends]
    fn = n_pos - tp; tn = n_neg - fp
    tpr = tp / n_pos
    pre_a = tp / np.maximum(tp + fp, 1e-9)
    f1_a = 2 * pre_a * tpr / np.maximum(pre_a + tpr, 1e-9)
    pre_b = tn / np.maximum(tn + fn, 1e-9)
    rec_b = tn / n_neg
    f1_b = 2 * pre_b * rec_b / np.maximum(pre_b + rec_b, 1e-9)
    mf1 = 0.5 * (f1_a + f1_b)
    return float(thr[int(mf1.argmax())])


def _rank_normalize(scores):
    """Convert scores to ranks in [0, 1] (higher = more anomalous)."""
    ranks = np.argsort(np.argsort(scores)).astype(np.float64)
    return ranks / max(len(ranks) - 1, 1)


def main():
    print(f"\n=== 5G-NIDD ensemble strategies (per-seed mean baseline: 0.5137) ===", flush=True)
    per_seed = []
    y_val_ref = None; y_test_ref = None
    cfg = load_run_config("A4_full", ROOT / "configs")

    for seed in DEFAULT_SEEDS:
        t0 = time.time()
        X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(DATASET, seed, ROOT / "data")
        if y_val_ref is None:
            y_val_ref, y_test_ref = y_val, y_test
        (ae, if_det), (ae_norm, if_norm, alpha, ae_feat_idx), _thr, _diag = _train_or_load_detector(
            cfg, X_train, y_train, X_val, y_val, ART, "A4_full", DATASET, seed
        )
        adapter = _EnsembleScoreAdapter(ae=ae, if_det=if_det, ae_norm=ae_norm, if_norm=if_norm,
                                         alpha=alpha, ae_feature_indices=ae_feat_idx)
        val = adapter.score(X_val)
        test = adapter.score(X_test)
        per_seed.append({'seed': seed, 'val': val, 'test': test})
        print(f"  seed {seed} scored ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n  benign val frac: {(y_val_ref == 0).mean():.4f}; "
          f"attack val frac: {(y_val_ref == 1).mean():.4f}")

    # ── Baseline: per-seed at FPR<=0.01 (matches matrix) ──────────────────────
    print("\n[A] Per-seed @ FPR<=0.01 (matrix protocol):", flush=True)
    base_f1 = []
    for s in per_seed:
        thr = _calibrate_fpr(s['val'], y_val_ref, 0.01)
        m = _metrics_at(s['test'], y_test_ref, thr)
        base_f1.append(m['macro_f1'])
        print(f"    seed {s['seed']}: f1={m['macro_f1']:.4f} AUC={m['auc_roc']:.4f} thr={thr:.4f}",
              flush=True)
    base_mean = float(np.mean(base_f1))
    print(f"    PER-SEED MEAN: {base_mean:.4f}")

    # ── Per-seed at MaxF1 on val (no FPR cap) ─────────────────────────────────
    print("\n[B] Per-seed @ MaxF1-on-val:", flush=True)
    maxf1_per_seed = []
    for s in per_seed:
        thr = _calibrate_maxf1(s['val'], y_val_ref)
        m = _metrics_at(s['test'], y_test_ref, thr)
        maxf1_per_seed.append(m)
        print(f"    seed {s['seed']}: f1={m['macro_f1']:.4f} AUC={m['auc_roc']:.4f} "
              f"FPR={m['fpr']:.4f} thr={thr:.4f}", flush=True)
    maxf1_mean = float(np.mean([m['macro_f1'] for m in maxf1_per_seed]))
    print(f"    PER-SEED MaxF1 MEAN: {maxf1_mean:.4f}")

    # ── Score-average ensemble @ FPR<=0.01 (current default) ──────────────────
    print("\n[C] Score-average ensemble @ FPR<=0.01:", flush=True)
    val_avg = np.mean([s['val']  for s in per_seed], axis=0)
    test_avg = np.mean([s['test'] for s in per_seed], axis=0)
    thr = _calibrate_fpr(val_avg, y_val_ref, 0.01)
    m_score_fpr = _metrics_at(test_avg, y_test_ref, thr)
    print(f"    f1={m_score_fpr['macro_f1']:.4f} AUC={m_score_fpr['auc_roc']:.4f} "
          f"FPR={m_score_fpr['fpr']:.4f} thr={thr:.4f}", flush=True)

    # ── Score-average ensemble @ MaxF1-on-val ─────────────────────────────────
    print("\n[D] Score-average ensemble @ MaxF1-on-val:", flush=True)
    thr = _calibrate_maxf1(val_avg, y_val_ref)
    m_score_maxf1 = _metrics_at(test_avg, y_test_ref, thr)
    print(f"    f1={m_score_maxf1['macro_f1']:.4f} AUC={m_score_maxf1['auc_roc']:.4f} "
          f"FPR={m_score_maxf1['fpr']:.4f} thr={thr:.4f}", flush=True)

    # ── Rank-average ensemble @ MaxF1 ─────────────────────────────────────────
    print("\n[E] Rank-average ensemble @ MaxF1-on-val:", flush=True)
    val_ranks = [_rank_normalize(s['val'])  for s in per_seed]
    test_ranks = [_rank_normalize(s['test']) for s in per_seed]
    val_rank_avg = np.mean(val_ranks, axis=0)
    test_rank_avg = np.mean(test_ranks, axis=0)
    thr = _calibrate_maxf1(val_rank_avg, y_val_ref)
    m_rank_maxf1 = _metrics_at(test_rank_avg, y_test_ref, thr)
    print(f"    f1={m_rank_maxf1['macro_f1']:.4f} AUC={m_rank_maxf1['auc_roc']:.4f} "
          f"FPR={m_rank_maxf1['fpr']:.4f} thr={thr:.4f}", flush=True)

    # ── Rank-average ensemble @ FPR<=0.01 ─────────────────────────────────────
    print("\n[F] Rank-average ensemble @ FPR<=0.01:", flush=True)
    thr = _calibrate_fpr(val_rank_avg, y_val_ref, 0.01)
    m_rank_fpr = _metrics_at(test_rank_avg, y_test_ref, thr)
    print(f"    f1={m_rank_fpr['macro_f1']:.4f} AUC={m_rank_fpr['auc_roc']:.4f} "
          f"FPR={m_rank_fpr['fpr']:.4f} thr={thr:.4f}", flush=True)

    # ── Per-seed predict at FPR<=0.01 → majority vote ─────────────────────────
    print("\n[G] Per-seed predict @ FPR<=0.01 → majority vote:", flush=True)
    test_preds = []
    for s in per_seed:
        thr = _calibrate_fpr(s['val'], y_val_ref, 0.01)
        test_preds.append((s['test'] >= thr).astype(np.int64))
    vote = (np.sum(test_preds, axis=0) >= 2).astype(np.int64)
    m_vote_fpr = dict(macro_f1=float(_macro_f1(y_test_ref, vote)),
                      fpr=float(((vote == 1) & (y_test_ref == 0)).sum()) / max(int((y_test_ref == 0).sum()), 1))
    print(f"    f1={m_vote_fpr['macro_f1']:.4f} FPR={m_vote_fpr['fpr']:.4f}", flush=True)

    # ── Per-seed predict at MaxF1 → majority vote ─────────────────────────────
    print("\n[H] Per-seed predict @ MaxF1-on-val → majority vote:", flush=True)
    test_preds = []
    for s in per_seed:
        thr = _calibrate_maxf1(s['val'], y_val_ref)
        test_preds.append((s['test'] >= thr).astype(np.int64))
    vote = (np.sum(test_preds, axis=0) >= 2).astype(np.int64)
    m_vote_maxf1 = dict(macro_f1=float(_macro_f1(y_test_ref, vote)),
                        fpr=float(((vote == 1) & (y_test_ref == 0)).sum()) / max(int((y_test_ref == 0).sum()), 1))
    print(f"    f1={m_vote_maxf1['macro_f1']:.4f} FPR={m_vote_maxf1['fpr']:.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    strategies = {
        "A_per_seed_fpr001":            base_mean,
        "B_per_seed_maxf1":             maxf1_mean,
        "C_score_avg_fpr001":           m_score_fpr['macro_f1'],
        "D_score_avg_maxf1":            m_score_maxf1['macro_f1'],
        "E_rank_avg_maxf1":             m_rank_maxf1['macro_f1'],
        "F_rank_avg_fpr001":            m_rank_fpr['macro_f1'],
        "G_per_seed_vote_fpr001":       m_vote_fpr['macro_f1'],
        "H_per_seed_vote_maxf1":        m_vote_maxf1['macro_f1'],
    }
    print("\n=== SUMMARY (5G-NIDD macro-F1) ===")
    for k, v in sorted(strategies.items(), key=lambda x: -x[1]):
        lift = (v - base_mean) * 100
        marker = "  <-- WINNER" if v == max(strategies.values()) else ""
        print(f"  {k:30s} {v:.4f}  (lift vs A: {lift:+.2f} pp){marker}")

    out = {
        "title": "5G-NIDD in-domain ensemble strategy comparison",
        "rationale": "FPR<=0.01 is wrong for 5G (60% attacks); MaxF1 + rank-ensemble are tested.",
        "per_seed_mean_baseline": base_mean,
        "bayes_ceiling_full_set": 0.7572,
        "w20_bar": 0.7837,
        "strategies": strategies,
        "winner": max(strategies, key=strategies.get),
        "winner_macro_f1": max(strategies.values()),
        "winner_lift_pp": (max(strategies.values()) - base_mean) * 100,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
