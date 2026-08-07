"""AUC-weighted seed ensemble probe.

Tests whether weighting each seed's normalised score by its val AUC^k (k in {0,1,2,4})
beats unweighted averaging. k=0 is the current default (equal weights).
Higher-AUC seeds get more weight as k grows.

Runs the same per-seed protocol as in_domain_3seed_ensemble.py, then averages
test scores weighted by val AUC.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from caushap_nids.experiments.runner import (
    _load_dataset, _train_or_load_detector, load_run_config, _EnsembleScoreAdapter,
)
from caushap_nids.experiments import DATASETS, DEFAULT_SEEDS

ART = ROOT / "artifacts"
OUT_JSON = ART / "in_domain_auc_weighted_probe.json"

# Same per-dataset threshold strategy as the main ensemble
THRESHOLD_STRATEGY = {
    "nf_cic2018":   "fpr_le_001",
    "nf_unsw15":    "fpr_le_001",
    "edge_iiotset": "maxf1",
    "5g_nidd":      "maxf1",
}


def _calibrate_fpr(scores, y, target_fpr=0.01):
    order = np.argsort(-scores, kind="mergesort")
    sord = scores[order]; yord = y[order]
    n_neg = int((yord == 0).sum())
    if n_neg == 0: return float(sord[0])
    cum_fp = np.cumsum(yord == 0)
    fpr = cum_fp / n_neg
    valid = np.where(fpr <= target_fpr)[0]
    return float(sord[0]) + 1e-9 if len(valid) == 0 else float(sord[int(valid[-1])])


def _calibrate_maxf1(scores, y):
    order = np.argsort(-scores, kind="mergesort")
    sord = scores[order]; yord = y[order]
    n_pos = int(yord.sum()); n_neg = int(len(yord) - n_pos)
    if n_pos == 0 or n_neg == 0: return float(sord[0])
    block = np.flatnonzero(np.r_[sord[1:] != sord[:-1], True])
    thr = sord[block]
    tp = np.cumsum(yord == 1).astype(np.float64)[block]
    fp = np.cumsum(yord == 0).astype(np.float64)[block]
    fn = n_pos - tp; tn = n_neg - fp
    tpr = tp / n_pos
    pre_a = tp / np.maximum(tp + fp, 1e-9); f1_a = 2 * pre_a * tpr / np.maximum(pre_a + tpr, 1e-9)
    pre_b = tn / np.maximum(tn + fn, 1e-9); rec_b = tn / n_neg
    f1_b = 2 * pre_b * rec_b / np.maximum(pre_b + rec_b, 1e-9)
    return float(thr[int((0.5 * (f1_a + f1_b)).argmax())])


def _calibrate(scores, y, strategy):
    return _calibrate_fpr(scores, y) if strategy == "fpr_le_001" else _calibrate_maxf1(scores, y)


def _macro_f1(y, pred):
    f1_a = f1_score(y, pred, pos_label=1, zero_division=0)
    f1_b = f1_score(y, pred, pos_label=0, zero_division=0)
    return 0.5 * (f1_a + f1_b)


def main():
    cfg = load_run_config("A4_full", ROOT / "configs")
    results = {}
    for ds in DATASETS:
        print(f"\n=== {ds} ({THRESHOLD_STRATEGY[ds]}) ===", flush=True)
        per_seed = []
        y_val_ref = None; y_test_ref = None
        for seed in DEFAULT_SEEDS:
            t0 = time.time()
            X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(ds, seed, ROOT / "data")
            if y_val_ref is None: y_val_ref, y_test_ref = y_val, y_test
            (ae, if_det), (ae_norm, if_norm, alpha, ae_feat_idx), _, _ = _train_or_load_detector(
                cfg, X_train, y_train, X_val, y_val, ART, "A4_full", ds, seed
            )
            adapter = _EnsembleScoreAdapter(ae=ae, if_det=if_det, ae_norm=ae_norm, if_norm=if_norm,
                                             alpha=alpha, ae_feature_indices=ae_feat_idx)
            val_scores = adapter.score(X_val)
            test_scores = adapter.score(X_test)
            val_auc = float(roc_auc_score(y_val, val_scores))
            per_seed.append({'seed': seed, 'val': val_scores, 'test': test_scores, 'val_auc': val_auc})
            print(f"  seed {seed} val_AUC={val_auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        # Try multiple k values
        ds_results = {}
        for k in [0, 1, 2, 4]:
            aucs = np.array([s['val_auc'] for s in per_seed])
            weights = aucs ** k
            weights = weights / weights.sum()
            val_avg = sum(w * s['val'] for w, s in zip(weights, per_seed))
            test_avg = sum(w * s['test'] for w, s in zip(weights, per_seed))
            thr = _calibrate(val_avg, y_val_ref, THRESHOLD_STRATEGY[ds])
            pred = (test_avg >= thr).astype(np.int64)
            mf1 = float(_macro_f1(y_test_ref, pred))
            ds_results[f'k_{k}'] = {
                'weights': [float(w) for w in weights],
                'macro_f1': mf1,
                'auc_roc': float(roc_auc_score(y_test_ref, test_avg)),
                'threshold': float(thr),
            }
            label = "EQUAL" if k == 0 else f"AUC^{k}"
            print(f"  {label:10s}  macro-F1={mf1:.4f}  weights={[f'{w:.3f}' for w in weights]}",
                  flush=True)

        # Highlight winner
        best_k = max(ds_results, key=lambda k: ds_results[k]['macro_f1'])
        winner_lift = (ds_results[best_k]['macro_f1'] - ds_results['k_0']['macro_f1']) * 100
        print(f"  WINNER: {best_k} (macro-F1={ds_results[best_k]['macro_f1']:.4f}, "
              f"lift vs equal: {winner_lift:+.2f} pp)", flush=True)
        results[ds] = {'best_k': best_k, 'lift_vs_equal_pp': winner_lift, 'results_by_k': ds_results}

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n=== SUMMARY ===")
    any_lift = False
    for ds, r in results.items():
        equal = r['results_by_k']['k_0']['macro_f1']
        best = r['results_by_k'][r['best_k']]['macro_f1']
        lift = (best - equal) * 100
        flag = " (LIFT)" if lift > 0.1 else ""
        print(f"  {ds:14s}  equal={equal:.4f}  best({r['best_k']})={best:.4f}  lift={lift:+.2f} pp{flag}")
        if lift > 0.1: any_lift = True
    print(f"\nVerdict: AUC-weighting {'IMPROVES' if any_lift else 'does NOT improve'} on at least one dataset.")


if __name__ == "__main__":
    main()
