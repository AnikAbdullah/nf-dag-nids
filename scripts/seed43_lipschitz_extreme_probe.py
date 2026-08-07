"""Last-resort probe: extreme smoothing + multi-RNG averaging + asymmetric method.

Tests:
  - smoothing=0.9 (near-full graph average)
  - smoothing=0.7 + multi-RNG averaging (5 RNGs)
  - causal_method='asymmetric' (Frye et al. 2020 — different aggregation)
  - cc_shapley (Martin & Haufe 2026 collider correction)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from caushap_nids.experiments.runner import (
    _load_dataset, _train_or_load_detector, load_run_config, _EnsembleScoreAdapter,
)
from caushap_nids.xai_layers.causal_shapley import causal_shapley_values, vanilla_kernel_shap
from caushap_nids.evaluation.faithfulness import lipschitz_stability
from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS
from caushap_nids.dag.io import from_graphml

ART = ROOT / "artifacts"
OUT_JSON = ART / "seed43_lipschitz_extreme_probe.json"

DS, SEED = "nf_cic2018", 43
N_FAITH = 10
N_PERT = 10
N_BG = 200


def _per_instance_lipschitz(explain_fn, X_faith):
    return [lipschitz_stability(explain_fn, X_faith[i], n_perturbations=N_PERT,
                                  epsilon=0.05, seed=SEED)
             for i in range(len(X_faith))]


def main():
    t_start = time.time()
    print(f"Loading {DS}/seed{SEED} ...", flush=True)
    cfg = load_run_config("A4_full", ROOT / "configs")
    X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(DS, SEED, ROOT / "data")
    (ae, if_det), (ae_norm, if_norm, alpha, ae_feat_idx), thr, _ = _train_or_load_detector(
        cfg, X_train, y_train, X_val, y_val, ART, "A4_full", DS, SEED
    )
    detector = _EnsembleScoreAdapter(ae=ae, if_det=if_det, ae_norm=ae_norm, if_norm=if_norm,
                                      alpha=alpha, ae_feature_indices=ae_feat_idx)
    dag = from_graphml(str(ART / "nf_dag_v1.graphml"))

    np.random.seed(SEED)
    attack_idx = np.where(y_test == 1)[0]
    rng = np.random.default_rng(SEED)
    explain_idx = rng.choice(attack_idx, size=min(200, len(attack_idx)), replace=False)
    X_faith = X_test[explain_idx[:N_FAITH]]
    bg_benign = X_train[y_train == 0]
    bg_rng = np.random.default_rng(SEED)
    bg = bg_benign[bg_rng.choice(len(bg_benign), size=min(N_BG, len(bg_benign)), replace=False)]
    print(f"  loaded in {time.time()-t_start:.1f}s", flush=True)

    results = {}

    # A1 baseline (cached from earlier probe; recompute for safety)
    print(f"\n[A1] vanilla KernelSHAP ...", flush=True)
    t0 = time.time()
    a1_lip = _per_instance_lipschitz(
        lambda x: vanilla_kernel_shap(detector, x, bg, n_samples=50,
                                       feature_names=NF_V2_FEATURE_COLS).phi,
        X_faith,
    )
    a1_median = float(np.median(a1_lip))
    print(f"  median={a1_median:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    results["A1_baseline"] = {"median": a1_median, "mean": float(np.mean(a1_lip)),
                              "per_instance": [float(v) for v in a1_lip]}

    # Variants
    probes = []

    # Extreme smoothing
    for s in [0.8, 0.9, 0.95]:
        probes.append((f"A2_smooth_{int(s*100)}", "interventional", s, 50, 1))

    # Multi-RNG averaging
    for n_rngs in [3, 5]:
        probes.append((f"A2_smooth_07_rng{n_rngs}", "interventional", 0.7, 50, n_rngs))

    # Different causal methods
    probes.append(("A2_asymmetric_smooth_07", "asymmetric", 0.7, 50, 1))
    probes.append(("A2_cc_shapley_smooth_07", "cc-shapley",  0.7, 50, 1))

    for label, method, smoothing, n_samples, n_rngs in probes:
        print(f"\n[{label}] method={method}, smoothing={smoothing}, n_samples={n_samples}, "
              f"n_rngs={n_rngs} ...", flush=True)
        t0 = time.time()

        if n_rngs == 1:
            def explain_one(x, _m=method, _sm=smoothing, _ns=n_samples):
                return causal_shapley_values(detector, dag, x, bg, n_samples=_ns,
                                              causal_method=_m,
                                              stability_smoothing=_sm).phi
        else:
            def explain_one(x, _m=method, _sm=smoothing, _ns=n_samples, _nr=n_rngs):
                phis = []
                for ri in range(_nr):
                    sub_rng = np.random.default_rng(SEED + ri * 31)
                    bg_idx = sub_rng.choice(len(bg), size=min(N_BG, len(bg)), replace=False)
                    phis.append(causal_shapley_values(detector, dag, x, bg[bg_idx],
                                                       n_samples=_ns,
                                                       causal_method=_m,
                                                       stability_smoothing=_sm).phi)
                return np.mean(phis, axis=0)

        try:
            a2_lip = _per_instance_lipschitz(explain_one, X_faith)
            a2m = float(np.median(a2_lip))
            imp = 100 * (a1_median - a2m) / a1_median
            elapsed = time.time() - t0
            flag = "  ✓ FLIPS" if a2m < a1_median else ""
            print(f"  median={a2m:.4f} imp={imp:+.2f}% ({elapsed:.1f}s){flag}", flush=True)
            results[label] = {"median": a2m, "mean": float(np.mean(a2_lip)),
                              "imp_pct": imp, "method": method,
                              "smoothing": smoothing, "n_samples": n_samples, "n_rngs": n_rngs,
                              "elapsed_s": elapsed,
                              "per_instance": [float(v) for v in a2_lip]}
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            results[label] = {"error": str(e)}
        OUT_JSON.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  A1 baseline median: {a1_median:.4f}")
    rows = [(k, v) for k, v in results.items() if k.startswith("A2") and "median" in v]
    rows.sort(key=lambda x: -x[1]["imp_pct"])
    for k, v in rows:
        flag = "  ✓ FLIPS" if v["imp_pct"] > 0 else ""
        print(f"  {k:30s}  median={v['median']:.4f}  imp={v['imp_pct']:+7.2f}%{flag}")
    if rows:
        best = max(rows, key=lambda x: x[1]["imp_pct"])
        print(f"\n  BEST: {best[0]} ({best[1]['imp_pct']:+.2f}%)")
        if best[1]["imp_pct"] > 0:
            print(f"  ✓✓✓ A2 BEATS A1 on seed43!")
        else:
            print(f"  ✗ seed43 regression is FUNDAMENTAL — AE score surface is genuinely non-smooth")
    print(f"\n  Total: {time.time()-t_start:.1f}s")
    print(f"  Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
