"""Lean targeted probe: try to flip the seed43 Lipschitz regression in ~10-15 min.

Uses fewer flows + perturbations to accelerate per-variant cost from 13 min → ~2 min,
then tests the three most-promising A2 variants:
  - smoothing=0.3 (graph-smooth attribution through DAG)
  - smoothing=0.5 (stronger smoothing)
  - smoothing=0.3 + n_samples=200 (smoothing + lower sampling noise)

Decision rule: a variant FLIPS the regression if A2 median per-instance < A1 median.
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
OUT_JSON = ART / "seed43_lipschitz_fast_probe.json"

DS, SEED = "nf_cic2018", 43
N_FAITH = 10        # half the matrix probe — still 10-point per-instance distribution
N_PERT = 10         # half — keeps statistical power for median
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
    print(f"  loaded in {time.time()-t_start:.1f}s; X_faith={X_faith.shape}", flush=True)

    results = {}

    # A1 baseline
    print(f"\n[A1] vanilla KernelSHAP (n_samples=50) ...", flush=True)
    t0 = time.time()
    a1_lip = _per_instance_lipschitz(
        lambda x: vanilla_kernel_shap(detector, x, bg, n_samples=50,
                                       feature_names=NF_V2_FEATURE_COLS).phi,
        X_faith,
    )
    a1_median = float(np.median(a1_lip)); a1_mean = float(np.mean(a1_lip))
    print(f"  median={a1_median:.4f} mean={a1_mean:.4f} ({time.time()-t0:.1f}s)", flush=True)
    results["A1_baseline"] = {"median": a1_median, "mean": a1_mean,
                              "per_instance": [float(v) for v in a1_lip]}

    # A2 baseline (smoothing=0, n_samples=50) — confirm the regression on the lean probe
    print(f"\n[A2_baseline] smoothing=0.0, n_samples=50 ...", flush=True)
    t0 = time.time()
    a2_base = _per_instance_lipschitz(
        lambda x: causal_shapley_values(detector, dag, x, bg, n_samples=50,
                                         causal_method="interventional",
                                         stability_smoothing=0.0).phi,
        X_faith,
    )
    a2b_median = float(np.median(a2_base))
    imp = 100 * (a1_median - a2b_median) / a1_median
    print(f"  median={a2b_median:.4f} imp_vs_a1={imp:+.2f}% ({time.time()-t0:.1f}s)", flush=True)
    results["A2_baseline"] = {"median": a2b_median, "mean": float(np.mean(a2_base)),
                              "imp_pct": imp,
                              "per_instance": [float(v) for v in a2_base]}

    # Try A2 variants
    probes = [
        ("A2_smooth_03", 0.3, 50),
        ("A2_smooth_05", 0.5, 50),
        ("A2_smooth_07", 0.7, 50),
        ("A2_smooth_03_n200", 0.3, 200),
    ]
    for label, smoothing, n_samples in probes:
        print(f"\n[{label}] smoothing={smoothing}, n_samples={n_samples} ...", flush=True)
        t0 = time.time()
        def explain_one(x, _sm=smoothing, _ns=n_samples):
            return causal_shapley_values(detector, dag, x, bg, n_samples=_ns,
                                          causal_method="interventional",
                                          stability_smoothing=_sm).phi
        a2_lip = _per_instance_lipschitz(explain_one, X_faith)
        a2m = float(np.median(a2_lip))
        imp = 100 * (a1_median - a2m) / a1_median
        elapsed = time.time() - t0
        flag = "  ✓ FLIPS" if a2m < a1_median else ""
        print(f"  median={a2m:.4f} imp_vs_a1={imp:+.2f}% ({elapsed:.1f}s){flag}", flush=True)
        results[label] = {"median": a2m, "mean": float(np.mean(a2_lip)),
                          "imp_pct": imp, "smoothing": smoothing, "n_samples": n_samples,
                          "elapsed_s": elapsed,
                          "per_instance": [float(v) for v in a2_lip]}
        OUT_JSON.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  A1 baseline median: {a1_median:.4f}")
    rows = [(k, v) for k, v in results.items() if k.startswith("A2")]
    rows.sort(key=lambda x: -x[1]["imp_pct"])
    for k, v in rows:
        flag = "  ✓ FLIPS" if v["imp_pct"] > 0 else ""
        print(f"  {k:24s}  median={v['median']:.4f}  imp={v['imp_pct']:+7.2f}%{flag}")
    winner = max(rows, key=lambda x: x[1]["imp_pct"])
    if winner[1]["imp_pct"] > 0:
        print(f"\n  ✓ WINNER: {winner[0]} flips A2 to BEAT A1 (+{winner[1]['imp_pct']:.2f}%)")
    else:
        print(f"\n  ✗ Best variant {winner[0]} still loses ({winner[1]['imp_pct']:+.2f}%)")
    print(f"\n  Total: {time.time()-t_start:.1f}s")
    print(f"  Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
