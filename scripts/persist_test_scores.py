"""Persist per-flow test scores for all 4 NF-v2 datasets so we can build ROC
curves on all of them (not just NF-CIC2018).

The 72-run matrix saves only aggregated AUC values, not per-flow scores. This
script reuses the cached AE+IF detectors (one per dataset, seed=42) from
`artifacts/models/_shared/detector-*/`, runs scoring on the test split, and
saves `(Label, attack_family, ae_score, if_score, ens_score, y_pred)` to
`artifacts/test_scores_<dataset>.csv` for downstream ROC-curve generation.

Why not re-run the matrix?  Re-running 72 cells would take ~17 hours. This
script only does *detection scoring* (no XAI), reuses the cached detectors,
and finishes in ~15-30 minutes total across all 4 datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from caushap_nids.experiments.runner import (
    load_run_config,
    _load_dataset,
    _train_or_load_detector,
    _EnsembleScoreAdapter,
)
from caushap_nids.data_pipeline.loaders import load_nf_v2

ARTIFACTS = ROOT / "artifacts"
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"

DATASETS = ("nf_cic2018", "nf_unsw15", "edge_iiotset", "5g_nidd")
SEED = 42  # one seed is enough for ROC curves (per-seed std visible in matrix AUCs)


def main() -> None:
    cfg = load_run_config("A0_baseline", CONFIGS_DIR)

    for ds in DATASETS:
        print(f"\n── {ds} (seed={SEED}) ──")
        # 1. Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(ds, SEED, DATA_DIR)
        # 2. Get the attack_family column from the raw test split
        test_df = load_nf_v2(ds, "test", seed=SEED, data_dir=str(DATA_DIR))
        attack_family = test_df["attack_family"].to_list() if "attack_family" in test_df.columns else [""] * len(y_test)

        # 3. Load detector from cache (or train if missing)
        (ae, if_det), (ae_norm, if_norm, alpha, ae_feature_indices), threshold, diag = _train_or_load_detector(
            cfg, X_train, y_train, X_val, y_val,
            ARTIFACTS, config_name="A0_baseline", dataset=ds, seed=SEED,
        )
        print(f"  detector: alpha={alpha:.3f}  threshold={threshold:.4f}  "
              f"feature_indices={ae_feature_indices}")

        # 4. Score test set — get individual + ensemble scores
        detector = _EnsembleScoreAdapter(
            ae=ae, if_det=if_det,
            ae_norm=ae_norm, if_norm=if_norm,
            alpha=alpha, ae_feature_indices=ae_feature_indices,
        )
        ens_scores = detector.score(X_test)

        # Raw AE + IF scores (pre-ensemble) for the comparison curves
        if ae_feature_indices is None:
            ae_raw = ae.score(X_test)
        else:
            ae_raw = ae.score(X_test, feature_indices=ae_feature_indices)
        ae_normalized = ae_norm.transform(ae_raw)
        if_raw = if_det.score(X_test)
        if_normalized = if_norm.transform(if_raw)

        y_pred = (ens_scores >= threshold).astype(int)

        # 5. Save per-flow scores
        out = pd.DataFrame({
            "Label":        y_test.astype(int),
            "attack_family": attack_family,
            "ae_score":     ae_normalized,
            "if_score":     if_normalized,
            "ens_score":    ens_scores,
            "y_pred":       y_pred,
        })
        out_path = ARTIFACTS / f"test_scores_{ds}.csv"
        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path.relative_to(ROOT)}  ({len(out):,} flows)")

        # Quick AUC sanity
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_test)) > 1:
            auc_ae = roc_auc_score(y_test, ae_normalized)
            auc_if = roc_auc_score(y_test, if_normalized)
            auc_en = roc_auc_score(y_test, ens_scores)
            print(f"  AUC: AE={auc_ae:.4f}  IF={auc_if:.4f}  AE+IF={auc_en:.4f}")
        else:
            print(f"  (single-class test set — skipping AUC)")


if __name__ == "__main__":
    main()
