"""
Path 2 — per-attack-family Causal Shapley.

This script is meant to be copy-pasted as a new cell at the end of
notebooks/04_layer_a_causal_shapley.ipynb, OR run stand-alone inside
the caushap-nids .venv after the notebook has already executed and the
global variables (detector, dag, scaler, bounds, feature_filter,
feature_cols_original, background) are alive.

When run stand-alone:
    python scripts/nb04_cell_per_family_shap.py

It reconstructs those objects from the frozen artifacts automatically.

Output:
    artifacts/causal_shapley_per_family.csv
        columns: attack_family, rank, feature, phi, abs_phi,
                 direct_effect, indirect_effect
"""
from __future__ import annotations
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── locate project root ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = _HERE.parent          # running from notebooks/ maybe
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from caushap_nids.dag.io import from_graphml
from caushap_nids.models.autoencoder import DeepAutoEncoder
from caushap_nids.xai_layers.causal_shapley import causal_shapley
from caushap_nids.xai_layers.causal_shapley.cache import clear_subset_cache
from caushap_nids.data_pipeline.loaders import repair_protocol_fields

ARTIFACTS = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_PARQUET = DATA_DIR / "NF-CSE-CIC-IDS2018-V2.parquet"
OUT_CSV = ARTIFACTS / "causal_shapley_per_family.csv"

# hyperparams — match NB04 §9 paper-grade settings
SHAPLEY_SAMPLES = 200
STABILITY_SMOOTHING = 0.40
TOP_K = 10
# number of candidate attack flows per family to score before picking the best
CANDIDATES_PER_FAMILY = 64
BACKGROUND_ROWS = 512
TEMPORAL_SPLIT_FRAC = 0.70  # first 70% = train region; last 30% = test region


# ── helpers ────────────────────────────────────────────────────────────────

def _load_stack():
    """Reconstruct detector, dag, scaler, bounds, feature_cols_original from artifacts."""
    with (ARTIFACTS / "p1_config.json").open() as f:
        cfg = json.load(f)
    feature_cols_original = cfg["feature_cols_original"]
    hidden_dims = cfg.get("ae_hidden_dims", [64, 32, 16])
    dropout = cfg.get("ae_dropout", 0.1)

    dag = from_graphml(ARTIFACTS / "nf_dag_v1.graphml")

    detector = DeepAutoEncoder(
        in_dim=len(feature_cols_original),
        hidden_dims=hidden_dims,
        dropout=dropout,
        device="cpu",
    )
    detector.load(ARTIFACTS / "models" / "ae.pt")

    with (ARTIFACTS / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    bounds = np.load(ARTIFACTS / "preprocessing_bounds.npz")
    feature_filter = np.load(ARTIFACTS / "feature_filter.npz")
    return detector, dag, scaler, bounds, feature_filter, feature_cols_original


def _preprocess(df_raw: pd.DataFrame, scaler, bounds, feature_filter,
                feature_cols_original: list[str]) -> np.ndarray:
    """Replicate the NB04 §3 preprocessing pipeline exactly."""
    df, _ = repair_protocol_fields(df_raw.copy())
    X = df[feature_cols_original].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, 0.0, None)
    X = np.clip(X, bounds["pct_low"], bounds["pct_high"])
    X = np.log1p(X)
    X = scaler.transform(X)
    X = X[:, feature_filter["kept_indices"]]
    return np.clip(X, -float(bounds["final_clip_limit"]),
                       float(bounds["final_clip_limit"])).astype(np.float64)


def _best_flow_for_family(pq_file: pq.ParquetFile, family: str,
                           total_rows: int, split_frac: float,
                           cols_needed: list[str], label_col: str
                          ) -> pd.DataFrame | None:
    """Scan test region for up to CANDIDATES_PER_FAMILY rows of `family`."""
    test_start = int(total_rows * split_frac)
    collected = []
    seen = 0
    batch_size = 65_536
    for batch in pq_file.iter_batches(batch_size=batch_size,
                                       columns=cols_needed + [label_col]):
        batch_rows = batch.num_rows
        overlap_start = max(0, test_start - seen)
        if seen + batch_rows <= test_start:
            seen += batch_rows
            continue
        df_b = batch.to_pandas().iloc[overlap_start:]
        mask = df_b[label_col].astype(str) == str(family)
        df_b = df_b[mask]
        if not df_b.empty:
            collected.append(df_b)
        if sum(len(x) for x in collected) >= CANDIDATES_PER_FAMILY:
            break
        seen += batch_rows

    if not collected:
        return None
    return pd.concat(collected, ignore_index=True).head(CANDIDATES_PER_FAMILY)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading trained stack…")
    detector, dag, scaler, bounds, feature_filter, feature_cols_original = _load_stack()

    pq_file = pq.ParquetFile(RAW_PARQUET)
    total_rows = pq_file.metadata.num_rows
    label_col = "Attack"  # NF-v2 multi-class attack name column

    # ── build background once (benign train region) ────────────────────────
    print("Building background reference…")
    bg_rows = []
    seen = 0
    train_end = int(total_rows * TEMPORAL_SPLIT_FRAC)
    for batch in pq_file.iter_batches(batch_size=65_536,
                                       columns=feature_cols_original + [label_col]):
        df_b = batch.to_pandas()
        is_benign = df_b[label_col].astype(str).str.lower() == "benign"
        bg_rows.append(df_b[is_benign])
        seen += len(df_b)
        if sum(len(r) for r in bg_rows) >= BACKGROUND_ROWS or seen >= train_end:
            break
    background_raw = pd.concat(bg_rows, ignore_index=True).head(BACKGROUND_ROWS)
    background = _preprocess(background_raw, scaler, bounds, feature_filter,
                              feature_cols_original)
    print(f"  background shape: {background.shape}")

    # ── enumerate attack families ─────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels_series = pq_file.read(columns=[label_col]).to_pandas()[label_col]
    attack_families = sorted(
        l for l in labels_series.unique()
        if str(l).lower() != "benign"
    )
    print(f"Found {len(attack_families)} attack families: {attack_families[:6]}…")

    records = []
    for family in attack_families:
        print(f"  [{family}]", end=" ", flush=True)
        df_family = _best_flow_for_family(
            pq_file, family, total_rows, TEMPORAL_SPLIT_FRAC,
            feature_cols_original, label_col)
        if df_family is None or df_family.empty:
            print("  → no flows in test region, skipping")
            continue

        X_cands = _preprocess(df_family, scaler, bounds, feature_filter,
                               feature_cols_original)
        scores = detector.score(X_cands)
        best_idx = int(np.argmax(scores))
        x = X_cands[best_idx]
        print(f"score={scores[best_idx]:.4f}", end="")

        clear_subset_cache()
        exp = causal_shapley(
            detector=detector,
            dag=dag,
            x=x,
            background=background,
            n_samples=SHAPLEY_SAMPLES,
            causal_method="interventional",
            stability_smoothing=STABILITY_SMOOTHING,
        )

        order = np.argsort(-np.abs(exp.phi))[:TOP_K]
        for rank_i, feat_i in enumerate(order, start=1):
            records.append({
                "attack_family": family,
                "rank": rank_i,
                "feature": exp.feature_names[feat_i],
                "phi": float(exp.phi[feat_i]),
                "abs_phi": float(abs(exp.phi[feat_i])),
                "direct_effect": float(exp.direct_effects[feat_i]),
                "indirect_effect": float(exp.indirect_effects[feat_i]),
            })
        print(f"  top: {exp.feature_names[order[0]]}")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUT_CSV}")
    return df_out


if __name__ == "__main__":
    main()
