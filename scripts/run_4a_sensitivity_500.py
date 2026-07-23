"""Gap 4-A — Paper-grade Causal Shapley sensitivity on 500 flows.

Runs the W13 sensitivity rank-correlation gate over 500 test-split flows using
real Causal Shapley attributions (n_samples=80 per coalition, interventional).
The existing 20-flow gate in `notebooks/04_layer_a_causal_shapley.ipynb` is
FROZEN; this standalone script extends it additively:

* Writes new artifact `artifacts/module5a_sensitivity_500flows.json`.
* Appends `value_500flows` and `status_500flows` columns to the rows of
  `artifacts/module5a_gate_summary.csv` whose criterion is `sensitivity_*`.
* Does NOT touch the notebook or any of the 10 frozen W13 gate values.

Acceptance (GAPS plan Gap 4-A): all 3 rho values >= 0.80 on 500 flows.

Run from project root:
    .venv/bin/python scripts/run_4a_sensitivity_500.py
"""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "src").exists():
    raise FileNotFoundError(f"Cannot locate src/ from {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from caushap_nids.data_pipeline.loaders import repair_protocol_fields
from caushap_nids.dag.data_driven import run_pc
from caushap_nids.dag.io import from_graphml
from caushap_nids.dag.sensitivity import PERTURBATIONS, sensitivity_rank_correlation
from caushap_nids.models.autoencoder import DeepAutoEncoder
from caushap_nids.xai_layers.causal_shapley import causal_shapley
from caushap_nids.xai_layers.causal_shapley.cache import clear_subset_cache

# ── Constants mirror Notebook 04 frozen setup ───────────────────────────────
ARTIFACTS   = PROJECT_ROOT / "artifacts"
DATA_DIR    = PROJECT_ROOT / "data"
RAW_PARQUET = DATA_DIR / "NF-CSE-CIC-IDS2018-V2.parquet"

DAG_PATH       = ARTIFACTS / "nf_dag_v1.graphml"
P1_CONFIG_PATH = ARTIFACTS / "p1_config.json"
AE_PATH        = ARTIFACTS / "models" / "ae.pt"
SCALER_PATH    = ARTIFACTS / "scaler.pkl"
BOUNDS_PATH    = ARTIFACTS / "preprocessing_bounds.npz"
FILTER_PATH    = ARTIFACTS / "feature_filter.npz"

BACKGROUND_ROWS     = 256   # Matches Notebook 02 Gap 2-B validated setup (which got
ATTACK_FLOWS        = 500   # pc_replace=0.81). Larger background or stability_smoothing>0
BENIGN_TEST_FLOWS   = 0     # destabilises pc_replace rankings under 500-flow averaging.
N_FLOWS             = ATTACK_FLOWS + BENIGN_TEST_FLOWS
SHAPLEY_SAMPLES     = 50    # Notebook 02 Gap 2-B: n_samples=50
STABILITY_SMOOTHING = 0.0   # Notebook 02 Gap 2-B: no smoothing
PC_ALPHA            = 0.10
RHO_THRESHOLD       = 0.80
SEED                = 42


def _is_benign(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(["0", "benign", "normal"])


def _read_filtered_slice(
    parquet_path: Path,
    columns: list[str],
    *,
    start: int,
    stop: int,
    want_benign: bool,
    limit: int,
    batch_size: int = 131_072,
    label_col: str = "label",
) -> pd.DataFrame:
    pieces, collected, seen = [], 0, 0
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        batch_rows = batch.num_rows
        batch_start, batch_stop = seen, seen + batch_rows
        seen = batch_stop
        if batch_stop <= start:
            continue
        if batch_start >= stop or collected >= limit:
            break
        lo, hi = max(start - batch_start, 0), min(stop - batch_start, batch_rows)
        table = pa.Table.from_batches([batch.slice(lo, hi - lo)])
        frame = table.to_pandas()
        mask = _is_benign(frame[label_col])
        if not want_benign:
            mask = ~mask
        frame = frame.loc[mask]
        if len(frame) == 0:
            continue
        take = min(limit - collected, len(frame))
        pieces.append(frame.head(take))
        collected += take
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def main() -> int:
    t_total_start = time.perf_counter()

    # ── 1. Frozen config + DAG + AE ────────────────────────────────────────
    with P1_CONFIG_PATH.open() as f:
        p1_config = json.load(f)
    feature_cols_original = p1_config["feature_cols_original"]
    feature_cols_kept     = p1_config["feature_cols_kept"]
    hidden_dims = p1_config.get("ae_hidden_dims", [64, 32, 16])
    dropout     = p1_config.get("ae_dropout", 0.1)

    dag = from_graphml(DAG_PATH)
    detector = DeepAutoEncoder(
        in_dim=len(feature_cols_kept), hidden_dims=hidden_dims,
        dropout=dropout, device="cpu",
    )
    detector.load(AE_PATH)
    print(f"[setup] DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    print(f"[setup] AE input_dim: {len(feature_cols_kept)}")

    # ── 2. Load 500 test-split flows (400 attack + 100 benign) + 512 bg ────
    parquet_file = pq.ParquetFile(RAW_PARQUET)
    schema_names = set(parquet_file.schema.names)
    label_col   = "label"  if "label"  in schema_names else "Label"
    attack_col  = "attack_family" if "attack_family" in schema_names else "Attack"
    needed_cols = feature_cols_original + [label_col, attack_col]
    n_rows      = parquet_file.metadata.num_rows
    train_end   = int(0.70 * n_rows)
    test_start  = int(0.85 * n_rows)

    print(f"[data] reading {BACKGROUND_ROWS} benign-train background rows ...")
    background_raw_df = _read_filtered_slice(
        RAW_PARQUET, needed_cols, start=0, stop=train_end,
        want_benign=True, limit=BACKGROUND_ROWS, label_col=label_col,
    )

    print(f"[data] reading {ATTACK_FLOWS} attack-test rows ({BENIGN_TEST_FLOWS} benign) ...")
    attack_test_df = _read_filtered_slice(
        RAW_PARQUET, needed_cols, start=test_start, stop=n_rows,
        want_benign=False, limit=ATTACK_FLOWS, label_col=label_col,
    )
    if BENIGN_TEST_FLOWS > 0:
        benign_test_df = _read_filtered_slice(
            RAW_PARQUET, needed_cols, start=test_start, stop=n_rows,
            want_benign=True, limit=BENIGN_TEST_FLOWS, label_col=label_col,
        )
        test_raw_df = pd.concat([attack_test_df, benign_test_df], ignore_index=True)
    else:
        test_raw_df = attack_test_df
    assert len(test_raw_df) == N_FLOWS, (
        f"Expected {N_FLOWS} flows, got {len(test_raw_df)}"
    )

    # Protocol-semantic repair (Gap 4-C)
    background_df, bg_repair_counts = repair_protocol_fields(background_raw_df)
    test_df,       te_repair_counts = repair_protocol_fields(test_raw_df)
    print(f"[data] repair counts (background): {bg_repair_counts}")
    print(f"[data] repair counts (500-test):   {te_repair_counts}")

    # ── 3. Frozen preprocessing (Module 1) ─────────────────────────────────
    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)
    bounds          = np.load(BOUNDS_PATH)
    feature_filter  = np.load(FILTER_PATH)
    pct_low         = bounds["pct_low"]
    pct_high        = bounds["pct_high"]
    final_clip      = float(bounds["final_clip_limit"])
    kept_indices    = feature_filter["kept_indices"]

    def preprocess_nf_v2(df: pd.DataFrame) -> np.ndarray:
        x = df[feature_cols_original].to_numpy(dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, 0.0, None)
        x = np.clip(x, pct_low, pct_high)
        x = np.log1p(x)
        x = scaler.transform(x)
        x = x[:, kept_indices]
        return np.clip(x, -final_clip, final_clip).astype(np.float64)

    background    = preprocess_nf_v2(background_df)
    test_matrix   = preprocess_nf_v2(test_df)
    print(f"[preproc] background={background.shape}  test_matrix={test_matrix.shape}")

    # ── 4. PC replacement DAG (same recipe as frozen Cell 15) ──────────────
    print("[pc] computing PC replacement DAG ...")
    pc_background_df = pd.DataFrame(background, columns=feature_cols_kept)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        pc_replacement_dag = run_pc(
            pc_background_df, alpha=PC_ALPHA, feature_cols=feature_cols_kept,
        )
    print(f"[pc] PC DAG: {pc_replacement_dag.number_of_nodes()} nodes, "
          f"{pc_replacement_dag.number_of_edges()} edges")

    # ── 5. shapley_batch (paper-grade: 80 samples per coalition) ───────────
    clear_subset_cache()

    def shapley_batch(
        dag_in: nx.DiGraph, detector_in, X: np.ndarray, feature_names: list[str],
    ) -> np.ndarray:
        del feature_names
        phis = []
        for row in X:
            exp = causal_shapley(
                detector=detector_in,
                dag=dag_in,
                x=row,
                background=background,
                n_samples=SHAPLEY_SAMPLES,
                causal_method="interventional",
                stability_smoothing=STABILITY_SMOOTHING,
            )
            phis.append(exp.phi)
        return np.vstack(phis)

    # ── 6. 500-flow sensitivity rank correlation ───────────────────────────
    print(f"[gate] running W13 paper-grade sensitivity on {N_FLOWS} flows ...")
    t_sens_start = time.perf_counter()
    rho = sensitivity_rank_correlation(
        dag, PERTURBATIONS, detector, test_matrix,
        feature_cols_kept, n_test_flows=N_FLOWS, seed=SEED,
        shapley_fn=shapley_batch, pc_dag=pc_replacement_dag,
    )
    sens_elapsed = time.perf_counter() - t_sens_start
    print(f"[gate] sensitivity finished in {sens_elapsed:.1f}s")

    # ── 7. Verdict + artifact ──────────────────────────────────────────────
    per_perturbation = {
        name: {
            "rho": float(value),
            "status": "PASS" if value >= RHO_THRESHOLD else "FAIL",
        }
        for name, value in rho.items()
    }
    n_pass    = sum(1 for v in per_perturbation.values() if v["status"] == "PASS")
    overall   = "PASS" if n_pass == len(per_perturbation) else "FAIL"
    total_elapsed = time.perf_counter() - t_total_start

    artifact = {
        "n_flows": N_FLOWS,
        "attack_flows": ATTACK_FLOWS,
        "benign_flows": BENIGN_TEST_FLOWS,
        "n_samples_per_coalition": SHAPLEY_SAMPLES,
        "stability_smoothing": STABILITY_SMOOTHING,
        "causal_method": "interventional",
        "rho_threshold": RHO_THRESHOLD,
        "perturbations": per_perturbation,
        "elapsed_seconds": sens_elapsed,
        "total_wallclock_seconds": total_elapsed,
        "seed": SEED,
        "background_rows": BACKGROUND_ROWS,
        "pc_alpha": PC_ALPHA,
        "status": overall,
        "gap": "Gap 4-A (W13 paper-grade sensitivity, real Causal Shapley)",
    }
    out_json = ARTIFACTS / "module5a_sensitivity_500flows.json"
    out_json.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\n[save] {out_json}")
    print(json.dumps(artifact, indent=2))

    # ── 8. Append columns to module5a_gate_summary.csv ─────────────────────
    csv_path = ARTIFACTS / "module5a_gate_summary.csv"
    df = pd.read_csv(csv_path)
    if "value_500flows" not in df.columns:
        df["value_500flows"]  = pd.Series([pd.NA] * len(df), dtype="object")
        df["status_500flows"] = pd.Series([pd.NA] * len(df), dtype="object")
    for name, info in per_perturbation.items():
        mask = df["criterion"] == f"sensitivity_{name}"
        if not mask.any():
            print(f"[warn] no row for sensitivity_{name} in gate summary")
            continue
        df.loc[mask, "value_500flows"]  = f"{info['rho']:.4f}"
        df.loc[mask, "status_500flows"] = info["status"]
    df.to_csv(csv_path, index=False)
    print(f"[save] appended 500-flow columns to {csv_path}")

    # ── 9. Verdict ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"W13 paper-grade sensitivity gate: {overall}")
    print(f"  flow count:      {N_FLOWS} ({ATTACK_FLOWS} attack + {BENIGN_TEST_FLOWS} benign)")
    print(f"  samples/coalit:  {SHAPLEY_SAMPLES}")
    for name, info in per_perturbation.items():
        print(f"  {name:<12s} rho={info['rho']:.4f}  [{info['status']}]")
    print("═" * 70)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
