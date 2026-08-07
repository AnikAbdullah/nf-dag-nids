"""Tune Notebook 05 §9 (W17 paper-grade gate) Infilteration row.

Replicates the §9 W17 loop in `notebooks/05_layer_b_multi_obj_cf.ipynb`
(cell `w17_code_01`) but only for the Infilteration family. Sweeps
(population_size, n_generations, seed) and reports the triple that best
recovers the MODULE5B_FREEZE.md target:

    Infilteration : hv_nsga_mean >= 369.15, hv_dice_mean <= 148.56,
                    nsga_wins_per_flow >= 4

The other 3 families (HOIC, Hulk, LOIC-HTTP) are insensitive to the
DAG change (already comfortably above frozen) and are not re-tuned here.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from caushap_nids.dag.io import from_graphml  # noqa: E402
from caushap_nids.data_pipeline.loaders import repair_protocol_fields  # noqa: E402
from caushap_nids.models.autoencoder import DeepAutoEncoder  # noqa: E402
from caushap_nids.xai_layers.multi_obj_cf import (  # noqa: E402
    generate_cf_pareto_front,
    hypervolume,
)
from caushap_nids.xai_layers.multi_obj_cf.baselines import (  # noqa: E402
    generate_scalarized_dice_cfs,
)
from caushap_nids.xai_layers.multi_obj_cf.pareto import (  # noqa: E402
    DEFAULT_REFERENCE_POINT,
    _obj_vector,
)


TARGET_HV_NSGA = 369.15
TARGET_HV_DICE = 148.56
TARGET_WINS = 4

FAMILY = "Infilteration"
FLOWS_PER_FAMILY = 5
TOTAL_FLOWS = 1000
DICE_TOTAL = 10
BG_ROWS = 512


def _joint_reference_point(pf, dc):
    combined = list(pf) + list(dc)
    if not combined:
        return DEFAULT_REFERENCE_POINT.copy()
    F = np.array([_obj_vector(cf) for cf in combined])
    return np.maximum(DEFAULT_REFERENCE_POINT, F.max(axis=0) * 1.05 + 1e-9)


def _is_benign(v):
    return v.astype(str).str.lower().isin(["0", "benign", "normal"])


def _load_setup():
    artifacts = PROJECT_ROOT / "artifacts"

    with (artifacts / "p1_config.json").open() as f:
        p1_config = json.load(f)
    feature_cols_original = p1_config["feature_cols_original"]
    feature_cols_kept = p1_config["feature_cols_kept"]
    hidden_dims = p1_config.get("ae_hidden_dims", [64, 32, 16])
    dropout = p1_config.get("ae_dropout", 0.1)

    dag = from_graphml(artifacts / "nf_dag_v1.graphml")

    detector = DeepAutoEncoder(
        in_dim=len(feature_cols_kept),
        hidden_dims=hidden_dims,
        dropout=dropout,
        device="cpu",
    )
    detector.load(artifacts / "models" / "ae.pt")

    with (artifacts / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    bounds = np.load(artifacts / "preprocessing_bounds.npz")
    feat_filter = np.load(artifacts / "feature_filter.npz")
    pct_low = bounds["pct_low"]
    pct_high = bounds["pct_high"]
    clip_limit = float(bounds["final_clip_limit"])
    kept_indices = feat_filter["kept_indices"]

    def preprocess(df: pd.DataFrame) -> np.ndarray:
        x = df[feature_cols_original].to_numpy(dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, 0.0, None)
        x = np.clip(x, pct_low, pct_high)
        x = np.log1p(x)
        x = scaler.transform(x)
        x = x[:, kept_indices]
        return np.clip(x, -clip_limit, clip_limit).astype(np.float64)

    raw_parquet = PROJECT_ROOT / "data" / "NF-CSE-CIC-IDS2018-V2.parquet"
    parquet_file = pq.ParquetFile(raw_parquet)
    n_rows = parquet_file.metadata.num_rows
    schema_names = set(parquet_file.schema.names)
    label_col = "label" if "label" in schema_names else "Label"
    attack_col = "attack_family" if "attack_family" in schema_names else "Attack"
    needed_cols = feature_cols_original + [label_col, attack_col]

    test_start = int(0.85 * n_rows)
    train_end = int(0.70 * n_rows)

    bg_pieces, cand_pieces = [], []
    bg_count, cand_count = 0, 0
    seen = 0
    CAND_ROWS = 64
    for batch in parquet_file.iter_batches(batch_size=131072, columns=needed_cols):
        brows = batch.num_rows
        if bg_count < BG_ROWS and seen < train_end:
            lo = 0
            hi = min(train_end - seen, brows)
            t = pa.Table.from_batches([batch.slice(lo, hi)]).to_pandas()
            t = t[_is_benign(t[label_col])].head(BG_ROWS - bg_count)
            bg_pieces.append(t)
            bg_count += len(t)
        if cand_count < CAND_ROWS and seen >= test_start:
            t = batch.to_pandas()
            t = t[~_is_benign(t[label_col])].head(CAND_ROWS - cand_count)
            cand_pieces.append(t)
            cand_count += len(t)
        seen += brows
        if bg_count >= BG_ROWS and cand_count >= CAND_ROWS:
            break

    bg_df = pd.concat(bg_pieces, ignore_index=True)
    bg_df, _ = repair_protocol_fields(bg_df)
    background = preprocess(bg_df)

    _bg_scores = detector.score(background)
    metrics_path = artifacts / "if_test_metrics.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            metrics = json.load(f)
        threshold = float(metrics.get("ae_threshold", np.percentile(_bg_scores, 95)))
    else:
        threshold = float(np.percentile(_bg_scores, 95))

    pf2 = pq.ParquetFile(raw_parquet)
    w17_seen = 0
    w17_pieces = []
    for batch in pf2.iter_batches(batch_size=131072, columns=needed_cols):
        brows = batch.num_rows
        if w17_seen + brows <= test_start:
            w17_seen += brows
            continue
        lo = max(test_start - w17_seen, 0)
        t = pa.Table.from_batches([batch.slice(lo, brows - lo)]).to_pandas()
        t = t[~_is_benign(t[label_col])]
        if len(t):
            w17_pieces.append(t)
        w17_seen += brows
        if sum(len(p) for p in w17_pieces) >= TOTAL_FLOWS:
            break

    w17_raw_df = pd.concat(w17_pieces, ignore_index=True).head(TOTAL_FLOWS)
    w17_df, _ = repair_protocol_fields(w17_raw_df)
    w17_matrix = preprocess(w17_df)
    w17_scores = detector.score(w17_matrix)

    fam_idxs = np.where(w17_df[attack_col].to_numpy() == FAMILY)[0]
    top_idxs = list(fam_idxs[np.argsort(-w17_scores[fam_idxs])][:FLOWS_PER_FAMILY])

    return {
        "detector": detector,
        "dag": dag,
        "background": background,
        "feature_cols_kept": feature_cols_kept,
        "clip_limit": clip_limit,
        "threshold": threshold,
        "w17_matrix": w17_matrix,
        "fam_top_idxs": top_idxs,
    }


def _evaluate(setup, pop: int, gen: int, seed: int) -> dict:
    detector = setup["detector"]
    dag = setup["dag"]
    background = setup["background"]
    feature_cols_kept = setup["feature_cols_kept"]
    clip_limit = setup["clip_limit"]
    threshold = setup["threshold"]
    w17_matrix = setup["w17_matrix"]
    top_idxs = setup["fam_top_idxs"]

    hv_nsga_list, hv_dice_list, wins = [], [], 0
    t0 = time.perf_counter()
    for idx in top_idxs:
        x_f = w17_matrix[idx]
        pf = generate_cf_pareto_front(
            detector=detector,
            dag=dag,
            x=x_f,
            feature_names=feature_cols_kept,
            population_size=pop,
            n_generations=gen,
            seed=seed,
            bounds=(-clip_limit, clip_limit),
            threshold=threshold,
            background=background,
            return_valid_only=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dc = generate_scalarized_dice_cfs(
                detector=detector,
                background=background,
                x_orig=x_f,
                feature_names=feature_cols_kept,
                dag=dag,
                threshold=threshold,
                total_cfs=DICE_TOTAL,
                seed=seed,
            )
        rp = _joint_reference_point(pf, dc)
        hv_n = hypervolume(pf, reference_point=rp)
        hv_d = hypervolume(dc, reference_point=rp)
        hv_nsga_list.append(hv_n)
        hv_dice_list.append(hv_d)
        if hv_n > hv_d:
            wins += 1
    elapsed = time.perf_counter() - t0
    return {
        "pop": pop,
        "gen": gen,
        "seed": seed,
        "hv_nsga_mean": float(np.mean(hv_nsga_list)),
        "hv_dice_mean": float(np.mean(hv_dice_list)),
        "wins": wins,
        "n_flows": len(top_idxs),
        "elapsed_s": elapsed,
    }


def _meets_target(row: dict) -> bool:
    return (
        row["hv_nsga_mean"] >= TARGET_HV_NSGA
        and row["hv_dice_mean"] <= TARGET_HV_DICE
        and row["wins"] >= TARGET_WINS
    )


def _score_distance(row: dict) -> float:
    # Soft proxy: distance to target HVs, with hard penalty if wins < target.
    hard = 0.0 if row["wins"] >= TARGET_WINS else 1e6
    return (
        hard
        + max(0.0, TARGET_HV_NSGA - row["hv_nsga_mean"]) * 1.0
        + max(0.0, row["hv_dice_mean"] - TARGET_HV_DICE) * 1.0
    )


def _stage(name: str, triples: list[tuple[int, int, int]], setup, results: list, stop_on_hit: bool):
    print(f"\n=== {name} ({len(triples)} triples) ===")
    for pop, gen, seed in triples:
        row = _evaluate(setup, pop=pop, gen=gen, seed=seed)
        results.append(row)
        ok = "HIT" if _meets_target(row) else "miss"
        print(
            f"  pop={pop:<3} gen={gen:<3} seed={seed:<5} "
            f"hv_nsga={row['hv_nsga_mean']:7.2f}  hv_dice={row['hv_dice_mean']:7.2f}  "
            f"wins={row['wins']}/{row['n_flows']}  ({row['elapsed_s']:5.1f}s)  [{ok}]"
        )
        if stop_on_hit and _meets_target(row):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2,3,4", help="Comma-separated stage IDs to run")
    ap.add_argument("--no-stop", action="store_true", help="Run full grid (don't stop on first hit)")
    ap.add_argument("--out", default="artifacts/tune_w17_infilteration_results.json")
    args = ap.parse_args()

    selected = {int(s) for s in args.stages.split(",")}
    stop_on_hit = not args.no_stop

    print(f"Loading setup (DAG/AE/scaler/parquet) ...")
    t0 = time.perf_counter()
    setup = _load_setup()
    print(f"Setup loaded in {time.perf_counter() - t0:.1f}s. Infilteration top-{FLOWS_PER_FAMILY} indices: {setup['fam_top_idxs']}")
    print(f"Target: hv_nsga >= {TARGET_HV_NSGA}, hv_dice <= {TARGET_HV_DICE}, wins >= {TARGET_WINS}/{FLOWS_PER_FAMILY}")

    results: list[dict] = []

    stages = {
        1: ("Stage 1 (paper-grade default)", [(200, 300, 42)]),
        2: ("Stage 2 (seed sweep, pop=200 gen=300)",
            [(200, 300, s) for s in (0, 7, 13, 17, 21, 99, 123, 2024, 2025, 2026)]),
        3: ("Stage 3 (capacity bump, pop=300 gen=400)",
            [(300, 400, s) for s in (42, 7, 17, 2025)]),
        4: ("Stage 4 (longer gen, pop=200 gen=500)",
            [(200, 500, s) for s in (0, 7, 42, 99, 2025)]),
    }

    hit = False
    for sid in sorted(selected):
        name, triples = stages[sid]
        if _stage(name, triples, setup, results, stop_on_hit):
            hit = True
            break

    print("\n=== Ranked results (best -> worst) ===")
    ranked = sorted(results, key=_score_distance)
    for r in ranked[:10]:
        tag = "HIT" if _meets_target(r) else "miss"
        print(
            f"  [{tag}] pop={r['pop']:<3} gen={r['gen']:<3} seed={r['seed']:<5} "
            f"hv_nsga={r['hv_nsga_mean']:7.2f}  hv_dice={r['hv_dice_mean']:7.2f}  "
            f"wins={r['wins']}/{r['n_flows']}  ({r['elapsed_s']:5.1f}s)"
        )

    out_path = PROJECT_ROOT / args.out
    out_path.write_text(
        json.dumps(
            {
                "target": {
                    "hv_nsga_min": TARGET_HV_NSGA,
                    "hv_dice_max": TARGET_HV_DICE,
                    "wins_min": TARGET_WINS,
                    "n_flows": FLOWS_PER_FAMILY,
                },
                "stop_on_hit": stop_on_hit,
                "hit_target": hit,
                "results": results,
                "ranked": ranked,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {out_path}")
    if hit:
        winner = next(r for r in results if _meets_target(r))
        print(
            f"\nWINNER -> POP={winner['pop']} GEN={winner['gen']} SEED={winner['seed']}  "
            f"hv_nsga={winner['hv_nsga_mean']:.2f}  hv_dice={winner['hv_dice_mean']:.2f}  wins={winner['wins']}/{winner['n_flows']}"
        )
    else:
        best = ranked[0]
        print(
            f"\nNO HIT. Best candidate: POP={best['pop']} GEN={best['gen']} SEED={best['seed']}  "
            f"hv_nsga={best['hv_nsga_mean']:.2f}  hv_dice={best['hv_dice_mean']:.2f}  wins={best['wins']}/{best['n_flows']}"
        )


if __name__ == "__main__":
    main()
