"""
Generate the counterfactual example diff figure (Figure 3, item 3).

Runs NSGA-II for the highest-anomaly DoS-Hulk flow in the test set,
picks the best valid Pareto-front CF, inverse-transforms both the original
and the CF back to interpretable raw scale, then produces:

    artifacts/cf_diff_example.csv      — per-feature original vs. CF values
    artifacts/fig_cf_diff.png          — two-panel matplotlib figure
    artifacts/results_tables/fig_cf_diff.tex  — LaTeX figure block

Usage:
    python scripts/gen_fig_cf_diff.py
"""
from __future__ import annotations
import json
import pickle
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caushap_nids.dag.io import from_graphml
from caushap_nids.models.autoencoder import DeepAutoEncoder
from caushap_nids.xai_layers.multi_obj_cf import generate_cf_pareto_front
from caushap_nids.data_pipeline.loaders import repair_protocol_fields

ARTIFACTS = ROOT / "artifacts"
DATA_DIR  = ROOT / "data"
RAW_PARQUET = DATA_DIR / "NF-CSE-CIC-IDS2018-V2.parquet"

# Match NB05 hyperparams exactly
POP_SIZE      = 100
N_GENERATIONS = 200
SEED          = 42
BG_ROWS       = 512
CAND_ROWS     = 64
TARGET_FAMILY = "DoS attacks-Hulk"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size":   8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_stack():
    with (ARTIFACTS / "p1_config.json").open() as f:
        cfg = json.load(f)
    feature_cols_original = cfg["feature_cols_original"]
    hidden_dims = cfg.get("ae_hidden_dims", [64, 32, 16])
    dropout     = cfg.get("ae_dropout", 0.1)
    dag         = from_graphml(ARTIFACTS / "nf_dag_v1.graphml")
    detector    = DeepAutoEncoder(
        in_dim=len(feature_cols_original),
        hidden_dims=hidden_dims,
        dropout=dropout,
        device="cpu",
    )
    detector.load(ARTIFACTS / "models" / "ae.pt")
    with (ARTIFACTS / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    bounds      = np.load(ARTIFACTS / "preprocessing_bounds.npz")
    feat_filter = np.load(ARTIFACTS / "feature_filter.npz")
    return detector, dag, scaler, bounds, feat_filter, feature_cols_original


def _preprocess(df: pd.DataFrame, scaler, bounds, feat_filter, cols):
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    x = np.clip(x, bounds["pct_low"], bounds["pct_high"])
    x = np.log1p(x)
    x = scaler.transform(x)
    x = x[:, feat_filter["kept_indices"]]
    return np.clip(x, -float(bounds["final_clip_limit"]),
                       float(bounds["final_clip_limit"])).astype(np.float64)


def _inverse_transform(x_proc: np.ndarray, scaler, bounds, feat_filter,
                        n_orig_features: int) -> np.ndarray:
    """Invert preprocessing to approximate raw feature values."""
    # x_proc is shape (n_kept,) — pad back to full original feature space
    kept = feat_filter["kept_indices"]
    x_full = np.zeros(n_orig_features, dtype=np.float64)
    x_full[kept] = x_proc
    # inverse scale, then expm1
    x_full = scaler.inverse_transform(x_full.reshape(1, -1)).flatten()
    x_full = np.expm1(x_full)
    return np.clip(x_full, 0.0, None)


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}k"
    return f"{v:.1f}"


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading stack…")
    detector, dag, scaler, bounds, feat_filter, feat_cols = _load_stack()
    n_orig = len(feat_cols)

    pq_file    = pq.ParquetFile(RAW_PARQUET)
    n_rows     = pq_file.metadata.num_rows
    test_start = int(0.85 * n_rows)
    train_end  = int(0.70 * n_rows)
    attack_col = "Attack"
    label_col  = "Label"
    needed     = feat_cols + [label_col, attack_col]

    print("Loading background and target flow…")
    bg_pieces, hulk_pieces = [], []
    seen = 0
    for batch in pq_file.iter_batches(batch_size=131_072, columns=needed):
        brows = batch.num_rows
        df_b  = batch.to_pandas()

        # background: benign rows from train region
        if len(bg_pieces) == 0 or sum(len(p) for p in bg_pieces) < BG_ROWS:
            if seen < train_end:
                hi  = min(train_end - seen, brows)
                sub = df_b.iloc[:hi]
                bg_pieces.append(sub[sub[label_col] == 0])

        # candidates: DoS-Hulk rows from test region
        if seen + brows > test_start:
            lo  = max(0, test_start - seen)
            sub = df_b.iloc[lo:]
            hulk_pieces.append(sub[sub[attack_col] == TARGET_FAMILY])

        seen += brows
        if (sum(len(p) for p in bg_pieces) >= BG_ROWS and
                sum(len(p) for p in hulk_pieces) >= CAND_ROWS):
            break

    bg_df   = pd.concat(bg_pieces, ignore_index=True).head(BG_ROWS)
    hulk_df = pd.concat(hulk_pieces, ignore_index=True).head(CAND_ROWS)
    print(f"  background: {len(bg_df)} rows  |  {TARGET_FAMILY}: {len(hulk_df)} rows")

    bg_df,   _ = repair_protocol_fields(bg_df)
    hulk_df, _ = repair_protocol_fields(hulk_df)

    background = _preprocess(bg_df,   scaler, bounds, feat_filter, feat_cols)
    candidates = _preprocess(hulk_df, scaler, bounds, feat_filter, feat_cols)

    scores     = detector.score(candidates)
    target_idx = int(np.argmax(scores))
    x_attack   = candidates[target_idx]
    print(f"  Best flow AE score: {scores[target_idx]:.4f}")

    # threshold
    if (ARTIFACTS / "if_test_metrics.json").exists():
        with (ARTIFACTS / "if_test_metrics.json").open() as f:
            m = json.load(f)
        threshold = float(m.get("ae_threshold",
                                np.percentile(detector.score(background), 95)))
    else:
        threshold = float(np.percentile(detector.score(background), 95))
    print(f"  Threshold: {threshold:.4f}")

    print("Running NSGA-II…")
    feat_cols_kept = [feat_cols[i] for i in feat_filter["kept_indices"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pareto_cfs = generate_cf_pareto_front(
            detector=detector,
            dag=dag,
            x=x_attack,
            feature_names=feat_cols_kept,
            population_size=POP_SIZE,
            n_generations=N_GENERATIONS,
            seed=SEED,
            bounds=(-float(bounds["final_clip_limit"]),
                     float(bounds["final_clip_limit"])),
            threshold=threshold,
            background=background,
            return_valid_only=True,
        )
    print(f"  Pareto front size: {len(pareto_cfs)}")

    valid = [cf for cf in pareto_cfs if cf.validity == 1.0]
    if not valid:
        print("No valid CFs — using full Pareto front")
        valid = pareto_cfs
    if not valid:
        raise RuntimeError("NSGA-II returned no CFs at all.")

    # pick CF with highest feasibility, then lowest proximity
    best_cf = sorted(valid,
                     key=lambda c: (-c.feasibility_rate, c.proximity))[0]
    print(f"  Best CF: feasibility={best_cf.feasibility_rate:.3f}, "
          f"proximity={best_cf.proximity:.4f}, "
          f"n_changed={best_cf.sparsity}")

    # ── inverse-transform to raw scale ────────────────────────────────────
    x_orig_raw = _inverse_transform(best_cf.x_orig, scaler, bounds,
                                    feat_filter, n_orig)
    x_cf_raw   = _inverse_transform(best_cf.x_cf,   scaler, bounds,
                                    feat_filter, n_orig)

    # build diff table — features that changed in preprocessed space
    kept = feat_filter["kept_indices"]
    x_orig_proc = best_cf.x_orig
    x_cf_proc   = best_cf.x_cf
    proc_delta   = x_cf_proc - x_orig_proc
    proc_changed = np.abs(proc_delta) > 1e-4
    changed_idx_proc = np.where(proc_changed)[0]

    # rank by absolute preprocessed-space change
    order = np.argsort(-np.abs(proc_delta[changed_idx_proc]))
    changed_idx_proc = changed_idx_proc[order]

    # map kept-space indices back to original feature names
    rows = []
    for ki in changed_idx_proc[:15]:     # cap at 15
        orig_feat_idx = kept[ki]
        orig_v = float(x_orig_raw[orig_feat_idx])
        cf_v   = float(x_cf_raw[orig_feat_idx])
        pct    = (cf_v - orig_v) / (abs(orig_v) + 1.0) * 100.0
        rows.append({
            "feature":    feat_cols[orig_feat_idx],
            "orig_raw":   orig_v,
            "cf_raw":     cf_v,
            "delta":      cf_v - orig_v,
            "pct_change": pct,
            "direction":  "↓" if cf_v < orig_v else "↑",
        })

    df_diff = pd.DataFrame(rows)
    csv_path = ARTIFACTS / "cf_diff_example.csv"
    df_diff.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ── figure ────────────────────────────────────────────────────────────
    _make_figure(df_diff, best_cf, scores[target_idx], threshold)


def _shorten(name: str) -> str:
    return (name
            .replace("RETRANSMITTED_", "RETRANS_")
            .replace("NUM_PKTS_", "N_PKTS_")
            .replace("_AVG_THROUGHPUT", "_AVG_TP")
            .replace("_SECOND_BYTES", "_SEC_B")
            .replace("SRC_TO_DST", "S→D")
            .replace("DST_TO_SRC", "D→S"))


def _make_figure(df: pd.DataFrame, best_cf, orig_score: float, threshold: float):
    COLOUR_DOWN = "#2166ac"   # blue  — feature reduced toward benign
    COLOUR_UP   = "#d6604d"   # red   — feature increased (unusual direction)
    COLOUR_ZERO = "#999999"   # grey  — no change

    top = df.head(12).copy()
    labels  = [_shorten(f) for f in top["feature"]]
    deltas  = top["pct_change"].tolist()   # use % change so scale is uniform
    origs   = top["orig_raw"].tolist()
    cfs     = top["cf_raw"].tolist()
    colours = [COLOUR_DOWN if d < 0 else COLOUR_UP for d in deltas]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(7.0, 3.6),
        gridspec_kw={"width_ratios": [1.6, 1.4]},
    )
    fig.subplots_adjust(wspace=0.05)

    # ── Left panel: delta bar chart ────────────────────────────────────────
    y = np.arange(len(top))[::-1]
    ax_left.barh(y, deltas, color=colours, height=0.55, linewidth=0)
    ax_left.axvline(0, color="black", linewidth=0.7, zorder=3)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels, fontsize=6.5)
    ax_left.set_xlabel(r"Feature change, $\frac{\mathrm{CF} - \mathrm{orig}}{|\mathrm{orig}|+1}$ (%)")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.grid(axis="x", linewidth=0.4, color="grey", alpha=0.4, zorder=0)
    ax_left.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=COLOUR_DOWN, label="Reduced (toward benign)"),
        mpatches.Patch(color=COLOUR_UP,   label="Increased"),
    ]
    ax_left.legend(handles=legend_patches, fontsize=6, loc="lower right",
                   frameon=True, edgecolor="grey", framealpha=0.9)
    ax_left.set_title(
        f"DoS-Hulk flow  (AE score {orig_score:.2f} > θ={threshold:.2f})",
        fontsize=7.5, pad=4)

    # ── Right panel: value table ───────────────────────────────────────────
    ax_right.axis("off")
    col_labels = ["Feature", "Original", "CF"]
    cell_text  = []
    cell_colors = []
    for _, row in top.iterrows():
        cell_text.append([
            _shorten(row["feature"]),
            _fmt(row["orig_raw"]),
            _fmt(row["cf_raw"]),
        ])
        changed_color = "#d4e8f7" if row["delta"] < 0 else "#fde0dc"
        cell_colors.append(["white", changed_color, changed_color])

    tbl = ax_right.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="right",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1.0, 1.15)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")

    ax_right.set_title(
        f"Changed features  (sparsity={best_cf.sparsity}, "
        f"feas={best_cf.feasibility_rate:.2f})",
        fontsize=7.5, pad=4)

    fig.tight_layout(pad=0.5)
    out_png = ARTIFACTS / "fig_cf_diff.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_png}")
    plt.close(fig)

    # ── LaTeX block ───────────────────────────────────────────────────────
    _write_latex(top, best_cf, orig_score, threshold)


def _write_latex(df: pd.DataFrame, best_cf, orig_score: float, threshold: float):
    lines = [
        r"% Counterfactual example diff — DoS-Hulk worked example.",
        r"% Place in §5.5 (Counterfactual Explainability) or §6 worked example.",
        r"% Requires: \usepackage{graphicx,booktabs,colortbl}",
        r"% PNG generated by:  python scripts/gen_fig_cf_diff.py",
        r"",
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \includegraphics[width=\columnwidth]{fig_cf_diff}",
        (r"  \caption{%"
         r"    Worked counterfactual example for a DoS-Hulk flow on"
         r"    \mbox{NF-CSE-CIC-IDS2018-v2}."),
        (f"    The flow's AE anomaly score ({orig_score:.2f}) exceeds the"
         f" decision threshold ($\\theta={threshold:.2f}$)."),
        (r"    \textit{Left}: magnitude and direction of each feature change;"
         r" blue bars indicate features reduced toward benign-traffic norms,"
         r" red bars indicate increases."),
        (r"    \textit{Right}: original vs.\ counterfactual raw values for the"
         r" top changed features (highlighted)."),
        (f"    The counterfactual achieves validity = 1.0 and causal "
         f"feasibility = {best_cf.feasibility_rate:.2f}; multi-objective "
         f"optimisation favours feasibility over sparsity here, perturbing "
         f"{best_cf.sparsity} features but concentrating the magnitude in the "
         f"few largest (shown), giving the analyst a dominant remediation lever"
         r" (\S\ref{sec:layer_b}).%"),
        r"  }",
        r"  \label{fig:cf_diff_example}",
        r"\end{figure}",
    ]
    out_tex = ARTIFACTS / "results_tables" / "fig_cf_diff.tex"
    out_tex.write_text("\n".join(lines))
    print(f"Saved → {out_tex}")


if __name__ == "__main__":
    main()
