"""Build ROC curves for all 4 NF-v2 datasets.

Per-flow scores come from:
  - `artifacts/test_scores_<dataset>.csv`  (all 4 datasets, matrix-aligned seed=42,
                                            from `persist_test_scores.py`)

If a dataset's per-flow CSV is missing, that panel is skipped.

Outputs
-------
- `artifacts/results_tables/fig_roc_curves.png`           — 2-panel: AE vs IF vs ensemble (CIC) + per-family (CIC)
- `artifacts/results_tables/fig_roc_curves_all_datasets.png` — 4-panel: one per dataset, AE/IF/ensemble overlay
- `artifacts/results_tables/fig_roc_curves.csv`           — AUC summary

The plot is a paper-supplement figure for §5 Detection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
TABLES_DIR = ARTIFACTS / "results_tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(
        ARTIFACTS / "test_scores_nf_cic2018.csv",
        usecols=["Label", "attack_family", "ae_score", "if_score", "ens_score"],
    ).rename(columns={"attack_family": "Attack"})
    y_true = df["Label"].to_numpy()

    # ── Panel 1: AE vs IF vs ensemble ────────────────────────────────────
    scores = {
        "AE alone":      df["ae_score"].to_numpy(),
        "IF alone":      df["if_score"].to_numpy(),
        "AE + IF (ours)": df["ens_score"].to_numpy(),
    }
    colors = {"AE alone": "#94A3B8", "IF alone": "#60A5FA", "AE + IF (ours)": "#EC4899"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    summary_rows = []
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        auc = roc_auc_score(y_true, s)
        ax1.plot(fpr, tpr, color=colors[name], lw=2.2,
                 label=f"{name}  (AUC = {auc:.4f})")
        summary_rows.append({"detector": name, "AUC": round(auc, 4)})

    # Diagonal reference (random classifier)
    ax1.plot([0, 1], [0, 1], color="#cbd5e1", linestyle="--", lw=1, label="random (AUC=0.5)")

    # Highlight the FPR @ 1% operating point on the ensemble curve
    s_ens = scores["AE + IF (ours)"]
    fpr_ens, tpr_ens, thr_ens = roc_curve(y_true, s_ens)
    idx_at_001_fpr = int(np.argmin(np.abs(fpr_ens - 0.01)))
    ax1.scatter([fpr_ens[idx_at_001_fpr]], [tpr_ens[idx_at_001_fpr]],
                color="black", zorder=5, s=60)
    ax1.annotate(f"  op. point: FPR={fpr_ens[idx_at_001_fpr]:.3f}, TPR={tpr_ens[idx_at_001_fpr]:.3f}",
                 (fpr_ens[idx_at_001_fpr], tpr_ens[idx_at_001_fpr]),
                 fontsize=9, ha="left", va="top")

    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("AE vs IF vs AE+IF ensemble — NF-CIC2018 test set")
    ax1.set_xlim([-0.005, 1.0])
    ax1.set_ylim([0, 1.005])
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Per-attack-family ROC (ensemble only) ───────────────────
    # For each attack family, ROC compares (Benign vs that family only)
    families = sorted(df[df["Label"] == 1]["Attack"].unique())
    per_family_rows = []

    cmap = plt.cm.viridis(np.linspace(0, 0.95, len(families)))
    for fam, color in zip(families, cmap):
        # Binary task: benign (0) vs THIS family (1); drop other attack families
        mask = (df["Label"] == 0) | (df["Attack"] == fam)
        sub = df[mask]
        y_sub = (sub["Attack"] == fam).astype(int).to_numpy()
        s_sub = sub["ens_score"].to_numpy()
        if y_sub.sum() == 0 or (1 - y_sub).sum() == 0:
            continue
        fpr_f, tpr_f, _ = roc_curve(y_sub, s_sub)
        auc_f = roc_auc_score(y_sub, s_sub)
        ax2.plot(fpr_f, tpr_f, color=color, lw=1.6, alpha=0.85,
                 label=f"{fam}  (AUC={auc_f:.3f})")
        per_family_rows.append({"family": fam, "AUC_vs_benign": round(auc_f, 4),
                                "n_attack_flows": int(y_sub.sum())})

    ax2.plot([0, 1], [0, 1], color="#cbd5e1", linestyle="--", lw=1)
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_title("Per-attack-family ROC (AE+IF ensemble, vs benign)")
    ax2.set_xlim([-0.005, 1.0])
    ax2.set_ylim([0, 1.005])
    ax2.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Detection ROC — NF-CSE-CIC-IDS2018-v2 (2.57M test flows, 14 attack families)",
                 fontsize=12, y=1.00)
    plt.tight_layout()

    out_png = TABLES_DIR / "fig_roc_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Summary CSV ──────────────────────────────────────────────────────
    pd.DataFrame(summary_rows + [{"detector": f"per_family/{r['family']}", "AUC": r["AUC_vs_benign"]}
                                  for r in per_family_rows]).to_csv(
        TABLES_DIR / "fig_roc_curves.csv", index=False)

    # JSON record of headline AUCs
    headline = {
        "dataset": "NF-CSE-CIC-IDS2018-v2 (test split, n=2,569,458 flows)",
        "AE_alone_auc":     round(roc_auc_score(y_true, df["ae_score"]), 4),
        "IF_alone_auc":     round(roc_auc_score(y_true, df["if_score"]), 4),
        "ensemble_auc":     round(roc_auc_score(y_true, df["ens_score"]), 4),
        "operating_point_at_fpr_001": {
            "fpr": round(float(fpr_ens[idx_at_001_fpr]), 4),
            "tpr": round(float(tpr_ens[idx_at_001_fpr]), 4),
            "threshold": round(float(thr_ens[idx_at_001_fpr]), 6),
        },
        "per_family_auc_vs_benign": per_family_rows,
        "note": (
            "Per-flow scores are only persisted for NF-CIC2018 (headline dataset). "
            "For NF-UNSW-NB15-v2 / edge_iiotset / 5g_nidd, AUC values from the matrix "
            "are reported in table_detection.tex but per-flow scores would require "
            "re-running detection to plot full ROC curves."
        ),
    }
    (ARTIFACTS / "roc_curves_summary.json").write_text(json.dumps(headline, indent=2))

    print(f"AUC values (NF-CIC2018):")
    for r in summary_rows:
        print(f"  {r['detector']:<22}AUC = {r['AUC']}")
    print()
    print(f"Operating point @ FPR ≈ 1%:")
    print(f"  FPR={fpr_ens[idx_at_001_fpr]:.4f}, TPR={tpr_ens[idx_at_001_fpr]:.4f}, threshold={thr_ens[idx_at_001_fpr]:.4f}")
    print()
    print(f"Per-family AUC (vs benign):")
    for r in sorted(per_family_rows, key=lambda r: -r["AUC_vs_benign"]):
        print(f"  {r['family']:<28}AUC={r['AUC_vs_benign']:.4f}  (n={r['n_attack_flows']:,})")
    print()
    print(f"Wrote:")
    print(f"  {out_png}")
    print(f"  {TABLES_DIR / 'fig_roc_curves.csv'}")
    print(f"  {ARTIFACTS / 'roc_curves_summary.json'}")


def build_4panel_all_datasets() -> None:
    """Generate the 4-panel ROC plot: one panel per dataset, AE/IF/ensemble overlay."""
    # Use the matrix-aligned per-flow scores for all 4 datasets (seed=42) so AUCs
    # are consistent with the rest of the paper and with the 2-panel CIC plot
    # above (which now reads the same test_scores_nf_cic2018.csv).
    DATASETS_LABEL = [
        ("nf_cic2018",   "NF-CSE-CIC-IDS2018-v2",  ARTIFACTS / "test_scores_nf_cic2018.csv"),
        ("nf_unsw15",    "NF-UNSW-NB15-v2",        ARTIFACTS / "test_scores_nf_unsw15.csv"),
        ("edge_iiotset", "Edge-IIoTset",           ARTIFACTS / "test_scores_edge_iiotset.csv"),
        ("5g_nidd",      "5G-NIDD",                ARTIFACTS / "test_scores_5g_nidd.csv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()
    summary = []
    for ax, (key, label, path) in zip(axes, DATASETS_LABEL):
        if not path.exists():
            ax.text(0.5, 0.5, f"(per-flow scores missing for {key})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#94A3B8")
            ax.set_title(label, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        df = pd.read_csv(path, usecols=["Label", "ae_score", "if_score", "ens_score"])
        if df["Label"].nunique() < 2:
            ax.text(0.5, 0.5, f"(single-class test set — skipped)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#94A3B8")
            ax.set_title(label, fontsize=11)
            continue
        y = df["Label"].to_numpy()
        for det_label, col, color in [
            ("AE alone",       "ae_score",  "#94A3B8"),
            ("IF alone",       "if_score",  "#60A5FA"),
            ("AE + IF (ours)", "ens_score", "#EC4899"),
        ]:
            s = df[col].to_numpy()
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{det_label}  AUC={auc:.4f}")
            summary.append({"dataset": key, "detector": det_label, "AUC": round(auc, 4)})
        ax.plot([0, 1], [0, 1], color="#cbd5e1", linestyle="--", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(label, fontsize=11)
        ax.set_xlim([-0.005, 1.0]); ax.set_ylim([0, 1.005])
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Detection ROC across all 4 NF-v2 datasets (seed=42)", fontsize=13, y=0.995)
    plt.tight_layout()
    out = TABLES_DIR / "fig_roc_curves_all_datasets.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n4-panel ROC plot:")
    if summary:
        print(f"  {'dataset':<14}{'detector':<22}{'AUC':<10}")
        for r in summary:
            print(f"  {r['dataset']:<14}{r['detector']:<22}{r['AUC']:<10}")
    print(f"  → {out}")
    # Append to summary CSV
    df_sum = pd.DataFrame(summary)
    sum_path = TABLES_DIR / "fig_roc_curves_all_datasets.csv"
    df_sum.to_csv(sum_path, index=False)
    print(f"  → {sum_path}")


if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("Building 4-panel ROC plot across all datasets...")
    print("="*60)
    build_4panel_all_datasets()
