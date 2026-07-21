"""
Path 2: generate fig_causal_shapley_heatmap.png from
artifacts/causal_shapley_per_family.csv (produced by nb04_cell_per_family_shap.py).

Usage:
    python scripts/gen_fig_causal_shapley_heatmap.py

Output:
    artifacts/fig_causal_shapley_heatmap.png          (300 dpi)
    artifacts/fig_causal_shapley_heatmap_indirect.png  (indirect-fraction overlay)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
})

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "artifacts" / "causal_shapley_per_family.csv"
OUT_PNG_ABS = ROOT / "artifacts" / "fig_causal_shapley_heatmap.png"
OUT_PNG_IND = ROOT / "artifacts" / "fig_causal_shapley_heatmap_indirect.png"

TOP_K_FEATURES = 15   # rows in the heatmap


def _shorten(name: str) -> str:
    return (name
            .replace("RETRANSMITTED_", "RETRANS_")
            .replace("NUM_PKTS_", "N_PKTS_")
            .replace("_TO_SRC", "→SRC")
            .replace("_TO_DST", "→DST")
            .replace("_AVG_THROUGHPUT", "_AVG_TP")
            .replace("_SECOND_BYTES", "_SEC_B")
            .replace("SRC_TO_DST", "SRC→DST")
            .replace("DST_TO_SRC", "DST→SRC"))


def _shorten_family(f: str) -> str:
    return (f.replace("-", "\n", 1)       # wrap long names
             .replace("BruteForce", "BF")
             .replace("PortScan", "PortScan")
             .replace("Infilteration", "Infiltr."))


def _build_matrix(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Pivot to (feature × family) matrix for a given metric column."""
    # global top-K features by mean abs_phi across families
    global_rank = (df.groupby("feature")["abs_phi"].mean()
                     .sort_values(ascending=False)
                     .head(TOP_K_FEATURES).index.tolist())
    families = sorted(df["attack_family"].unique())
    pivot = df[df["feature"].isin(global_rank)].pivot_table(
        index="feature", columns="attack_family", values=metric,
        aggfunc="mean", fill_value=0.0)
    # reorder rows by global rank
    pivot = pivot.reindex(global_rank)
    pivot.index = [_shorten(f) for f in pivot.index]
    pivot.columns = [_shorten_family(f) for f in pivot.columns]
    return pivot, global_rank, families


def _plot_heatmap(matrix: pd.DataFrame, title: str, cmap: str,
                  vmin: float, vmax: float, out_path: Path,
                  annotate: bool = True) -> None:
    n_feat = len(matrix.index)
    n_fam = len(matrix.columns)
    figw = max(3.5, 0.55 * n_fam + 1.8)
    figh = max(3.0, 0.35 * n_feat + 0.9)

    fig, ax = plt.subplots(figsize=(figw, figh))
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix.values, cmap=cmap, norm=norm,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(n_fam))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels(matrix.index, fontsize=6.5)

    ax.set_xlabel("Attack family")
    ax.set_ylabel("Feature")
    ax.set_title(title, fontsize=8, pad=4)

    if annotate and n_fam * n_feat <= 200:
        for i in range(n_feat):
            for j in range(n_fam):
                v = matrix.values[i, j]
                if abs(v) > 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=4.5,
                            color="white" if abs(v) > 0.6 * vmax else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=6)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n_fam) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_feat) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found.\n"
            "Run:  python scripts/nb04_cell_per_family_shap.py\n"
            "inside the caushap-nids .venv first."
        )

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows across "
          f"{df['attack_family'].nunique()} families, "
          f"{df['feature'].nunique()} unique features.")

    # ── Figure A: total |φ| magnitude heatmap ─────────────────────────────
    abs_mat, _, _ = _build_matrix(df, "abs_phi")
    vmax_abs = float(abs_mat.values.max())
    _plot_heatmap(
        abs_mat,
        title=r"Mean $|\varphi_i|$ per attack family",
        cmap="YlOrRd",
        vmin=0, vmax=vmax_abs,
        out_path=OUT_PNG_ABS,
    )

    # ── Figure B: indirect fraction heatmap ───────────────────────────────
    # indirect_frac_i = |indirect_effect| / (|direct_effect| + |indirect_effect| + eps)
    df["indirect_frac"] = (df["indirect_effect"].abs() /
                           (df["direct_effect"].abs() + df["indirect_effect"].abs() + 1e-9))
    ind_mat, _, _ = _build_matrix(df, "indirect_frac")
    _plot_heatmap(
        ind_mat,
        title="Indirect-effect fraction (unique to causal Shapley)",
        cmap="PuBuGn",
        vmin=0, vmax=1.0,
        out_path=OUT_PNG_IND,
        annotate=False,
    )


if __name__ == "__main__":
    main()
