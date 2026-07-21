"""
Path 1: generate fig_causal_shapley_direct_indirect.png from the frozen
causal_shapley_top_features.csv (single representative FTP-Hulk flow).

Usage:
    python scripts/gen_fig_causal_shapley.py

Output:
    artifacts/fig_causal_shapley_direct_indirect.png   (300 dpi, ~3.5 in wide)
"""
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "artifacts" / "causal_shapley_top_features.csv"
OUT_PNG = ROOT / "artifacts" / "fig_causal_shapley_direct_indirect.png"

TOP_K = 10
COLOUR_DIRECT = "#2166ac"    # blue  — direct causal effect
COLOUR_INDIRECT = "#f4a582"  # coral — indirect effect through DAG descendants
COLOUR_NEG = "#4dac26"       # green — negative (protective) attribution


def _shorten(name: str) -> str:
    return (name
            .replace("RETRANSMITTED_", "RETRANS_")
            .replace("NUM_PKTS_", "N_PKTS_")
            .replace("_TO_", "→")
            .replace("DST", "DST")
            .replace("SRC", "SRC")
            .replace("_AVG_THROUGHPUT", "_AVG_TP")
            .replace("_SECOND_BYTES", "_SEC_B"))


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    top = df.head(TOP_K).copy()

    # Decompose each bar into: direct portion, indirect portion.
    # Convention: phi = direct_portion + indirect_portion
    # where indirect_portion = indirect_effect as recorded,
    # and direct_portion = phi - indirect_effect (can be < |phi| when indirect > 0).
    # For negative-phi features indirect_effect is 0.0 in this artefact.
    top["direct_portion"] = top["phi"] - top["indirect_effect"]
    top["indirect_portion"] = top["indirect_effect"]

    labels = [_shorten(f) for f in top["feature"]]
    y = np.arange(TOP_K)
    # invert so rank-1 is at top
    y_pos = y[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for i, (_, row) in enumerate(top.iterrows()):
        yp = y_pos[i]
        dp = float(row["direct_portion"])
        ip = float(row["indirect_portion"])
        phi = float(row["phi"])

        if phi >= 0:
            # direct segment from 0 → dp
            ax.barh(yp, dp, left=0, height=0.55,
                    color=COLOUR_DIRECT, linewidth=0)
            # indirect segment from dp → phi
            if ip > 1e-6:
                ax.barh(yp, ip, left=dp, height=0.55,
                        color=COLOUR_INDIRECT, linewidth=0)
        else:
            # negative attribution (protective feature)
            ax.barh(yp, phi, left=0, height=0.55,
                    color=COLOUR_NEG, linewidth=0, alpha=0.8)

    ax.axvline(0, color="black", linewidth=0.7, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Causal Shapley value $\varphi_i$")
    ax.set_xlim(-0.32, 0.95)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linewidth=0.4, color="grey", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=COLOUR_DIRECT, label="Direct causal effect"),
        mpatches.Patch(color=COLOUR_INDIRECT, label="Indirect (through DAG)"),
        mpatches.Patch(color=COLOUR_NEG, alpha=0.8, label="Negative attribution"),
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc="lower right",
              frameon=True, edgecolor="grey", framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
