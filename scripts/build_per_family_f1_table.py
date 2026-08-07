"""Per-attack-family F1 breakdown table.

For each (dataset, attack_family) pair, compute F1 at two operating points:
  - **Strict (FPR<=0.01) — in-domain 3-seed ensemble** for NF-CSE-CIC and NF-UNSW
  - **MaxF1-on-val ensemble**                          for Edge-IIoTset and 5G-NIDD

Family F1 is computed by binarising attack_family != 'Benign' vs benign baseline,
then comparing to the ensemble's binary prediction at the calibrated threshold.
This is the standard reviewer-asked-for breakdown: "does it detect Bot well? what
about Infiltration? what's the rare-class recall?"

Output: artifacts/results_tables/table_per_family_f1.{csv,tex}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "artifacts"
TBL = ART / "results_tables"


def _per_family_metrics(scores: np.ndarray, families: np.ndarray, labels: np.ndarray, threshold: float) -> list[dict]:
    """For each non-benign family, compute precision/recall/F1 vs the same benign pool.

    A flow from family X is "true positive" if scored >= threshold; the benign pool
    (identified via Label==0, robust to "Benign"/"Normal" naming differences across
    datasets) contributes the FP/TN counts. Each family row sees the same benign FPR.
    """
    benign_mask = labels == 0
    benign_scores = scores[benign_mask]
    n_benign = int(benign_mask.sum())
    fp = int((benign_scores >= threshold).sum())
    fpr = fp / max(n_benign, 1)
    tn = n_benign - fp

    rows = [{
        "attack_family": "Benign (baseline)",
        "n_flows": n_benign,
        "tp": "N/A", "fp": fp, "fn": "N/A", "tn": tn,
        "recall": "N/A", "precision": "N/A", "f1": "N/A",
        "benign_fpr": round(fpr, 4),
    }]
    # Iterate over attack families (anything with at least one Label==1 row)
    attack_mask = labels == 1
    families_unique = sorted(set(families[attack_mask]))
    for fam in families_unique:
        # Constrain to attack rows of this family (avoids counting a "Normal"-named row as attack)
        fam_mask = (families == fam) & attack_mask
        fam_scores = scores[fam_mask]
        n_fam = int(fam_mask.sum())
        if n_fam == 0:
            continue
        tp = int((fam_scores >= threshold).sum())
        fn = n_fam - tp
        # Use this family's TP/FN against the benign pool's FP/TN for precision
        prec = tp / max(tp + fp, 1)
        rec = tp / max(n_fam, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({
            "attack_family": fam,
            "n_flows": n_fam,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "benign_fpr": round(fpr, 4),
        })
    return rows


def _calibrate_fpr(scores, y, target_fpr=0.01):
    order = np.argsort(-scores, kind="mergesort")
    sord = scores[order]; yord = y[order]
    n_neg = int((yord == 0).sum())
    if n_neg == 0: return float(sord[0])
    cum_fp = np.cumsum(yord == 0)
    fpr = cum_fp / n_neg
    valid = np.where(fpr <= target_fpr)[0]
    return float(sord[0]) + 1e-9 if len(valid) == 0 else float(sord[int(valid[-1])])


def _calibrate_maxf1(scores, y):
    order = np.argsort(-scores, kind="mergesort")
    sord = scores[order]; yord = y[order]
    n_pos = int(yord.sum()); n_neg = int(len(yord) - n_pos)
    if n_pos == 0 or n_neg == 0: return float(sord[0])
    block = np.flatnonzero(np.r_[sord[1:] != sord[:-1], True])
    thr = sord[block]
    tp = np.cumsum(yord == 1).astype(np.float64)[block]
    fp = np.cumsum(yord == 0).astype(np.float64)[block]
    fn = n_pos - tp; tn = n_neg - fp
    tpr = tp / n_pos
    pre_a = tp / np.maximum(tp + fp, 1e-9); f1_a = 2 * pre_a * tpr / np.maximum(pre_a + tpr, 1e-9)
    pre_b = tn / np.maximum(tn + fn, 1e-9); rec_b = tn / n_neg
    f1_b = 2 * pre_b * rec_b / np.maximum(pre_b + rec_b, 1e-9)
    return float(thr[int((0.5 * (f1_a + f1_b)).argmax())])


def main():
    DATASETS = [
        ("nf_cic2018",   "NF-CSE-CIC-IDS2018-v2", "fpr_le_001"),
        ("nf_unsw15",    "NF-UNSW-NB15-v2",       "fpr_le_001"),
        ("edge_iiotset", "Edge-IIoTset",          "maxf1"),
        ("5g_nidd",      "5G-NIDD",               "maxf1"),
    ]

    all_rows = []
    for ds_key, ds_label, strategy in DATASETS:
        scores_path = ART / f"test_scores_{ds_key}.csv"
        if not scores_path.exists():
            print(f"  SKIP {ds_label}: {scores_path} missing"); continue
        df = pd.read_csv(scores_path)
        # Threshold from ensemble JSON (matches what the paper reports)
        ens = json.loads((ART / "in_domain_3seed_ensemble.json").read_text())
        ds_result = next((r for r in ens["results"] if r["dataset"] == ds_key), None)
        if ds_result is None or "ensemble" not in ds_result:
            print(f"  SKIP {ds_label}: ensemble result not found"); continue
        thr = float(ds_result["ensemble"]["threshold"])

        scores = df["ens_score"].to_numpy(dtype=np.float64)
        families = df["attack_family"].to_numpy()
        labels = df["Label"].to_numpy(dtype=np.int64)
        print(f"\n{ds_label} (threshold={thr:.4f} via {strategy})")
        rows = _per_family_metrics(scores, families, labels, thr)
        for r in rows:
            r["dataset"] = ds_label
            r["threshold"] = round(thr, 4)
            r["strategy"] = strategy
            all_rows.append(r)
            tag = "" if r["attack_family"].startswith("Benign") else f"  F1={r['f1']}  rec={r['recall']}  prec={r['precision']}"
            print(f"  {r['attack_family']:30s} n={r['n_flows']:>8}{tag}")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(TBL / "table_per_family_f1.csv", index=False)
    print(f"\nWrote {TBL/'table_per_family_f1.csv'}  ({len(out_df)} rows)")

    # ── LaTeX (compact per-dataset blocks) ────────────────────────────────────
    blocks = []
    for ds_key, ds_label, _ in DATASETS:
        ds_rows = [r for r in all_rows if r["dataset"] == ds_label]
        if not ds_rows: continue
        ben = ds_rows[0]
        attack_rows = ds_rows[1:]
        body = "\n".join([
            f"{r['attack_family']} & {r['n_flows']:,} & {r['recall']:.4f} & "
            f"{r['precision']:.4f} & \\textbf{{{r['f1']:.4f}}} \\\\"
            for r in attack_rows
        ])
        blocks.append(
            f"\\textbf{{{ds_label}}} "
            f"(threshold {ben['threshold']:.4f}, benign FPR {ben['benign_fpr']:.4f})\\\\\n"
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Attack family & $n$ test flows & Recall & Precision & F1 \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}"
        )

    tex = (
        "\\begin{table*}[!t]\n\\centering\n\\small\n"
        "\\caption{Per-attack-family detection breakdown at the in-domain 3-seed ensemble "
        "operating point (NF-CSE-CIC / NF-UNSW: FPR$\\leq$0.01 strict; Edge-IIoTset / 5G-NIDD: "
        "MaxF1-on-val per the per-dataset adaptive strategy --- attack rates 85\\%\\ / 61\\%\\ "
        "make FPR$\\leq$0.01 over-restrictive). The benign pool is shared across families within "
        "each dataset (same FPR baseline).}\n"
        "\\label{tab:per_family_f1}\n"
        + "\n\\vspace{1ex}\n".join(blocks) + "\n"
        "\\end{table*}\n"
    )
    (TBL / "table_per_family_f1.tex").write_text(tex)
    print(f"Wrote {TBL/'table_per_family_f1.tex'}")


if __name__ == "__main__":
    main()
