"""Generate Q1-paper LaTeX tables from frozen artifacts (Task 4).

Outputs in `artifacts/results_tables/`:
- table_detection.tex     — Detection backbone comparison (Anomal-E, Butt, Mohale, Ours)
- table_causal_shap.tex   — W13 Causal Shapley gates (Module 5a)
- table_counterfactuals.tex — W17 Pareto HV per family (Module 5b)
- table_stl.tex           — W22 STL precision per MITRE technique (Module 4)

All tables use the booktabs convention (\\toprule, \\midrule, \\bottomrule),
caption above, \\label{tab:...}. No external dependencies beyond pandas + json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
TBL = ART / "results_tables"
TBL.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# 4.1 — table_detection.tex
# ──────────────────────────────────────────────────────────────────────────
def _ci(value, lo, hi, digits=4):
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def gen_detection() -> None:
    prim = json.loads((ART / "test_metrics.json").read_text())
    strict = json.loads((ART / "test_metrics_strict.json").read_text())
    prec = json.loads((ART / "test_metrics_precision.json").read_text())
    hi_mf1 = json.loads((ART / "test_metrics_highMF1.json").read_text())
    ens = json.loads((ART / "test_metrics_strict_ensemble.json").read_text())

    rows = [
        # (Method, Dataset, MF1 (CI), AUC-ROC, FPR, Causal XAI?)
        ("Butt et al.\\ 2026 -- BERT", "NF-CSE-CIC", "0.883", "--", "--", "No"),
        ("Mohale \\& Obagbuwa 2025 -- CatBoost$^{\\ast}$", "UNSW-NB15", "0.870", "0.940", "0.070", "No"),
        ("Anomal-E (Caville et al.\\ 2022)", "NF-CSE-CIC, 4\\% contam.", "0.9438", "--", "--", "No"),
        ("Anomal-E (Caville et al.\\ 2022)", "NF-CSE-CIC, 0\\% contam.", "0.9539", "--", "--", "No"),
        (
            "Ours -- primary (FPR$\\leq$0.03)",
            "NF-CSE-CIC",
            _ci(prim["macro_f1_mean"], prim["macro_f1_lo"], prim["macro_f1_hi"]),
            _ci(prim["auc_roc_mean"], prim["auc_roc_lo"], prim["auc_roc_hi"]),
            _ci(prim["fpr_mean"], prim["fpr_lo"], prim["fpr_hi"]),
            "\\textbf{Yes}",
        ),
        (
            "Ours -- strict, 3-seed ensemble (FPR$\\leq$0.01) [headline]",
            "NF-CSE-CIC",
            f"\\textbf{{{ens['macro_f1']:.4f}}}",
            f"{ens['auc_roc']:.4f}",
            f"{ens['fpr']:.4f}",
            "\\textbf{Yes}",
        ),
        (
            "Ours -- strict, single-run (FPR$\\leq$0.01)",
            "NF-CSE-CIC",
            _ci(strict["macro_f1_mean"], strict["macro_f1_lo"], strict["macro_f1_hi"]),
            _ci(strict["auc_roc_mean"], strict["auc_roc_lo"], strict["auc_roc_hi"]),
            _ci(strict["fpr_mean"], strict["fpr_lo"], strict["fpr_hi"]),
            "\\textbf{Yes}",
        ),
        (
            "Ours -- precision-controlled ($\\geq$0.85)",
            "NF-CSE-CIC",
            _ci(prec["macro_f1_mean"], prec["macro_f1_lo"], prec["macro_f1_hi"]),
            _ci(prec["auc_roc_mean"], prec["auc_roc_lo"], prec["auc_roc_hi"]),
            _ci(prec["fpr_mean"], prec["fpr_lo"], prec["fpr_hi"]),
            "\\textbf{Yes}",
        ),
        (
            "Ours -- MaxF1 supplementary (test-oracle)",
            "NF-CSE-CIC",
            _ci(hi_mf1["macro_f1_mean"], hi_mf1["macro_f1_lo"], hi_mf1["macro_f1_hi"]),
            _ci(hi_mf1["auc_roc_mean"], hi_mf1["auc_roc_lo"], hi_mf1["auc_roc_hi"]),
            _ci(hi_mf1["fpr_mean"], hi_mf1["fpr_lo"], hi_mf1["fpr_hi"]),
            "\\textbf{Yes}",
        ),
    ]

    body = "\n".join(
        f"{m} & {d} & {mf1} & {auc} & {fpr} & {xai} \\\\"
        for (m, d, mf1, auc, fpr, xai) in rows
    )

    tex = (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\caption{Detection performance on NF-CSE-CIC-IDS2018-V2 (NetFlow v2). "
        "Macro-F1 reported with bootstrap 95\\% CIs ($n{=}1000$ resamples) where "
        "available. Anomal-E numbers source-verified from Caville et al.\\ 2022. "
        "$^{\\ast}$ Different dataset (UNSW-NB15). At the strict FPR$\\leq$0.01 "
        "operating point we report two rows: the \\emph{3-seed ensemble} is the "
        "deployment headline (point estimate of averaged per-seed scores; depicted "
        "in Fig.~\\ref{fig:confusion_matrix}), and the \\emph{single-run} row is the "
        "single-seed baseline at the same FPR cap. The MaxF1 supplementary "
        "row is the unconstrained operating point obtained on the same frozen "
        "AE+IF detector; the FPR cap of the primary row does not clip its "
        "macro-F1 (see Section~5).}\n"
        "\\label{tab:detection}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\small\n"
        "\\begin{tabular}{llllll}\n"
        "\\toprule\n"
        "Method & Dataset & Macro-F1 (95\\% CI) & AUC-ROC (95\\% CI) & FPR (95\\% CI) & Causal XAI \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    (TBL / "table_detection.tex").write_text(tex)
    print(f"  wrote {TBL/'table_detection.tex'}")


# ──────────────────────────────────────────────────────────────────────────
# 4.2 — table_causal_shap.tex
# ──────────────────────────────────────────────────────────────────────────
def _tex_escape(s: str) -> str:
    return (
        s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
        .replace("$", "\\$").replace("{", "\\{").replace("}", "\\}")
    )


def gen_causal_shap() -> None:
    gates = pd.read_csv(ART / "module5a_gate_summary.csv")
    suff = pd.read_csv(ART / "module5a_faithfulness_smoke.csv")

    pretty = {
        "wall_clock_lt_2s": "Wall clock (s)",
        "efficiency_relative_gap": "Efficiency rel.\\ gap",
        "protocol_semantic_top_feature": "Protocol-semantic top feature",
        "cc_shapley_synthetic_collider_change": "CC-Shapley collider change",
        "sensitivity_flip_5": "Sensitivity $\\rho$ (flip-5)",
        "sensitivity_remove_5": "Sensitivity $\\rho$ (remove-5)",
        "sensitivity_pc_replace": "Sensitivity $\\rho$ (pc-replace)",
        "eraser_sufficiency_vs_kernelshap": "ERASER sufficiency vs KernelSHAP",
        "lipschitz_vs_kernelshap": "Lipschitz improvement vs KernelSHAP",
        "r_package_validation": "R \\texttt{shapr} validation (MAPE)",
    }

    def _fmt(v):
        if pd.isna(v):
            return "--"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return _tex_escape(str(v))

    body_lines: list[str] = []
    for _, row in gates.iterrows():
        crit = pretty.get(row["criterion"], _tex_escape(str(row["criterion"])))
        v_smoke = _fmt(row["value"])
        v_500 = _fmt(row["value_500flows"]) if "value_500flows" in row else "--"
        target = _tex_escape(str(row["target"]))
        # Status comes from value_500flows for sensitivity rows; otherwise the
        # value column.
        if "sensitivity" in str(row["criterion"]):
            status = str(row["status_500flows"]) if "status_500flows" in row and not pd.isna(row["status_500flows"]) else str(row["status"])
        else:
            status = str(row["status"])
        body_lines.append(
            f"{crit} & {target} & {v_smoke} & {v_500} & {_tex_escape(status)} \\\\"
        )

    # Append summary rows
    body_lines.append("\\midrule")
    body_lines.append("\\multicolumn{5}{l}{\\textit{Per-family ERASER sufficiency (causal Shapley vs KernelSHAP):}} \\\\")
    for _, r in suff.iterrows():
        fam = _tex_escape(str(r["attack_family"]))
        c = float(r["causal_sufficiency"])
        v = float(r["vanilla_sufficiency"])
        win = "\\checkmark" if bool(r["causal_beats_vanilla"]) else "--"
        body_lines.append(
            f"\\quad {fam} & causal / vanilla & {c:.4f} & {v:.4f} & {win} \\\\"
        )

    tex = (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\caption{Causal Shapley quality gates (Module~5a, W13). The 500-flow "
        "column reports the paper-grade run (Gap~4-A; cf.\\ Section~6). "
        "Lipschitz and ERASER sufficiency are reported as improvement over "
        "vanilla KernelSHAP on identical flow sets and identical empty-DAG "
        "baseline. The per-family sufficiency rows (causal/vanilla) document "
        "the KernelSHAP collapse on LOIC-HTTP "
        "(see \\texttt{artifacts/module5a\\_sufficiency\\_note.md}).}\n"
        "\\label{tab:causal_shap}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\small\n"
        "\\begin{tabular}{lllll}\n"
        "\\toprule\n"
        "Criterion & Threshold & 20-flow (smoke) & 500-flow (paper) & Status \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    (TBL / "table_causal_shap.tex").write_text(tex)
    print(f"  wrote {TBL/'table_causal_shap.tex'}")


# ──────────────────────────────────────────────────────────────────────────
# 4.3 — table_counterfactuals.tex
# ──────────────────────────────────────────────────────────────────────────
def gen_counterfactuals() -> None:
    df = pd.read_csv(ART / "module5b_w17_gate.csv")
    meta = json.loads((ART / "module5b_w17_gate.json").read_text())

    body_lines: list[str] = []
    for _, r in df.iterrows():
        fam = _tex_escape(str(r["family"]))
        n = int(r["n_flows"])
        hn = float(r["hv_nsga_mean"])
        hd = float(r["hv_dice_mean"])
        wins = int(r["nsga_wins_per_flow"])
        won = "\\textbf{WON}" if bool(r["family_won"]) else "lost"
        body_lines.append(
            f"{fam} & {n} & {hn:.2f} & {hd:.2f} & {wins}/{n} & {won} \\\\"
        )

    body_lines.append("\\midrule")
    body_lines.append(
        f"\\multicolumn{{6}}{{l}}{{\\textbf{{Families won: "
        f"{meta['families_won']} / {len(meta['families'])} "
        f"(gate target: $\\geq$ {meta['families_required']}) "
        f"\\textit{{ -- {meta['status']}}}}}}} \\\\"
    )

    tex = (
        "\\begin{table}[!t]\n"
        "\\centering\n"
        "\\caption{Multi-objective counterfactual quality on "
        "NF-CSE-CIC-IDS2018-V2 (Module~5b, W17 gate, Gap~5-B). "
        "Hypervolume (HV) is computed over (validity, proximity, sparsity, "
        "plausibility, actionability) using a joint per-flow reference "
        f"point. NSGA-II vs scalarised DiCE on identical candidate pool "
        f"({meta['flows_per_family']} top-AE-score flows per family from a "
        f"{meta['candidate_pool']}-row attack-test pool, "
        f"{meta['dice_total_cfs']} DiCE candidates per flow, "
        f"seed~{meta['seed']}, "
        f"elapsed~{meta['elapsed_seconds']:.1f}s).\n"
        "Galwaduge \\& Samarabandu 2025 (TabDiff) comparison deferred to "
        "cross-dataset evaluation (Limitation~X-1).}\n"
        "\\label{tab:counterfactuals}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrl}\n"
        "\\toprule\n"
        "Attack family & $n$ flows & HV (NSGA-II) & HV (DiCE) & Per-flow wins & Result \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    (TBL / "table_counterfactuals.tex").write_text(tex)
    print(f"  wrote {TBL/'table_counterfactuals.tex'}")


# ──────────────────────────────────────────────────────────────────────────
# 4.4 — table_stl.tex
# ──────────────────────────────────────────────────────────────────────────
def gen_stl() -> None:
    df = pd.read_csv(ART / "stl_technique_status.csv")

    body_lines: list[str] = []
    footnote_rows: list[str] = []
    for _, r in df.iterrows():
        mid = _tex_escape(str(r["mitre_id"]))
        name = _tex_escape(str(r["name"]))
        prec = float(r["precision"])
        rec = float(r["recall"])
        tp = int(r["tp"])
        fp = int(r["fp"])
        fn = int(r["fn"])
        method = str(r["method"])
        prec_str = f"{prec:.3f}"
        is_stl_pass = (method == "STL") and (prec >= 0.95)
        if is_stl_pass:
            prec_str = f"\\textbf{{{prec_str}}}"
        method_str = _tex_escape(method)
        if "IF-THEN" in method:
            method_str = method_str + "$^{a}$"
            footnote_rows.append(str(r["rationale"]))
        body_lines.append(
            f"{mid} & {name} & {prec_str} & {rec:.3f} & {tp} & {fp} & {fn} & {method_str} \\\\"
        )

    notes_block = ""
    if footnote_rows:
        notes_block = (
            "\\\\\n"
            "\\footnotesize $^{a}$ "
            + "; ".join(_tex_escape(s) for s in footnote_rows)
            + "\n"
        )

    tex = (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\caption{STL technique-level precision and recall on the "
        "NF-CSE-CIC-IDS2018-V2 test split (Module~4, W22 gate). "
        "Precision $\\geq$ 0.95 enforced for STL adoption; T1046 falls back "
        "to typed IF-THEN due to dataset coverage (no pure scan label, "
        "window-level uniqueness aggregation absent). Bold values are STL "
        "techniques meeting the adoption threshold.}\n"
        "\\label{tab:stl}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\small\n"
        "\\begin{tabular}{llrrrrrl}\n"
        "\\toprule\n"
        "MITRE ID & Technique & Precision & Recall & TP & FP & FN & Method \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        + notes_block
        + "\\end{table*}\n"
    )
    (TBL / "table_stl.tex").write_text(tex)
    print(f"  wrote {TBL/'table_stl.tex'}")


def main() -> None:
    print("[generate_paper_tables] writing 4 LaTeX tables ...")
    gen_detection()
    gen_causal_shap()
    gen_counterfactuals()
    gen_stl()
    print("[generate_paper_tables] done.")


if __name__ == "__main__":
    main()
