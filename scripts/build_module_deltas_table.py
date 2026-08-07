"""Build the per-config means + per-module deltas table for the paper.

Each row is one ablation config (A0..A5). Columns are every metric that
varies across configs. The Δ row pairs (A0→A1, A1→A2, …) underneath each
config row show what that specific module *adds* (positive = improvement
where applicable; for Lipschitz lower is better so we report -Δ%).

Outputs
-------
- `artifacts/results_tables/table_module_deltas.{tex,csv}` — paper-ready
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "artifacts" / "results"
TABLES_DIR = ROOT / "artifacts" / "results_tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_ORDER = ["A0_baseline", "A1_stl_only", "A2_causal_shap",
                "A3_moocf", "A4_full", "A5_random_dag"]

MODULE_ADDED = {
    "A0_baseline":    "(baseline — detection only)",
    "A1_stl_only":    "+ STL typing",
    "A2_causal_shap": "+ Causal Shapley",
    "A3_moocf":       "+ NSGA-II multi-obj CFs",
    "A4_full":        "+ Concept abduction (full pipeline)",
    "A5_random_dag":  "(negative control: random DAG replaces NF-DAG-v1)",
}


def _mean(vals):
    finite = [v for v in vals if v is not None]
    return st.mean(finite) if finite else None


def main() -> None:
    # ── Aggregate all 72 result.json by config ───────────────────────────
    agg = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob(str(RESULTS / "*" / "*" / "*" / "result.json"))):
        r = json.loads(Path(f).read_text())
        cfg = r["config_name"]
        d = r["detection"]; cm = r.get("cf_metrics") or {}
        fa = r.get("faithfulness") or {}; cc = r.get("concept_metrics") or {}
        agg[cfg]["macro_f1"].append(d["macro_f1"][0])
        agg[cfg]["auc"].append(d["auc_roc"][0])
        agg[cfg]["fpr"].append(d["fpr"][0])
        agg[cfg]["cf_validity"].append(cm.get("validity"))
        agg[cfg]["cf_feasibility"].append(cm.get("feasibility_rate"))
        agg[cfg]["cf_proximity"].append(cm.get("proximity"))
        agg[cfg]["cf_sparsity"].append(cm.get("sparsity"))
        agg[cfg]["cf_hypervolume"].append(cm.get("hypervolume"))
        agg[cfg]["sufficiency"].append(fa.get("sufficiency"))
        agg[cfg]["comprehensiveness"].append(fa.get("comprehensiveness"))
        agg[cfg]["lipschitz"].append(fa.get("lipschitz"))
        agg[cfg]["concept_fidelity"].append(cc.get("mean_fidelity"))

    # ── Per-config means ─────────────────────────────────────────────────
    METRICS = [
        ("macro_f1",          4),  # column name, decimal places
        ("auc",               4),
        ("fpr",               4),
        ("cf_validity",       3),
        ("cf_feasibility",    4),
        ("cf_proximity",      3),
        ("cf_sparsity",       2),
        ("cf_hypervolume",    1),
        ("sufficiency",       4),
        ("comprehensiveness", 4),
        ("lipschitz",         4),
        ("concept_fidelity",  4),
    ]
    rows = []
    for cfg in CONFIG_ORDER:
        row = {"method": cfg, "added": MODULE_ADDED[cfg]}
        for name, dp in METRICS:
            v = _mean(agg[cfg][name])
            row[name] = round(v, dp) if v is not None else None
        rows.append(row)

    # ── Δ rows: change vs immediately previous config ───────────────────
    delta_rows = []
    for i in range(1, len(CONFIG_ORDER)):
        cur, prev = CONFIG_ORDER[i], CONFIG_ORDER[i-1]
        # For A5: compare vs A4 (it's the negative control on A4, not a cumulative add)
        ref = "A4_full" if cur == "A5_random_dag" else prev
        d = {"method": f"Δ ({ref}→{cur})", "added": "[module impact]"}
        for name, dp in METRICS:
            v_cur = _mean(agg[cur][name])
            v_ref = _mean(agg[ref][name])
            if v_cur is None or v_ref is None:
                d[name] = None
            else:
                d[name] = round(v_cur - v_ref, dp)
        delta_rows.append(d)

    # Interleave config + delta rows for readability
    interleaved = [rows[0]]
    for i in range(1, len(rows)):
        interleaved.append(rows[i])
        interleaved.append(delta_rows[i-1])

    # ── Write CSV ────────────────────────────────────────────────────────
    csv_path = TABLES_DIR / "table_module_deltas.csv"
    with csv_path.open("w", newline="") as f:
        cols = ["method", "added"] + [m for m, _ in METRICS]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in interleaved:
            w.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in cols})

    # ── Write LaTeX ──────────────────────────────────────────────────────
    # Custom LaTeX since the shared writer doesn't handle interleaved delta rows
    tex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{Per-module ablation deltas. Each $A_i$ row is the mean across "
        "4 datasets $\\times$ 3 seeds = 12 cells. Each $\\Delta$ row reports the change "
        "from the previous config (for A5: change vs A4, since A5 is the negative "
        "control replacing the expert NF-DAG-v1 with a random DAG, not a cumulative add). "
        "Detection metrics (Macro-F1, AUC, FPR) are config-invariant by design because "
        "the AE+IF detector is trained once per (dataset, seed) and reused across configs.}",
        "\\label{tab:module_deltas}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{ll" + "r"*len(METRICS) + "}",
        "\\toprule",
        "Config & Module added & "
        + " & ".join(name.replace("_", "\\_") for name, _ in METRICS) + " \\\\",
        "\\midrule",
    ]
    for r in interleaved:
        method = r["method"].replace("_", "\\_")
        added = r["added"].replace("_", "\\_").replace("→", "$\\to$")
        cells = []
        for name, dp in METRICS:
            v = r.get(name)
            if v is None:
                cells.append("---")
            elif isinstance(r["method"], str) and r["method"].startswith("Δ"):
                # Format deltas with explicit sign
                sign = "+" if v > 0 else ""
                cells.append(f"{sign}{v:.{dp}f}")
            else:
                cells.append(f"{v:.{dp}f}")
        tex_lines.append(f"{method} & {added} & " + " & ".join(cells) + " \\\\")
        if r["method"].startswith("Δ"):
            tex_lines.append("\\midrule")
    if tex_lines[-1] == "\\midrule":
        tex_lines.pop()  # don't end with midrule
    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ]
    (TABLES_DIR / "table_module_deltas.tex").write_text("\n".join(tex_lines))

    # ── Print summary to console ─────────────────────────────────────────
    print(f"{'method':<24}{'cf_feas':<10}{'cf_prox':<10}{'lip':<10}{'concept':<10}{'sparsity':<11}{'HV':<10}")
    print("-"*85)
    for r in interleaved:
        method = r["method"][:23]
        cf_f = r.get("cf_feasibility")
        cf_p = r.get("cf_proximity")
        lip = r.get("lipschitz")
        cf = r.get("concept_fidelity")
        sp = r.get("cf_sparsity")
        hv = r.get("cf_hypervolume")
        is_delta = method.startswith("Δ")
        def f(v, dp=4):
            if v is None: return "—"
            sign = "+" if (is_delta and v > 0) else ""
            return f"{sign}{v:.{dp}f}"
        print(f"{method:<24}{f(cf_f):<10}{f(cf_p, 3):<10}{f(lip):<10}{f(cf):<10}{f(sp, 2):<11}{f(hv, 1):<10}")

    print()
    print(f"Wrote:")
    print(f"  {TABLES_DIR / 'table_module_deltas.csv'}")
    print(f"  {TABLES_DIR / 'table_module_deltas.tex'}")


if __name__ == "__main__":
    main()
