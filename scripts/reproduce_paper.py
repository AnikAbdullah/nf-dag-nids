"""Single-command paper reproducibility (CODING_PLAN_FINAL_v2 Week 25 deliverable).

Orchestrates every artifact a Q1 paper submission needs:

  Phase 0 — Matrix:        72-cell ablation matrix (slow; skip if cached)
  Phase 1 — Ensembles:     in-domain 3-seed adaptive-threshold ensemble; XeNIDS DA
  Phase 2 — Faithfulness:  Lipschitz robust recompute (per-cell paired robust median)
  Phase 3 — Sub-modules:   STL gate, DAG sensitivity, Galwaduge comparison, zero-day, adversarial
  Phase 4 — Tables:        every paper table + criteria dashboard

Each phase is skippable via flags and is idempotent (checks for output artifacts first).

Usage:
  python scripts/reproduce_paper.py --table=all          # Run everything
  python scripts/reproduce_paper.py --phase=4            # Just re-generate tables
  python scripts/reproduce_paper.py --table=detection    # Just the detection table
  python scripts/reproduce_paper.py --table=all --force  # Re-run even if artifacts exist
  python scripts/reproduce_paper.py --check              # Verify all artifacts exist
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "artifacts"
TABLES = ART / "results_tables"
SCRIPTS = ROOT / "scripts"


# ── Artifact registry ────────────────────────────────────────────────────────
# Each row: (path, phase, generator, table-name)
ARTIFACT_REGISTRY = [
    # Phase 0: matrix
    (ART / "results" / "A4_full" / "nf_cic2018" / "42" / "result.json", 0, "matrix", None),
    (ART / "results" / "A5_random_dag" / "nf_cic2018" / "44" / "result.json", 0, "matrix", None),
    # Phase 1: ensembles
    (ART / "in_domain_3seed_ensemble.json", 1, "in_domain_3seed_ensemble", None),
    (ART / "module_x1_xenids_unsw.json", 1, "module_x1_xenids", None),
    # Phase 2: faithfulness
    (ART / "lipschitz_robust_recompute.json", 2, "recompute_lipschitz_robust", None),
    # Phase 3: sub-modules
    (ART / "stl_gate_result.json", 3, "stl_gate", None),
    (ART / "dag_summary.json", 3, "dag", None),
    (ART / "module5c_galwaduge_comparison.json", 3, "galwaduge", None),
    (ART / "module_x2_zero_day.json", 3, "zero_day", None),
    (ART / "module_x4_adversarial_stability.json", 3, "adversarial", None),
    # Phase 4: tables (the user-facing paper artifacts)
    (TABLES / "table_detection.tex", 4, "tables", "detection"),
    (TABLES / "table_vs_literature.tex", 4, "tables", "vs_literature"),
    (TABLES / "table_in_domain_ensemble.tex", 4, "tables", "in_domain_ensemble"),
    (TABLES / "table_ablation_pairs.tex", 4, "tables", "ablation_pairs"),
    (TABLES / "table_feasibility_ablation.tex", 4, "tables", "feasibility_ablation"),
    (TABLES / "table_faithfulness.tex", 4, "tables", "faithfulness"),
    (TABLES / "table_cf_metrics.tex", 4, "tables", "cf_metrics"),
    (TABLES / "table_galwaduge_comparison.tex", 4, "tables", "galwaduge"),
    (TABLES / "table_zero_day.tex", 4, "tables", "zero_day"),
    (TABLES / "table_adversarial_stability.tex", 4, "tables", "adversarial_stability"),
    (TABLES / "table_module_deltas.tex", 4, "tables", "module_deltas"),
    (TABLES / "table_campaign_summary.tex", 4, "tables", "campaign_summary"),
    (TABLES / "table_stl.tex", 4, "tables", "stl"),
    (TABLES / "table_causal_shap.tex", 4, "tables", "causal_shap"),
    (TABLES / "table_counterfactuals.tex", 4, "tables", "counterfactuals"),
    (TABLES / "criteria_dashboard.csv", 4, "tables", "criteria_dashboard"),
    (ART / "notebook01_benchmark_verdict.json", 4, "tables", "benchmark_verdict"),
]


def _run(cmd: list[str], description: str, timeout: float | None = None) -> bool:
    print(f"\n  ▶ {description}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, check=True, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        elapsed = time.time() - t0
        print(f"  ✓ done ({elapsed:.1f}s)", flush=True)
        # Show last 3 lines if non-trivial output
        tail = result.stdout.strip().split("\n")[-3:]
        for line in tail:
            print(f"    | {line[:140]}", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ FAILED (exit {e.returncode})", flush=True)
        print(e.stdout[-500:] if e.stdout else "", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT after {timeout}s", flush=True)
        return False


def phase_0_matrix(force: bool = False):
    """Run the 72-cell ablation matrix (skip if all cells exist)."""
    print("\n" + "═" * 70)
    print("PHASE 0 — Matrix (72 cells, ~19 CPU-h if uncached)")
    print("═" * 70)
    from caushap_nids.experiments import CONFIG_NAMES, DATASETS, DEFAULT_SEEDS
    expected = 0; existing = 0
    for cfg in CONFIG_NAMES:
        for ds in DATASETS:
            for seed in DEFAULT_SEEDS:
                expected += 1
                rp = ART / "results" / cfg / ds / str(seed) / "result.json"
                if rp.exists(): existing += 1
    print(f"  Matrix cells on disk: {existing}/{expected}")
    if existing == expected and not force:
        print("  ✓ Matrix already complete — skipping (use --force to re-run)")
        return True
    return _run(
        [sys.executable, "scripts/run_matrix_mac.py", "--jobs", "1"],
        "running 72-cell matrix",
        timeout=86_400,
    )


def phase_1_ensembles(force: bool = False):
    print("\n" + "═" * 70)
    print("PHASE 1 — Ensembles (in-domain 3-seed + XeNIDS DA)")
    print("═" * 70)
    ok = True
    if force or not (ART / "in_domain_3seed_ensemble.json").exists():
        ok &= _run([sys.executable, "scripts/in_domain_3seed_ensemble.py"],
                    "in-domain 3-seed ensemble (adaptive threshold)", timeout=3600)
    else:
        print("  ✓ in_domain_3seed_ensemble.json exists — skipping")
    if force or not (ART / "module_x1_xenids_unsw.json").exists():
        # XeNIDS DA is encoded in NB07 cell 35; the persistent script is regen_x1_targetrefit
        ok &= _run([sys.executable, "scripts/regen_x1_targetrefit.py"],
                    "XeNIDS DA-ensemble (regen target-refit)", timeout=3600)
    else:
        print("  ✓ module_x1_xenids_unsw.json exists — skipping")
    return ok


def phase_2_faithfulness(force: bool = False):
    print("\n" + "═" * 70)
    print("PHASE 2 — Faithfulness (Lipschitz robust recompute)")
    print("═" * 70)
    if not force and (ART / "lipschitz_robust_recompute.json").exists():
        lr = json.loads((ART / "lipschitz_robust_recompute.json").read_text())
        if len(lr) >= 24:
            print(f"  ✓ {len(lr)} robust Lipschitz cells already computed — skipping")
            return True
    return _run([sys.executable, "scripts/recompute_lipschitz_robust.py"],
                 "robust Lipschitz on A1/A2 × 4 datasets × 3 seeds", timeout=10_800)


def phase_3_submodules(force: bool = False):
    print("\n" + "═" * 70)
    print("PHASE 3 — Sub-modules (STL, DAG, Galwaduge, zero-day, adversarial)")
    print("═" * 70)
    ok = True
    # STL/DAG come from notebooks 02/03 which are FROZEN — verify presence only
    for name, path in [("STL gate", ART / "stl_gate_result.json"),
                       ("DAG summary", ART / "dag_summary.json")]:
        if path.exists():
            print(f"  ✓ {name} exists — frozen")
        else:
            print(f"  ✗ {name} MISSING — re-run notebook 0{'2' if 'DAG' in name else '3'}.ipynb")
            ok = False
    # Galwaduge / zero-day / adversarial have dedicated scripts
    for script, label, art in [
        ("build_galwaduge_comparison.py", "Galwaduge 2025 comparison",
         ART / "module5c_galwaduge_comparison.json"),
        ("build_zero_day_table.py",        "Zero-day X-2 evaluation",
         ART / "module_x2_zero_day.json"),
        ("build_adversarial_table.py",     "Adversarial X-4 (W13 robust median)",
         ART / "module_x4_adversarial_stability.json"),
    ]:
        if not force and art.exists():
            print(f"  ✓ {label} exists — skipping (--force to re-run)")
            continue
        ok &= _run([sys.executable, f"scripts/{script}"], label, timeout=600)
    return ok


def phase_4_tables(table_name: str = "all", force: bool = False):
    print("\n" + "═" * 70)
    print(f"PHASE 4 — Tables ({table_name})")
    print("═" * 70)
    ok = True

    # The dedicated table scripts
    if table_name in ("all", "vs_literature", "in_domain_ensemble"):
        ok &= _run([sys.executable, "scripts/in_domain_3seed_ensemble.py"],
                    "in_domain_ensemble + vs_literature update (idempotent)", timeout=3600)
    if table_name in ("all", "module_deltas"):
        ok &= _run([sys.executable, "scripts/build_module_deltas_table.py"],
                    "module deltas", timeout=120)
    if table_name in ("all", "cf_rules"):
        ok &= _run([sys.executable, "scripts/build_cf_rules.py"],
                    "CF rules deployment", timeout=120)
    if table_name in ("all", "roc_curves"):
        ok &= _run([sys.executable, "scripts/build_roc_curves.py"],
                    "ROC curves", timeout=120)
    if table_name in ("all", "campaign_summary"):
        ok &= _run([sys.executable, "scripts/generate_campaign_production.py"],
                    "campaign summary", timeout=300)
    if table_name in ("all", "detection", "causal_shap", "counterfactuals", "stl"):
        ok &= _run([sys.executable, "scripts/generate_paper_tables.py"],
                    "paper tables (detection, causal_shap, CFs, STL)", timeout=120)

    # Criteria dashboard + benchmark verdict (always regenerate, fast)
    if table_name in ("all", "criteria_dashboard", "benchmark_verdict"):
        ok &= _regen_criteria()
        ok &= _regen_benchmark_verdict()
    return ok


def _regen_criteria() -> bool:
    """Regenerate criteria_dashboard.csv (no separate script — runs the eval directly)."""
    print(f"\n  ▶ regenerating criteria_dashboard.csv", flush=True)
    try:
        import polars as pl
        from caushap_nids.experiments import (
            CONFIG_NAMES, DATASETS, DEFAULT_SEEDS, CRITERIA_VERSION, load_all_results,
        )
        from caushap_nids.evaluation import (
            evaluate_ablation_criteria, criteria_to_rows, all_criteria_pass,
        )
        results = load_all_results(ART, criteria_version=CRITERIA_VERSION, n_explain=200)
        expected = len(CONFIG_NAMES) * len(DATASETS) * len(DEFAULT_SEEDS)
        checks = evaluate_ablation_criteria(results, expected_runs=expected, artifact_dir=ART)
        pl.DataFrame(criteria_to_rows(checks)).write_csv(TABLES / "criteria_dashboard.csv")
        n_pass = sum(1 for c in checks if c.passed)
        sym = "✓" if all_criteria_pass(checks) else "✗"
        print(f"  {sym} criteria_dashboard: {n_pass}/{len(checks)} PASS", flush=True)
        for c in checks:
            if not c.passed:
                print(f"    FAIL: {c.check}: {c.value}", flush=True)
        return all_criteria_pass(checks)
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}", flush=True)
        return False


def _regen_benchmark_verdict() -> bool:
    """Regenerate notebook01_benchmark_verdict.json from current artifacts."""
    print(f"\n  ▶ regenerating notebook01_benchmark_verdict.json", flush=True)
    try:
        m = json.loads((ART / "test_metrics.json").read_text())
        strict = json.loads((ART / "test_metrics_strict.json").read_text())
        prec = json.loads((ART / "test_metrics_precision.json").read_text())
        hi = json.loads((ART / "test_metrics_highMF1.json").read_text())
        ens = json.loads((ART / "in_domain_3seed_ensemble.json").read_text())
        x1 = json.loads((ART / "module_x1_xenids_unsw.json").read_text())
        cic_ens = next(r for r in ens["results"] if r["dataset"] == "nf_cic2018")["ensemble"]
        unsw_ens = next(r for r in ens["results"] if r["dataset"] == "nf_unsw15")["ensemble"]
        cic_f1, unsw_f1, unsw_auc = cic_ens["macro_f1"], unsw_ens["macro_f1"], unsw_ens["auc_roc"]
        da_f1, da_auc = x1["da_headline_macro_f1"], x1["da_headline_auc_roc"]
        prec_t, prec_f = prec["attack_precision"], prec["fpr"]

        checks = [
            dict(check="P0 minimum Macro-F1", criterion=">= 0.85", value=m["macro_f1"],
                  passed=m["macro_f1"] >= 0.85, scope="Notebook 01"),
            dict(check="Paper detection target Macro-F1", criterion=">= 0.88", value=m["macro_f1"],
                  passed=m["macro_f1"] >= 0.88, scope="Notebook 01"),
            dict(check="Primary FPR", criterion="<= 0.03", value=m["fpr"],
                  passed=m["fpr"] <= 0.03, scope="Notebook 01"),
            dict(check="AUC-ROC target", criterion=">= 0.95", value=m["auc_roc"],
                  passed=m["auc_roc"] >= 0.95, scope="Notebook 01"),
            dict(check="Strict supplementary FPR", criterion="<= 0.01", value=strict["fpr"],
                  passed=strict["fpr"] <= 0.01, scope="Notebook 01"),
            dict(check="Attack precision target", criterion=">= 0.85", value=prec_t,
                  passed=prec_t >= 0.85, scope="Notebook 01 supplementary"),
            dict(check="Precision-row FPR cap", criterion="<= 0.03", value=prec_f,
                  passed=prec_f <= 0.03, scope="Notebook 01 supplementary"),
            dict(check="Butt et al. 2026 NF-CSE-CIC sanity bar",
                  criterion=">= 0.88 (in-domain 3-seed ensemble)",
                  value=cic_f1, passed=cic_f1 >= 0.88, scope="In-domain ensemble"),
            dict(check="Mohale numeric sanity bar",
                  criterion="MF1 >= 0.87, AUC >= 0.94, FPR < 0.07",
                  value=f"MF1={m['macro_f1']:.4f}; AUC={m['auc_roc']:.4f}; FPR={m['fpr']:.4f}",
                  passed=m["macro_f1"] >= 0.87 and m["auc_roc"] >= 0.94 and m["fpr"] < 0.07,
                  scope="Numeric only; different dataset"),
            dict(check="Anomal-E NF-CSE-CIC 4%",
                  criterion=">= 0.9438 (in-domain 3-seed ensemble)",
                  value=cic_f1, passed=cic_f1 >= 0.9438, scope="In-domain ensemble"),
            dict(check="Anomal-E NF-CSE-CIC 0%",
                  criterion=">= 0.9539 (in-domain 3-seed ensemble)",
                  value=cic_f1, passed=cic_f1 >= 0.9539, scope="In-domain ensemble"),
            dict(check="Anomal-E NF-UNSW 0%",
                  criterion=">= 0.8845 (in-domain 3-seed ensemble)",
                  value=unsw_f1, passed=unsw_f1 >= 0.8845, scope="In-domain ensemble"),
            dict(check="Anomal-E NF-UNSW 4%",
                  criterion=">= 0.9235 (XeNIDS DA-ensemble)",
                  value=da_f1, passed=da_f1 >= 0.9235, scope="XeNIDS DA"),
            dict(check="Butt et al. NF-UNSW BERT",
                  criterion=">= 0.834 (in-domain 3-seed ensemble)",
                  value=unsw_f1, passed=unsw_f1 >= 0.834, scope="In-domain ensemble"),
            dict(check="Min (MemAE) NF-UNSW AUC",
                  criterion=">= 0.9113", value=unsw_auc, passed=unsw_auc >= 0.9113,
                  scope="In-domain 3-seed ensemble"),
            dict(check="Xu (Deep IF) NF-UNSW AUC",
                  criterion=">= 0.932", value=unsw_auc, passed=unsw_auc >= 0.932,
                  scope="In-domain 3-seed ensemble"),
            dict(check="Koukoulis NF-UNSW AUC",
                  criterion=">= 0.95", value=unsw_auc, passed=unsw_auc >= 0.95,
                  scope="In-domain 3-seed ensemble"),
            dict(check="Causal SHAP / CF / STL / concept gates",
                  criterion="Owned by notebooks 02-08",
                  value="not_in_scope_for_01", passed=True, scope="Later notebooks"),
            dict(check="Anomal-E NF-CSE-CIC 4% high-MF1 sweep",
                  criterion=">= 0.9438 (test-oracle MaxF1)",
                  value=hi["macro_f1"], passed=hi["macro_f1"] >= 0.9438,
                  scope="Matrix best-seed test-oracle"),
            dict(check="Anomal-E NF-CSE-CIC 0% high-MF1 sweep",
                  criterion=">= 0.9539 (test-oracle MaxF1)",
                  value=hi["macro_f1"], passed=hi["macro_f1"] >= 0.9539,
                  scope="Matrix best-seed test-oracle"),
        ]
        for c in checks: c["passed"] = bool(c["passed"])
        (ART / "notebook01_benchmark_verdict.json").write_text(json.dumps(checks, indent=2))
        import csv
        with (ART / "notebook01_benchmark_verdict.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["check", "criterion", "value", "passed", "scope"])
            w.writeheader()
            for r in checks: w.writerow(r)
        n_pass = sum(1 for c in checks if c["passed"])
        print(f"  ✓ benchmark_verdict: {n_pass}/{len(checks)} PASS", flush=True)
        return n_pass == len(checks)
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}", flush=True)
        return False


def check_artifacts() -> bool:
    print("\n" + "═" * 70)
    print("CHECK — all expected artifacts on disk?")
    print("═" * 70)
    missing = []
    for path, phase, gen, name in ARTIFACT_REGISTRY:
        if not path.exists():
            missing.append((phase, str(path.relative_to(ROOT))))
    if missing:
        print(f"\n  ✗ {len(missing)} missing artifact(s):")
        for phase, p in sorted(missing):
            print(f"    [phase {phase}] {p}")
        return False
    print(f"\n  ✓ All {len(ARTIFACT_REGISTRY)} artifacts present")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="all", help="Which table(s) to regenerate (or 'all')")
    ap.add_argument("--phase", type=int, choices=[0, 1, 2, 3, 4], help="Run only one phase")
    ap.add_argument("--force", action="store_true", help="Re-run even if artifacts exist")
    ap.add_argument("--check", action="store_true", help="Just check artifacts, don't generate")
    args = ap.parse_args()

    if args.check:
        ok = check_artifacts()
        sys.exit(0 if ok else 1)

    print("Q1-paper reproducibility script (caushap_nids)")
    print(f"  artifact root: {ART}")
    print(f"  force: {args.force}")
    print(f"  phase: {args.phase if args.phase is not None else 'all'}")
    print(f"  table: {args.table}")

    overall = True
    if args.phase is None or args.phase == 0:
        overall &= phase_0_matrix(force=args.force)
    if args.phase is None or args.phase == 1:
        overall &= phase_1_ensembles(force=args.force)
    if args.phase is None or args.phase == 2:
        overall &= phase_2_faithfulness(force=args.force)
    if args.phase is None or args.phase == 3:
        overall &= phase_3_submodules(force=args.force)
    if args.phase is None or args.phase == 4:
        overall &= phase_4_tables(table_name=args.table, force=args.force)

    print("\n" + "═" * 70)
    print(f"{'✓ ALL PHASES PASSED' if overall else '✗ ONE OR MORE PHASES FAILED'}")
    print("═" * 70)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
