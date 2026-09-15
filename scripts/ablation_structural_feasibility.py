"""A4 vs A5 ablation under the structural feasibility criterion.

Reruns the paper's decisive ablation -- NF-DAG-v1 against a size-matched random
DAG -- with two changes that matter:

  1. Feasibility is the causal-recourse criterion (structural.py), not the
     co-change criterion, which was satisfiable by changing every feature.
  2. Counterfactuals are additionally scored by the graph-independent NF-v2
     plausibility oracle (plausibility.py).  The published comparison scored
     each configuration against the very graph under test, so it could not
     distinguish a better graph from a more easily satisfied one.  The oracle
     is fixed across configurations, so it can.

The detector is held fixed across configurations, matching the paper's design.

Usage:
    python scripts/ablation_structural_feasibility.py --anchors 24 --generations 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl

from scipy.stats import wilcoxon

from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS
from caushap_nids.models import DeepAutoEncoder, IFDetector
from caushap_nids.models.ensemble import QuantileNormalizer, geometric_ensemble
from caushap_nids.xai_layers.multi_obj_cf.nsga import generate_cf_pareto_front
from caushap_nids.xai_layers.multi_obj_cf.plausibility import (
    FeatureTransform,
    plausibility_rate,
    violated_invariants,
)
from caushap_nids.xai_layers.multi_obj_cf.structural import fit_structural_equations

NAMES = list(NF_V2_FEATURE_COLS)


class _AEScorer:
    """Adapts DeepAutoEncoder to the detector interface the CF search expects."""

    def __init__(self, ae: DeepAutoEncoder) -> None:
        self._ae = ae

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._ae.score(np.asarray(X, dtype=np.float32)), dtype=np.float64)


class _EnsembleScorer:
    """AE + Isolation Forest geometric-mean ensemble, matching the paper's detector.

    Normalisers are fitted on benign scores here rather than reloaded from the
    published calibration, so the operating point is an approximation of the
    paper's -- fine for an ablation, where the detector is held fixed across
    configurations and only the explanation layer varies.
    """

    def __init__(self, ae: DeepAutoEncoder, if_det: IFDetector, X_benign: np.ndarray,
                 alpha: float) -> None:
        self._ae, self._if, self._alpha = ae, if_det, alpha
        self._ae_norm = QuantileNormalizer().fit(
            ae.score(np.asarray(X_benign, dtype=np.float32))
        )
        self._if_norm = QuantileNormalizer().fit(if_det.score(X_benign))

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        ae_n = self._ae_norm.transform(self._ae.score(X.astype(np.float32)))
        if_n = self._if_norm.transform(self._if.score(X))
        return geometric_ensemble(ae_n, if_n, alpha=self._alpha)


def random_dag_like(dag: nx.DiGraph, seed: int) -> nx.DiGraph:
    """Random DAG with the same node set and edge count, acyclic by construction."""
    rng = np.random.default_rng(seed)
    nodes = list(dag.nodes())
    order = list(rng.permutation(len(nodes)))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    pairs = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    chosen = rng.choice(len(pairs), size=dag.number_of_edges(), replace=False)
    for c in chosen:
        i, j = pairs[c]
        g.add_edge(nodes[order[i]], nodes[order[j]])
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NF-UNSW-NB15-v2.parquet")
    ap.add_argument("--ae", default="artifacts/models/_shared/detector-5821bc9691f4/nf_unsw15/42/ae.pt")
    ap.add_argument("--if-model", default=None,
                    help="path to if.pkl; enables the full AE+IF ensemble detector")
    ap.add_argument("--alpha", type=float, default=0.9,
                    help="ensemble mixing weight (paper's strict operating point)")
    ap.add_argument("--dag", default="artifacts/nf_dag_v1.graphml")
    ap.add_argument("--pruned-dag", default=None,
                    help="optional second graph to compare (e.g. the data-pruned NF-DAG-v1)")
    ap.add_argument("--anchors", type=int, default=24)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-controls", type=int, default=3,
                    help="number of independent random DAGs (a single one is a sample of one)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/ablation_structural.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- data ---------------------------------------------------------------
    # scan+head so the 17M-row CIC2018 file is not materialised in full.
    df = pl.scan_parquet(Path("data") / args.dataset).head(args.rows).collect()
    label_col = "Label" if "Label" in df.columns else "label"
    y = df[label_col].to_numpy()
    X_raw = df.select(NAMES).to_numpy().astype(np.float64)

    ft = FeatureTransform.fit(X_raw[y == 0])
    X = ft.transform(X_raw)
    X_benign, X_attack = X[y == 0], X[y == 1]
    print(f"benign={len(X_benign):,}  attack={len(X_attack):,}")

    # ---- detector (held fixed across configurations) ------------------------
    ae = DeepAutoEncoder(in_dim=len(NAMES), hidden_dims=[64, 32, 16], dropout=0.1)
    ae.load(args.ae)
    if args.if_model:
        if_det = IFDetector()
        if_det.load(args.if_model)
        cal = X_benign[rng.choice(len(X_benign), size=min(20_000, len(X_benign)), replace=False)]
        detector = _EnsembleScorer(ae, if_det, cal, alpha=args.alpha)
        print(f"detector: AE+IF ensemble (alpha={args.alpha})")
    else:
        detector = _AEScorer(ae)
        print("detector: AE only")

    benign_scores = detector.score(X_benign[:20_000])
    threshold = float(np.quantile(benign_scores, 0.99))   # FPR ~= 0.01
    print(f"threshold (benign 99th pct) = {threshold:.4f}")

    # Anchors: attack flows the detector actually flags.
    attack_scores = detector.score(X_attack[:20_000])
    flagged = X_attack[:20_000][attack_scores >= threshold]
    idx = rng.choice(len(flagged), size=min(args.anchors, len(flagged)), replace=False)
    anchors = flagged[idx]
    print(f"anchors: {len(anchors)} flagged attack flows\n")

    # ---- graphs & structural equations --------------------------------------
    dag_v1 = nx.read_graphml(args.dag)
    configs = {"A4_nf_dag_v1": (dag_v1, fit_structural_equations(X_benign, dag_v1, NAMES))}
    if args.pruned_dag:
        dp = nx.read_graphml(args.pruned_dag)
        configs["A4p_pruned"] = (dp, fit_structural_equations(X_benign, dp, NAMES))
        print(f"pruned graph: {dp.number_of_edges()} edges "
              f"(vs {dag_v1.number_of_edges()} full)")
    for c in range(args.random_controls):
        g = random_dag_like(dag_v1, seed=args.seed + 1000 + c)
        configs[f"A5_random_{c}"] = (g, fit_structural_equations(X_benign, g, NAMES))

    bg = X_benign[rng.choice(len(X_benign), size=min(3000, len(X_benign)), replace=False)]
    results: dict[str, dict] = {}
    per_anchor: dict[str, list] = {}

    for cfg_name, (dag, sem) in configs.items():
        print(f"--- {cfg_name} ---")
        rows = []
        for i, x in enumerate(anchors, 1):
            cfs = generate_cf_pareto_front(
                detector, dag, x,
                feature_names=NAMES,
                threshold=threshold,
                background=bg,
                population_size=args.population,
                n_generations=args.generations,
                n_cfs=1,
                return_valid_only=True,
                sem=sem,
                seed=args.seed,
            )
            if not cfs:
                # Record the failure so configurations are compared on the same
                # anchors rather than on whichever ones each happened to solve.
                rows.append(None)
                continue
            cf = cfs[0]
            x_cf_raw = ft.inverse_transform(cf.x_cf)
            rows.append({
                "validity": cf.validity,
                "feasibility": cf.feasibility_rate,
                "sparsity": float(cf.sparsity),
                "proximity": cf.proximity,
                "plausibility": plausibility_rate(x_cf_raw, NAMES),
                "n_impossible": float(len(violated_invariants(x_cf_raw, NAMES))),
                "possible": float(len(violated_invariants(x_cf_raw, NAMES)) == 0),
            })
            if i % 10 == 0:
                print(f"  {i}/{len(anchors)}", flush=True)

        per_anchor[cfg_name] = rows
        solved = [r for r in rows if r is not None]
        agg = {k: float(np.mean([r[k] for r in solved])) for k in solved[0]}
        agg["n_solved"] = len(solved)
        agg["pct_physically_possible"] = agg.pop("possible")
        results[cfg_name] = agg
        print(f"  {agg}\n", flush=True)

    # ---- report -------------------------------------------------------------
    print("=" * 78)
    hdr = f"{'metric':<26}" + "".join(f"{c:>24}" for c in results)
    print(hdr)
    print("-" * len(hdr))
    for k in ("validity", "feasibility", "sparsity", "proximity",
              "plausibility", "pct_physically_possible"):
        print(f"{k:<26}" + "".join(f"{results[c][k]:>24.4f}" for c in results))

    a4 = results["A4_nf_dag_v1"]
    ctrl = [v for k, v in results.items() if k.startswith("A5_")]
    mean_ctrl = {k: float(np.mean([c[k] for c in ctrl])) for k in a4 if k != "n"}

    ctrl_names = [k for k in results if k.startswith("A5_")]

    # ---- paired tests, anchor by anchor -------------------------------------
    # The mean-vs-mean comparison the paper reports cannot tell a real gap from
    # the spread between random graphs. Pairing on anchors can, and the spread
    # across controls is reported alongside so the reader can judge for himself.
    print("\n" + "=" * 78)
    print("PAIRED COMPARISON: NF-DAG-v1 vs pooled random controls (same anchors)")
    print("=" * 78)

    stats_out: dict[str, dict] = {}
    for metric in ("feasibility", "plausibility", "sparsity", "possible"):
        a4_vals, ctrl_vals = [], []
        for i in range(len(anchors)):
            a = per_anchor["A4_nf_dag_v1"][i]
            cs = [per_anchor[c][i] for c in ctrl_names]
            if a is None or any(c is None for c in cs):
                continue
            a4_vals.append(a[metric])
            ctrl_vals.append(float(np.mean([c[metric] for c in cs])))

        a4_arr, c_arr = np.array(a4_vals), np.array(ctrl_vals)
        diff = a4_arr - c_arr
        gap = float(diff.mean())

        if np.allclose(diff, 0):
            p, note = 1.0, "identical on every anchor"
        else:
            try:
                p = float(wilcoxon(a4_arr, c_arr).pvalue)
                note = ""
            except ValueError as exc:
                p, note = float("nan"), str(exc)

        # Between-control spread: the noise floor any real gap must clear.
        spread = float(np.mean([
            np.std([per_anchor[c][i][metric] for c in ctrl_names])
            for i in range(len(anchors))
            if all(per_anchor[c][i] is not None for c in ctrl_names)
        ]))

        boot = np.array([
            diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)
        ])
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

        verdict = "SIGNIFICANT" if (p < 0.05 and abs(gap) > spread) else "not distinguishable"
        stats_out[metric] = {
            "gap": gap, "p": p, "ci95": ci,
            "between_control_sd": spread, "n_pairs": len(diff), "verdict": verdict,
        }
        print(f"\n{metric}:")
        print(f"  gap (A4 - random)      {gap:+.4f}   95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"  Wilcoxon p             {p:.4g} {note}")
        print(f"  between-control SD     {spread:.4f}   <- noise floor")
        print(f"  verdict                {verdict}   (n={len(diff)} paired anchors)")

    results["_paired_stats"] = stats_out
    results["_random_control_mean"] = mean_ctrl
    results["_random_control_mean"] = mean_ctrl

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
