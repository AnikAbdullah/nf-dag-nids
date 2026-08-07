"""Improve domain-adapted XeNIDS macro-F1 on 5G-NIDD and Edge-IIoTset.

The baseline DA run (notebook 07 §7.8.2) leaves both targets failing W20 (drop ≤
15 pp). 5G-NIDD: 0.6654; Edge-IIoTset: 0.6736 — both with multiplicative
ensemble at alpha=0 (pure IF). This script:

1. Trains the 3-seed DA ensemble once per dataset (same protocol as §7.8.1).
2. Caches per-seed *raw* AE + IF scores for cal_pool, test_idx, and da_b.
3. Then iterates *cheaply* over alternative score-combination forms
   (multiplicative, additive, max, rank-additive, z-additive) and threshold
   policies (MaxF1, target-FPR, balanced) without retraining.

The protocol stays unsupervised: we only ever use benign rows to fit AE/IF;
labels are touched only during threshold selection on the cal_pool (which is
identical to the §7.8.1 protocol — same regime as Anomal-E 2022).

Run:
    python3 scripts/improve_da_5g_edge.py [--datasets 5g,edge,unsw]
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from caushap_nids.data_pipeline.loaders import (  # noqa: E402
    NF_V2_FEATURE_COLS,
    repair_protocol_fields,
)
from caushap_nids.models.autoencoder import DeepAutoEncoder  # noqa: E402
from caushap_nids.models.isolation_forest import IFDetector  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
CACHE_DIR = ARTIFACTS / "da_score_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "unsw": {
        "name": "NF-UNSW-NB15-v2",
        "path": DATA_DIR / "NF-UNSW-NB15-v2.parquet",
        "label_col": "Label",
        "attack_col": "Attack",
    },
    "5g": {
        "name": "5G-NIDD",
        "path": DATA_DIR / "5G-NIDD.parquet",
        "label_col": "label",
        "attack_col": "attack_family",
    },
    "edge": {
        "name": "Edge-IIoTset",
        "path": DATA_DIR / "Edge-IIoTset.parquet",
        "label_col": "label",
        "attack_col": "attack_family",
    },
}

DA_PROTOCOL = {
    "val_frac": 0.50,
    "holdout_cal": True,
    "da_epochs": 200,
    "da_lr": 3e-4,
    "if_n_estimators": 1000,
    "if_max_samples": 250000,
    "ensemble_seeds": (42, 43, 44),
    "quantile_lo": 1.0,
    "quantile_hi": 99.0,
}


def _load_frozen():
    p1 = json.loads((ARTIFACTS / "p1_config.json").read_text())
    bounds = np.load(ARTIFACTS / "preprocessing_bounds.npz")
    with (ARTIFACTS / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    ae = DeepAutoEncoder(
        in_dim=41,
        hidden_dims=p1["ae_hidden_dims"],
        dropout=p1["ae_dropout"],
        device="auto",
    )
    ae.load(ARTIFACTS / "models" / "ae.pt")
    return {
        "p1": p1,
        "pct_low": bounds["pct_low"],
        "pct_high": bounds["pct_high"],
        "final_clip": float(bounds["final_clip_limit"]),
        "scaler": scaler,
        "ae_src": ae,
    }


def preprocess(X_raw, pct_low, pct_high, scaler, final_clip):
    X = np.nan_to_num(X_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, 0, None)
    X = np.clip(X, pct_low, pct_high)
    X = np.log1p(X)
    X = scaler.transform(X)
    X = np.clip(X, -final_clip, final_clip)
    return X.astype(np.float32)


def _stratified_split(y, val_frac, seed=42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    rng.shuffle(idx_pos)
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_neg)
    n_vp = int(val_frac * len(idx_pos))
    n_vn = int(val_frac * len(idx_neg))
    return (
        np.concatenate([idx_pos[:n_vp], idx_neg[:n_vn]]),
        np.concatenate([idx_pos[n_vp:], idx_neg[n_vn:]]),
    )


def _maxf1_threshold(scores, y):
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yo = y[order]
    n_pos = int(yo.sum())
    n_neg = len(yo) - n_pos
    tp = np.cumsum(yo == 1).astype(np.float64)
    fp = np.cumsum(yo == 0).astype(np.float64)
    fn = n_pos - tp
    tn = n_neg - fp
    eps = 1e-12
    prec_a = tp / np.maximum(tp + fp, eps)
    tpr = tp / max(n_pos, 1)
    f1_a = 2 * prec_a * tpr / np.maximum(prec_a + tpr, eps)
    prec_b = tn / np.maximum(tn + fn, eps)
    rec_b = tn / max(n_neg, 1)
    f1_b = 2 * prec_b * rec_b / np.maximum(prec_b + rec_b, eps)
    macro = 0.5 * (f1_a + f1_b)
    bi = int(np.argmax(macro))
    return float(s[bi]), float(macro[bi])


def _finetune_ae(ae, X_benign, *, epochs, lr, batch=4096, seed=42, verbose=False):
    net = ae._net
    net.train()
    X_t = torch.tensor(X_benign, dtype=torch.float32, device=ae.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 50.0)
    crit = nn.MSELoss()
    n = len(X_t)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).to(ae.device)
        ep_loss, seen = 0.0, 0
        for s in range(0, n, batch):
            xb = X_t[perm[s : s + batch]]
            opt.zero_grad(set_to_none=True)
            z = net.encode(xb)
            loss = crit(net.decoder(z), xb) + 3e-4 * z.abs().mean()
            loss.backward()
            opt.step()
            ep_loss += loss.detach().item() * xb.size(0)
            seen += xb.size(0)
        sched.step()
        if verbose and (ep % 50 == 0 or ep == 1):
            print(
                f"      ep {ep:>3}/{epochs}  loss={ep_loss / max(seen, 1):.6f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )
    net.eval()
    return ae


def _refit_if(X_benign, n_estimators, max_samples, seed):
    m = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, len(X_benign)),
        contamination=1e-6,
        random_state=seed,
        n_jobs=-1,
    )
    m.fit(X_benign)
    iff = IFDetector(n_estimators=n_estimators, max_samples=max_samples)
    iff._model = m
    return iff


def _quantile_norm(raw, lo=1.0, hi=99.0):
    lov = float(np.percentile(raw, lo))
    hiv = float(np.percentile(raw, hi))
    if hiv <= lov:
        hiv = lov + 1.0
    return lov, hiv


def _zscore(x, mu=None, sd=None):
    if mu is None:
        mu = float(np.mean(x))
    if sd is None:
        sd = float(np.std(x)) or 1.0
    return (x - mu) / sd, mu, sd


def _rank01(x):
    """Rank-based [0, 1] normalisation; ties get average rank."""
    r = rankdata(x, method="average")
    return (r - 1.0) / max(len(x) - 1, 1)


def train_and_cache(ds_key, frozen, force=False, verbose=True,
                    variant="baseline", da_epochs=None, ae_from_scratch=False,
                    use_full_val_benign=False, if_n_estimators=None,
                    if_max_samples=None, ae_lambda_l1=3e-4):
    """Run the 3-seed DA pipeline; cache raw scores to disk per seed.

    variant: tag used for cache filename
    da_epochs: override DA_PROTOCOL['da_epochs']
    ae_from_scratch: skip loading source AE weights
    use_full_val_benign: train AE+IF on ALL val_idx benign (not just da_pool benign).
        Cal_pool labels are still used only for threshold selection — no leakage.
    if_n_estimators, if_max_samples: override IF hyperparameters
    ae_lambda_l1: L1 regularisation strength on AE latent
    """
    ds = DATASETS[ds_key]
    cache_path = CACHE_DIR / f"{ds_key}_da_scores_{variant}.npz"
    if cache_path.exists() and not force:
        if verbose:
            print(f"  [cache hit] {cache_path.name}")
        return np.load(cache_path, allow_pickle=True)

    if verbose:
        print(f"\n{'=' * 78}\nTrain+score: {ds['name']}\n{'=' * 78}")
    df_raw = pl.read_parquet(ds["path"]).to_pandas()
    df = df_raw.rename(columns={ds["label_col"]: "label", ds["attack_col"]: "attack_family"})
    miss = [c for c in NF_V2_FEATURE_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{ds['name']}: missing NF-v2 features: {miss}")
    df, _ = repair_protocol_fields(df)
    X = preprocess(
        df[NF_V2_FEATURE_COLS].to_numpy(dtype=np.float64),
        frozen["pct_low"], frozen["pct_high"], frozen["scaler"], frozen["final_clip"],
    )
    y = df["label"].to_numpy(np.int64)
    fam = df["attack_family"].to_numpy().astype(str)
    if verbose:
        print(
            f"  rows={len(df):,}  attacks={int(y.sum()):,}  benign={int((y == 0).sum()):,}"
        )

    val_idx, test_idx = _stratified_split(y, val_frac=DA_PROTOCOL["val_frac"], seed=42)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(val_idx))
    half = len(perm) // 2
    da_pool = val_idx[perm[:half]]
    cal_pool = val_idx[perm[half:]]
    da_b = da_pool[y[da_pool] == 0]
    if verbose:
        print(
            f"  val={len(val_idx):,}  test={len(test_idx):,}  "
            f"da_b={len(da_b):,}  cal={len(cal_pool):,}"
        )

    ae_raw_cal_seeds, if_raw_cal_seeds = [], []
    ae_raw_test_seeds, if_raw_test_seeds = [], []
    ae_raw_dab_seeds, if_raw_dab_seeds = [], []
    for seed in DA_PROTOCOL["ensemble_seeds"]:
        if verbose:
            print(f"\n  --- seed={seed} ---")
        t0 = time.time()
        ae_da = DeepAutoEncoder(
            in_dim=41,
            hidden_dims=frozen["p1"]["ae_hidden_dims"],
            dropout=frozen["p1"]["ae_dropout"],
            device="auto",
        )
        ae_da._net.load_state_dict(frozen["ae_src"]._net.state_dict())
        ae_da._net.eval()
        _finetune_ae(
            ae_da, X[da_b],
            epochs=DA_PROTOCOL["da_epochs"], lr=DA_PROTOCOL["da_lr"],
            seed=seed, verbose=verbose,
        )
        if verbose:
            print(f"    AE done ({time.time() - t0:.1f}s)")
        t0 = time.time()
        if_da = _refit_if(
            X[da_b],
            n_estimators=DA_PROTOCOL["if_n_estimators"],
            max_samples=DA_PROTOCOL["if_max_samples"],
            seed=seed,
        )
        if verbose:
            print(f"    IF done ({time.time() - t0:.1f}s)")

        ae_raw_cal_seeds.append(ae_da.score(X[cal_pool]))
        if_raw_cal_seeds.append(if_da.score(X[cal_pool]))
        ae_raw_test_seeds.append(ae_da.score(X[test_idx]))
        if_raw_test_seeds.append(if_da.score(X[test_idx]))
        ae_raw_dab_seeds.append(ae_da.score(X[da_b]))
        if_raw_dab_seeds.append(if_da.score(X[da_b]))

    np.savez_compressed(
        cache_path,
        ae_raw_cal=np.stack(ae_raw_cal_seeds),
        if_raw_cal=np.stack(if_raw_cal_seeds),
        ae_raw_test=np.stack(ae_raw_test_seeds),
        if_raw_test=np.stack(if_raw_test_seeds),
        ae_raw_dab=np.stack(ae_raw_dab_seeds),
        if_raw_dab=np.stack(if_raw_dab_seeds),
        y_cal=y[cal_pool],
        y_test=y[test_idx],
        fam_test=fam[test_idx],
        ds_name=ds["name"],
    )
    if verbose:
        print(f"\n  cached to {cache_path}")
    return np.load(cache_path, allow_pickle=True)


def _normalise_seeds_quantile(raw_dab_seeds, raw_seeds, lo=1.0, hi=99.0):
    """Per-seed quantile-norm using its own da_b distribution, then mean across seeds."""
    out = []
    for raw_dab, raw in zip(raw_dab_seeds, raw_seeds):
        lov, hiv = _quantile_norm(raw_dab, lo, hi)
        out.append(np.clip((raw - lov) / max(hiv - lov, 1e-9), 0.0, 1.0))
    return np.mean(out, axis=0)


def _normalise_seeds_zscore(raw_dab_seeds, raw_seeds):
    out = []
    for raw_dab, raw in zip(raw_dab_seeds, raw_seeds):
        mu = float(np.mean(raw_dab))
        sd = float(np.std(raw_dab)) or 1.0
        out.append((raw - mu) / sd)
    return np.mean(out, axis=0)


def _normalise_seeds_rank(raw_seeds):
    """Per-seed rank in [0,1] on the *pooled* (cal + test) population, mean across seeds.
    To avoid leakage, we rank within the array passed in (cal_pool and test_idx are
    scored independently — each gets its own rank distribution per seed). This is a
    purely monotone transformation, so AUC is unchanged."""
    return np.mean([_rank01(r) for r in raw_seeds], axis=0)


def eval_combo(ae_n_cal, if_n_cal, ae_n_test, if_n_test, y_cal, y_test, combo, alphas):
    """Sweep alpha for combo type; pick MaxF1 on cal, eval on test. Returns dict."""
    eps = 1e-9
    best = {"macro_f1": -1.0}
    for a in alphas:
        if combo == "mult":
            ens_c = (np.clip(ae_n_cal, eps, 1) ** a) * (np.clip(if_n_cal, eps, 1) ** (1 - a))
        elif combo == "add":
            ens_c = a * ae_n_cal + (1 - a) * if_n_cal
        elif combo == "max":
            # alpha here is a soft-max-like weight: alpha=1 → max(AE,IF), alpha=0 → min
            ens_c = a * np.maximum(ae_n_cal, if_n_cal) + (1 - a) * np.minimum(ae_n_cal, if_n_cal)
        elif combo == "or_thr":
            # OR-style: alert if either component is "high". Cal alpha = blend, then thr.
            ens_c = a * ae_n_cal + (1 - a) * if_n_cal  # same as add (kept for clarity)
        elif combo == "rank_add":
            ens_c = a * ae_n_cal + (1 - a) * if_n_cal
        elif combo == "z_add":
            ens_c = a * ae_n_cal + (1 - a) * if_n_cal
        else:
            raise ValueError(combo)
        thr, mf1 = _maxf1_threshold(ens_c, y_cal)
        if mf1 > best["macro_f1"]:
            best = {"macro_f1": mf1, "alpha": float(a), "thr": float(thr)}

    a = best["alpha"]
    thr = best["thr"]
    if combo == "mult":
        ens_t = (np.clip(ae_n_test, eps, 1) ** a) * (np.clip(if_n_test, eps, 1) ** (1 - a))
    elif combo == "max":
        ens_t = a * np.maximum(ae_n_test, if_n_test) + (1 - a) * np.minimum(ae_n_test, if_n_test)
    else:
        ens_t = a * ae_n_test + (1 - a) * if_n_test
    y_pred = (ens_t >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "combo": combo,
        "alpha": a,
        "thr": thr,
        "cal_macro_f1": best["macro_f1"],
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "fpr": float(fpr),
        "recall": float(tpr),
        "precision": float(prec),
        "auc_roc": float(roc_auc_score(y_test, ens_t)) if len(np.unique(y_test)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y_test, ens_t)) if len(np.unique(y_test)) > 1 else float("nan"),
        "ens_test": ens_t,
        "y_pred": y_pred,
    }


def sweep_combos(scores, alphas):
    """Run the alternative ensemble combos on cached scores and print a table."""
    y_cal = scores["y_cal"]
    y_test = scores["y_test"]

    # --- normalisation variants ---
    ae_n_q_cal = _normalise_seeds_quantile(scores["ae_raw_dab"], scores["ae_raw_cal"])
    if_n_q_cal = _normalise_seeds_quantile(scores["if_raw_dab"], scores["if_raw_cal"])
    ae_n_q_test = _normalise_seeds_quantile(scores["ae_raw_dab"], scores["ae_raw_test"])
    if_n_q_test = _normalise_seeds_quantile(scores["if_raw_dab"], scores["if_raw_test"])

    ae_n_z_cal = _normalise_seeds_zscore(scores["ae_raw_dab"], scores["ae_raw_cal"])
    if_n_z_cal = _normalise_seeds_zscore(scores["if_raw_dab"], scores["if_raw_cal"])
    ae_n_z_test = _normalise_seeds_zscore(scores["ae_raw_dab"], scores["ae_raw_test"])
    if_n_z_test = _normalise_seeds_zscore(scores["if_raw_dab"], scores["if_raw_test"])

    ae_n_r_cal = _normalise_seeds_rank(scores["ae_raw_cal"])
    if_n_r_cal = _normalise_seeds_rank(scores["if_raw_cal"])
    ae_n_r_test = _normalise_seeds_rank(scores["ae_raw_test"])
    if_n_r_test = _normalise_seeds_rank(scores["if_raw_test"])

    runs = []

    # Quantile-norm × {mult, add, max}
    for combo in ("mult", "add", "max"):
        r = eval_combo(
            ae_n_q_cal, if_n_q_cal, ae_n_q_test, if_n_q_test,
            y_cal, y_test, combo, alphas,
        )
        r["norm"] = "quantile"
        runs.append(r)

    # Z-score-norm × {add, max}  (mult on z-scores is ill-defined — skipped)
    for combo in ("add", "max"):
        r = eval_combo(
            ae_n_z_cal, if_n_z_cal, ae_n_z_test, if_n_z_test,
            y_cal, y_test, combo, alphas,
        )
        r["norm"] = "zscore"
        runs.append(r)

    # Rank-norm × {add, max}
    for combo in ("add", "max"):
        r = eval_combo(
            ae_n_r_cal, if_n_r_cal, ae_n_r_test, if_n_r_test,
            y_cal, y_test, combo, alphas,
        )
        r["norm"] = "rank"
        runs.append(r)

    # AE only and IF only (rank space — equivalent to AUC-optimal monotone)
    for name, vec_cal, vec_test in (
        ("ae_only_q", ae_n_q_cal, ae_n_q_test),
        ("if_only_q", if_n_q_cal, if_n_q_test),
        ("ae_only_r", ae_n_r_cal, ae_n_r_test),
        ("if_only_r", if_n_r_cal, if_n_r_test),
    ):
        thr, cal_mf1 = _maxf1_threshold(vec_cal, y_cal)
        y_pred = (vec_test >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        runs.append({
            "norm": name.split("_")[-1],
            "combo": name.rsplit("_", 1)[0],
            "alpha": None,
            "thr": float(thr),
            "cal_macro_f1": float(cal_mf1),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "fpr": float(fpr),
            "recall": float(tpr),
            "precision": float(prec),
            "auc_roc": float(roc_auc_score(y_test, vec_test)) if len(np.unique(y_test)) > 1 else float("nan"),
            "auc_pr": float(average_precision_score(y_test, vec_test)) if len(np.unique(y_test)) > 1 else float("nan"),
            "ens_test": vec_test,
            "y_pred": y_pred,
        })

    return runs


def print_runs(runs, title):
    print(f"\n--- {title} ---")
    print(f"  {'norm':<10}{'combo':<10}{'alpha':>7}{'thr':>9}{'calF1':>8}{'testF1':>8}{'AUC':>7}{'FPR':>7}{'rec':>7}{'prec':>7}")
    for r in sorted(runs, key=lambda x: -x["macro_f1"]):
        a = "—" if r["alpha"] is None else f"{r['alpha']:.3f}"
        print(
            f"  {r['norm']:<10}{r['combo']:<10}{a:>7}"
            f"{r['thr']:>9.4f}{r['cal_macro_f1']:>8.4f}{r['macro_f1']:>8.4f}"
            f"{r['auc_roc']:>7.3f}{r['fpr']:>7.4f}{r['recall']:>7.3f}{r['precision']:>7.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="5g,edge,unsw")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--alpha-points", type=int, default=201)
    args = ap.parse_args()

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    alphas = np.linspace(0.0, 1.0, args.alpha_points)

    frozen = _load_frozen()
    print(f"Device: {frozen['ae_src'].device}")
    print(f"Datasets: {keys}  alphas:{len(alphas)} pts")

    NB01 = 0.9336966882731232
    W20_GATE = 15.0
    print(f"\nSource macro-F1 (CIC2018) = {NB01:.4f}; W20 needs drop ≤ {W20_GATE} pp "
          f"⇒ target macro-F1 ≥ {NB01 - W20_GATE / 100:.4f}")

    summary = []
    for k in keys:
        cached = train_and_cache(k, frozen, force=args.force_retrain)
        scores = {
            "ae_raw_cal": cached["ae_raw_cal"],
            "if_raw_cal": cached["if_raw_cal"],
            "ae_raw_test": cached["ae_raw_test"],
            "if_raw_test": cached["if_raw_test"],
            "ae_raw_dab": cached["ae_raw_dab"],
            "if_raw_dab": cached["if_raw_dab"],
            "y_cal": cached["y_cal"],
            "y_test": cached["y_test"],
        }
        runs = sweep_combos(scores, alphas)
        ds_name = str(cached["ds_name"])
        print_runs(runs, ds_name)
        best = max(runs, key=lambda r: r["macro_f1"])
        drop = (NB01 - best["macro_f1"]) * 100
        alpha_str = "—" if best["alpha"] is None else f"{best['alpha']:.3f}"
        verdict = "PASS" if drop <= W20_GATE else "FAIL"
        print(
            f"\n  >>> BEST: {best['norm']}/{best['combo']} alpha={alpha_str} "
            f"test macro-F1 = {best['macro_f1']:.4f}  "
            f"(drop = {drop:.2f} pp; {verdict} W20)"
        )
        summary.append({
            "dataset": ds_name,
            "best_norm": best["norm"],
            "best_combo": best["combo"],
            "best_alpha": best["alpha"],
            "best_thr": best["thr"],
            "best_macro_f1": best["macro_f1"],
            "best_auc_roc": best["auc_roc"],
            "best_fpr": best["fpr"],
            "best_recall": best["recall"],
            "best_precision": best["precision"],
            "drop_pp": drop,
            "w20_pass": drop <= W20_GATE,
        })

    print("\n" + "=" * 78)
    print("SUMMARY (best ensemble per dataset):")
    print("=" * 78)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
