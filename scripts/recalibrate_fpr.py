"""Recalibrate the AE+IF ensemble operating point to a tighter FPR target.

Reuses the existing trained artifacts (ae.pt, if.pkl, scaler.pkl, percentile
bounds) and re-picks alpha and threshold so that validation FPR <= TARGET_FPR.
Rewrites threshold_config.json, test_metrics.json, p1_config.json,
score_normalizers.json, ensemble_threshold_sweep.csv,
confusion_matrix_singlerun.png, test_results.csv. Does NOT re-train the AE or
IF. NB: the canonical lead figure confusion_matrix.png is the HEADLINE
(3-seed ensemble strict) matrix built by scripts/in_domain_3seed_ensemble.py;
this script only writes the single-run matrix, under a distinct filename so it
does not clobber the headline figure.

Run from project root:
    .venv/bin/python scripts/recalibrate_fpr.py --target-fpr 0.01
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
SEED = 42


# ── DeepAutoEncoder — same architecture as notebooks/01 cell 17 ────────────────
class DeepAutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 41, hidden_dims=None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        enc, prev = [], in_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout
        self.input_dim = in_dim

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def per_feature_error(self, x):
        return (x - self(x)) ** 2

    @torch.no_grad()
    def reconstruction_error(self, x, method: str = "mean"):
        err = self.per_feature_error(x)
        if method == "mean":
            return err.mean(dim=1)
        raise ValueError(f"unknown method: {method}")


def load_split_data(p1_config: dict):
    parquet = DATA_DIR / "NF-CSE-CIC-IDS2018-V2.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    t0 = time.time()
    df = pd.read_parquet(parquet, engine="pyarrow")
    print(f"  loaded {df.shape} in {time.time()-t0:.1f}s")

    label_cols = ["Label", "Attack"]
    feature_cols = p1_config["feature_cols_original"]

    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    df["Attack"] = df["Attack"].fillna("Unknown").astype(str)

    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    df_train = df.iloc[:train_end].copy().drop_duplicates().reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].copy().drop_duplicates().reset_index(drop=True)
    df_test = df.iloc[val_end:].copy().drop_duplicates().reset_index(drop=True)
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")
    return df_train, df_val, df_test, feature_cols


def preprocess(X_raw: np.ndarray, pct_low: np.ndarray, pct_high: np.ndarray,
               scaler, final_clip: float) -> np.ndarray:
    X = np.nan_to_num(X_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, 0, None)
    X = np.clip(X, pct_low, pct_high)
    X = np.log1p(X)
    X = scaler.transform(X)
    X = np.clip(X, -final_clip, final_clip)
    return X.astype(np.float32)


class QuantileNormalizer:
    def __init__(self, low=1.0, high=99.0):
        self.low, self.high = float(low), float(high)
        self.lo_ = self.hi_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64).ravel()
        self.lo_ = float(np.percentile(X, self.low))
        self.hi_ = float(np.percentile(X, self.high))
        if self.hi_ <= self.lo_:
            self.hi_ = self.lo_ + 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).ravel()
        return np.clip((X - self.lo_) / (self.hi_ - self.lo_), 0.0, 1.0)


def sweep_thresholds(scores: np.ndarray, y_true: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(np.int64)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    fn = n_pos - tp
    tn = n_neg - fp
    eps = 1e-12
    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    prec_a = tp / np.maximum(tp + fp, eps)
    f1_a = 2 * prec_a * tpr / np.maximum(prec_a + tpr, eps)
    prec_b = tn / np.maximum(tn + fn, eps)
    rec_b = tn / max(n_neg, 1)
    f1_b = 2 * prec_b * rec_b / np.maximum(prec_b + rec_b, eps)
    macro_f1 = 0.5 * (f1_a + f1_b)
    return s_sorted, tpr, fpr, macro_f1, prec_a


def select_operating_point(scores: np.ndarray, y_true: np.ndarray, target_fpr: float):
    thr, tpr, fpr, mf1, prec = sweep_thresholds(scores, y_true)
    mask = fpr <= target_fpr
    if mask.any():
        pool = np.where(mask)[0]
        composite = mf1[pool] + 1e-6 * tpr[pool] + 1e-9 * prec[pool]
        best_local = pool[int(np.argmax(composite))]
        mode = f"FPR-controlled <= {target_fpr:.2f}"
    else:
        best_local = int(np.argmax(mf1))
        mode = "fallback: best Macro-F1, FPR target not reached"
    return {
        "threshold": float(thr[best_local]),
        "macro_f1": float(mf1[best_local]),
        "fpr": float(fpr[best_local]),
        "tpr": float(tpr[best_local]),
        "attack_precision": float(prec[best_local]),
        "mode": mode,
    }


def geometric_ensemble(ae_n: np.ndarray, if_n: np.ndarray, alpha: float, eps=1e-9):
    return (np.clip(ae_n, eps, 1.0) ** alpha) * (np.clip(if_n, eps, 1.0) ** (1.0 - alpha))


def binary_metrics(y_true, y_pred, score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    out = {
        "macro_f1": float(macro),
        "fpr": float(fpr), "fnr": float(fnr),
        "attack_recall_tpr": float(tpr),
        "attack_precision": float(prec),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if score is not None and len(np.unique(y_true)) > 1:
        out["auc_roc"] = float(roc_auc_score(y_true, score))
        out["auc_pr"] = float(average_precision_score(y_true, score))
    return out


def bootstrap_metrics(y_true, y_pred, score, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["macro_f1", "fpr", "attack_recall_tpr", "attack_precision", "auc_roc", "auc_pr"]
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = binary_metrics(y_true[idx], y_pred[idx], score[idx])
        for k in keys:
            if k in m and not np.isnan(m[k]):
                boot[k].append(m[k])
    cis = {}
    for k, vals in boot.items():
        if vals:
            arr = np.array(vals, dtype=np.float64)
            cis[f"{k}_mean"] = float(arr.mean())
            cis[f"{k}_lo"] = float(np.quantile(arr, 0.025))
            cis[f"{k}_hi"] = float(np.quantile(arr, 0.975))
    return cis


def main(target_fpr: float):
    print(f"\n=== Recalibrating to target_fpr={target_fpr} ===\n")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[1/7] Load p1_config and preprocessing bounds")
    with open(ARTIFACT_DIR / "p1_config.json") as f:
        p1 = json.load(f)
    bounds = np.load(ARTIFACT_DIR / "preprocessing_bounds.npz")
    pct_low = bounds["pct_low"]
    pct_high = bounds["pct_high"]
    final_clip = float(bounds["final_clip_limit"])
    with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("[2/7] Load + split parquet")
    df_train, df_val, df_test, feature_cols = load_split_data(p1)

    print("[3/7] Preprocess val + test")
    X_val_raw = df_val[feature_cols].values
    X_test_raw = df_test[feature_cols].values
    X_val_s = preprocess(X_val_raw, pct_low, pct_high, scaler, final_clip)
    X_test_s = preprocess(X_test_raw, pct_low, pct_high, scaler, final_clip)
    y_val = df_val["Label"].values.astype(int)
    y_test = df_test["Label"].values.astype(int)
    print(f"  X_val={X_val_s.shape}  X_test={X_test_s.shape}")

    print("[4/7] Load AE + score")
    device = torch.device("cpu")
    ae = DeepAutoEncoder(
        in_dim=p1["input_dim"],
        hidden_dims=p1["ae_hidden_dims"],
        dropout=p1["ae_dropout"],
    )
    state = torch.load(ARTIFACT_DIR / "ae.pt", map_location=device, weights_only=True)
    ae.load_state_dict(state)
    ae.eval().to(device)

    @torch.no_grad()
    def ae_score(X: np.ndarray, batch=8192) -> np.ndarray:
        out = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), batch):
            t = torch.from_numpy(X[i:i + batch]).to(device)
            out[i:i + batch] = ae.reconstruction_error(t, method="mean").cpu().numpy()
        return out

    t0 = time.time()
    ae_val = ae_score(X_val_s)
    ae_test = ae_score(X_test_s)
    print(f"  AE scored in {time.time()-t0:.1f}s")

    print("[5/7] Load IF + score")
    with open(ARTIFACT_DIR / "if.pkl", "rb") as f:
        if_model = pickle.load(f)
    t0 = time.time()
    if_val = -if_model.score_samples(X_val_s)
    if_test = -if_model.score_samples(X_test_s)
    print(f"  IF scored in {time.time()-t0:.1f}s")

    print("[6/7] Fit quantile normalizers and sweep α × threshold")
    ae_norm = QuantileNormalizer(1, 99).fit(ae_val)
    if_norm = QuantileNormalizer(1, 99).fit(if_val)
    ae_val_n = ae_norm.transform(ae_val)
    if_val_n = if_norm.transform(if_val)
    ae_test_n = ae_norm.transform(ae_test)
    if_test_n = if_norm.transform(if_test)

    # IF standalone op for the threshold_config payload
    if_op = select_operating_point(if_val, y_val, target_fpr)

    alphas = np.round(np.linspace(0.0, 1.0, 101), 2)
    rows = []
    best = {"macro_f1": -1.0}
    t0 = time.time()
    for a in alphas:
        ev = geometric_ensemble(ae_val_n, if_val_n, a)
        op = select_operating_point(ev, y_val, target_fpr)
        rows.append({
            "alpha": float(a),
            "threshold": op["threshold"],
            "macro_f1": op["macro_f1"],
            "fpr": op["fpr"],
            "tpr": op["tpr"],
            "attack_precision": op["attack_precision"],
            "mode": op["mode"],
        })
        if (op["macro_f1"] > best["macro_f1"]
                or (op["macro_f1"] == best["macro_f1"] and op["tpr"] > best.get("tpr", -1.0))):
            best = {**op, "alpha": float(a)}
    print(f"  swept 101 α × full threshold grid in {time.time()-t0:.1f}s")
    print(f"  selected α={best['alpha']:.2f}  threshold={best['threshold']:.6f}  "
          f"val_macro_f1={best['macro_f1']:.4f}  val_fpr={best['fpr']:.4f}")

    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "ensemble_threshold_sweep.csv", index=False)

    print("[7/7] Test eval + bootstrap CIs + persist")
    ens_test = geometric_ensemble(ae_test_n, if_test_n, best["alpha"])
    y_pred = (ens_test >= best["threshold"]).astype(int)
    metrics = binary_metrics(y_test, y_pred, ens_test)
    metrics.update({
        "alpha": float(best["alpha"]),
        "threshold": float(best["threshold"]),
        "target_fpr": float(target_fpr),
        "threshold_mode": best["mode"],
        "ae_score_method": p1["ae_score_method"],
        "ae_topk": p1["ae_topk"],
        "if_max_samples": p1["if_max_samples"],
    })
    print("  bootstrap (n=1000)...")
    metrics.update(bootstrap_metrics(y_test, y_pred, ens_test, n_boot=1000, seed=SEED))

    print("\n=== TEST METRICS @ target_fpr={:.3f} ===".format(target_fpr))
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"], zero_division=0))
    print(f"  Macro-F1 : {metrics['macro_f1']:.4f}  [{metrics['macro_f1_lo']:.4f}, {metrics['macro_f1_hi']:.4f}]")
    print(f"  FPR      : {metrics['fpr']:.4f}  [{metrics['fpr_lo']:.4f}, {metrics['fpr_hi']:.4f}]")
    print(f"  Att TPR  : {metrics['attack_recall_tpr']:.4f}  [{metrics['attack_recall_tpr_lo']:.4f}, {metrics['attack_recall_tpr_hi']:.4f}]")
    print(f"  Att Prec : {metrics['attack_precision']:.4f}  [{metrics['attack_precision_lo']:.4f}, {metrics['attack_precision_hi']:.4f}]")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}  [{metrics['auc_roc_lo']:.4f}, {metrics['auc_roc_hi']:.4f}]")
    print(f"  AUC-PR   : {metrics['auc_pr']:.4f}  [{metrics['auc_pr_lo']:.4f}, {metrics['auc_pr_hi']:.4f}]")

    # ── persist updated artifacts ──────────────────────────────────────────────
    threshold_config = {
        "target_fpr": float(target_fpr),
        "if_best_threshold": float(if_op["threshold"]),
        "if_threshold_mode": if_op["mode"],
        "if_max_samples": int(p1["if_max_samples"]),
        "if_max_samples_candidates": p1["if_max_samples_candidates"],
        "ensemble_alpha": float(best["alpha"]),
        "ensemble_threshold": float(best["threshold"]),
        "ensemble_threshold_mode": best["mode"],
        "ensemble_method": "geometric_mean",
        "ae_score_method": p1["ae_score_method"],
        "ae_topk": p1["ae_topk"],
    }
    with open(ARTIFACT_DIR / "threshold_config.json", "w") as f:
        json.dump(threshold_config, f, indent=2)

    p1.update({
        "best_threshold": float(best["threshold"]),
        "alpha": float(best["alpha"]),
        "target_fpr": float(target_fpr),
        "ensemble_threshold_mode": best["mode"],
        "if_best_threshold": float(if_op["threshold"]),
        "if_threshold_mode": if_op["mode"],
        "notebook_version": "v6-may2026-fpr01",
    })
    with open(ARTIFACT_DIR / "p1_config.json", "w") as f:
        json.dump(p1, f, indent=2)

    with open(ARTIFACT_DIR / "score_normalizers.json", "w") as f:
        json.dump({
            "ae": {"lo": float(ae_norm.lo_), "hi": float(ae_norm.hi_),
                   "low_q": float(ae_norm.low), "high_q": float(ae_norm.high)},
            "if": {"lo": float(if_norm.lo_), "hi": float(if_norm.hi_),
                   "low_q": float(if_norm.low), "high_q": float(if_norm.high)},
            "ensemble": "geometric_mean",
            "transform_rule": "clip((x - lo) / (hi - lo), 0, 1)",
        }, f, indent=2)

    with open(ARTIFACT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "Label": y_test,
        "Attack": df_test["Attack"].values,
        "ae_score": ae_test,
        "if_score": if_test,
        "ens_score": ens_test,
        "y_pred": y_pred,
    }).to_csv(ARTIFACT_DIR / "test_results.csv", index=False)

    # Single-run matrix only. The canonical lead figure confusion_matrix.png is
    # the headline 3-seed ensemble strict matrix (scripts/in_domain_3seed_ensemble.py);
    # write here under a distinct name so we never clobber it.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Benign", "Attack"], cmap="Blues", ax=ax,
    )
    plt.title(f"Confusion Matrix — single-run (target_fpr={target_fpr:.3f})")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "confusion_matrix_singlerun.png", dpi=150)
    plt.close(fig)

    print("\n✓ artifacts updated.")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.01)
    args = ap.parse_args()
    m = main(args.target_fpr)
    # exit code 0 if both paper bars met; 2 if FPR target failed; 3 if macro-F1 < 0.88
    import sys
    if m["fpr"] > args.target_fpr * 1.01:  # 1% tolerance
        print(f"\n[WARN] achieved FPR {m['fpr']:.4f} > target {args.target_fpr}")
        sys.exit(2)
    if m["macro_f1"] < 0.88:
        print(f"\n[WARN] macro_f1 {m['macro_f1']:.4f} < paper bar 0.88")
        sys.exit(3)
