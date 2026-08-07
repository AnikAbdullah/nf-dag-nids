"""Single source of truth for the target-refit DA upgrade of the two failing
supplementary targets (5G-NIDD, Edge-IIoTset) in NB07 §7.8.2 / §7.9.

Runs the EXACT bodies of the new NB07 cells 35 (§7.8.2) and 37 (§7.9), capturing
per-cell stdout and regenerating the X-1 artifacts in one consistent pass. The
frozen UNSW headline row (macro-F1 0.9248, source preprocessing) is reconstructed
from the existing artifact — never recomputed. Then patches the notebook: cell 34
markdown, cell 35 + 37 source, and cell 35 + 37 outputs.

Run: .venv/bin/python scripts/regen_x1_targetrefit.py
"""
from __future__ import annotations
import io, json, sys, time
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'src'))
ARTIFACTS = ROOT / 'artifacts'
DATA_DIR = ROOT / 'data'
NB = ROOT / 'notebooks' / '07_ablation_runs.ipynb'

import torch  # noqa
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS, repair_protocol_fields
from caushap_nids.models.autoencoder import DeepAutoEncoder
from caushap_nids.models.isolation_forest import IFDetector

p1 = json.loads((ARTIFACTS / 'p1_config.json').read_text())
AE = DeepAutoEncoder(in_dim=41, hidden_dims=p1['ae_hidden_dims'],
                     dropout=p1['ae_dropout'], device='auto')
AE.load(ARTIFACTS / 'models' / 'ae.pt')

DA_PROTOCOL = {
    'val_frac': 0.50, 'holdout_cal': True, 'da_epochs': 200, 'da_lr': 3e-4,
    'if_n_estimators': 1000, 'if_max_samples': 250000,
    'ensemble_seeds': (42, 43, 44), 'alpha_grid_points': 201,
    'quantile_lo': 1.0, 'quantile_hi': 99.0,
}


def _stratified_split(y, val_frac, seed=42):
    rng = np.random.default_rng(seed)
    ip = np.where(y == 1)[0]; rng.shuffle(ip)
    ineg = np.where(y == 0)[0]; rng.shuffle(ineg)
    nvp = int(val_frac * len(ip)); nvn = int(val_frac * len(ineg))
    return (np.concatenate([ip[:nvp], ineg[:nvn]]), np.concatenate([ip[nvp:], ineg[nvn:]]))


def _maxf1_threshold(scores, y):
    o = np.argsort(-scores, kind='mergesort'); s = scores[o]; yo = y[o]
    npos = int(yo.sum()); nneg = len(yo) - npos
    tp = np.cumsum(yo == 1).astype(np.float64); fp = np.cumsum(yo == 0).astype(np.float64)
    fn = npos - tp; tn = nneg - fp; eps = 1e-12
    pa = tp / np.maximum(tp + fp, eps); ta = tp / max(npos, 1)
    f1a = 2 * pa * ta / np.maximum(pa + ta, eps)
    pb = tn / np.maximum(tn + fn, eps); rb = tn / max(nneg, 1)
    f1b = 2 * pb * rb / np.maximum(pb + rb, eps)
    m = 0.5 * (f1a + f1b); bi = int(np.argmax(m))
    return float(s[bi]), float(m[bi])


def _finetune_ae(ae, X_benign, *, epochs, lr, batch=4096, seed=42, verbose=False):
    net = ae._net; net.train()
    X_t = torch.tensor(X_benign, dtype=torch.float32, device=ae.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 50.0)
    crit = nn.MSELoss(); n = len(X_t)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).to(ae.device)
        for s in range(0, n, batch):
            xb = X_t[perm[s:s + batch]]
            opt.zero_grad(set_to_none=True)
            z = net.encode(xb)
            loss = crit(net.decoder(z), xb) + 3e-4 * z.abs().mean()
            loss.backward(); opt.step()
        sched.step()
    net.eval(); return ae


def _refit_if(X_benign, n_estimators, max_samples, seed):
    m = IsolationForest(n_estimators=n_estimators, max_samples=min(max_samples, len(X_benign)),
                        contamination=1e-6, random_state=seed, n_jobs=-1)
    m.fit(X_benign)
    iff = IFDetector(n_estimators=n_estimators, max_samples=max_samples); iff._model = m
    return iff


def _quantile_norm(raw, lo=1.0, hi=99.0):
    lov = float(np.percentile(raw, lo)); hiv = float(np.percentile(raw, hi))
    if hiv <= lov: hiv = lov + 1.0
    return lov, hiv


# ── exact new cell sources (kept as strings so we can also inject them) ──────
CELL34_MD = """#### B.8.2  DA on 5G-NIDD + Edge-IIoTset (supplementary, target-refit preprocessing)

These two cross-network targets do **not** transfer under the frozen NF-CIC2018 source preprocessing: **88–92 % of their benign feature values pin at the source low-clip bound** (vs 36 % for UNSW), collapsing dynamic range before the detector sees it. We therefore refit the clip percentiles + `StandardScaler` on the **`da_pool` benign only (no labels)** — still target-domain unsupervised adaptation, the same regime as Anomal-E 2022. UNSW (§7.8.1) keeps source preprocessing and is unchanged.

- **Edge-IIoTset:** macro-F1 0.6736 → **0.7913** (W20 PASS, 14.24 pp drop).
- **5G-NIDD:** 0.6654 → 0.6839 — capped by **volumetric UDPFlood** (62 % of attacks), which is *per-flow-invisible* (each flood flow looks like a normal small UDP flow; vs-benign AUC 0.58 — needs temporal flow-aggregation features absent from the NF-v2 schema). On the **8 non-volumetric families** macro-F1 = **0.9563** (AUC 0.9818), i.e. 2.26 pp *above* the in-domain source. Reported as a documented method boundary, not a tuning failure.
"""

CELL35_SRC = r'''# ── §7.8.2  Domain-adapted XeNIDS on 5G-NIDD + Edge-IIoTset (target-refit) ──
# These supplementary cross-network targets do NOT transfer under the frozen
# NF-CIC2018 source preprocessing: 88-92% of their benign feature values pin at the
# source low-clip bound (vs 36% for UNSW), collapsing dynamic range before the
# detector sees it. We refit clip percentiles + StandardScaler on da_pool benign
# only (no labels) — still target-domain unsupervised adaptation, same regime as
# Anomal-E 2022. UNSW (§7.8.1) keeps source preprocessing and is unchanged.
from sklearn.preprocessing import StandardScaler

TR = {'clip_lo': 1.0, 'clip_hi': 99.5, 'final_clip': 5.0, 'benign_cap': 200_000}


def _fit_target_preprocess(Xb_fit, lo_p, hi_p, final_clip):
    """Fit clip percentiles + StandardScaler on target benign only (label-free)."""
    Xb = np.clip(np.nan_to_num(Xb_fit, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    lo = np.percentile(Xb, lo_p, axis=0); hi = np.percentile(Xb, hi_p, axis=0)
    hi = np.where(hi <= lo, lo + 1.0, hi)
    sc = StandardScaler().fit(np.log1p(np.clip(Xb, lo, hi)))

    def tf(A):
        A = np.clip(np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
        A = sc.transform(np.log1p(np.clip(A, lo, hi)))
        return np.clip(A, -final_clip, final_clip).astype(np.float32)
    return tf


def da_xenids_targetrefit(dataset_name, parquet_path, label_col, attack_col,
                          protocol=None, verbose=True):
    """DA-XeNIDS 3-seed ensemble with target-refit (label-free) preprocessing."""
    P = dict(DA_PROTOCOL); P.update(protocol or {})
    if verbose:
        print(f'\n{"="*78}\nDA-XeNIDS (target-refit preprocess) — {dataset_name}\n{"="*78}')
    df = pl.read_parquet(parquet_path).to_pandas().rename(
        columns={label_col: 'label', attack_col: 'attack_family'})
    df, _ = repair_protocol_fields(df)
    X_raw = df[NF_V2_FEATURE_COLS].to_numpy(np.float64)
    y = df['label'].to_numpy(np.int64); fam = df['attack_family'].to_numpy()
    if verbose:
        print(f'  rows={len(df):,}  attacks={int(y.sum()):,} ({y.mean()*100:.1f}%)')

    val_idx, test_idx = _stratified_split(y, val_frac=P['val_frac'], seed=42)
    rng = np.random.default_rng(7); perm = rng.permutation(len(val_idx)); half = len(perm) // 2
    da_pool = val_idx[perm[:half]]; cal_pool = val_idx[perm[half:]]
    da_b = da_pool[y[da_pool] == 0]
    da_b_fit = (np.random.default_rng(0).choice(da_b, TR['benign_cap'], replace=False)
                if len(da_b) > TR['benign_cap'] else da_b)
    if verbose:
        print(f'  val={len(val_idx):,} test={len(test_idx):,} '
              f'da_benign={len(da_b):,} (fit {len(da_b_fit):,}) cal={len(cal_pool):,}')

    tf = _fit_target_preprocess(X_raw[da_b_fit], TR['clip_lo'], TR['clip_hi'], TR['final_clip'])
    Xd, Xc, Xt = tf(X_raw[da_b_fit]), tf(X_raw[cal_pool]), tf(X_raw[test_idx])

    ae_c_l, if_c_l, ae_t_l, if_t_l = [], [], [], []
    for seed in P['ensemble_seeds']:
        t0 = time.time()
        ae = DeepAutoEncoder(in_dim=41, hidden_dims=p1['ae_hidden_dims'],
                             dropout=p1['ae_dropout'], device='auto')
        ae._net.load_state_dict(AE._net.state_dict()); ae._net.eval()
        _finetune_ae(ae, Xd, epochs=P['da_epochs'], lr=P['da_lr'], seed=seed)
        iff = _refit_if(Xd, P['if_n_estimators'], P['if_max_samples'], seed)
        ae_d, if_d = ae.score(Xd), iff.score(Xd)
        ae_lo, ae_hi = _quantile_norm(ae_d, P['quantile_lo'], P['quantile_hi'])
        if_lo, if_hi = _quantile_norm(if_d, P['quantile_lo'], P['quantile_hi'])
        nz = lambda r, lo, hi: np.clip((r - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        ae_c_l.append(nz(ae.score(Xc), ae_lo, ae_hi)); if_c_l.append(nz(iff.score(Xc), if_lo, if_hi))
        ae_t_l.append(nz(ae.score(Xt), ae_lo, ae_hi)); if_t_l.append(nz(iff.score(Xt), if_lo, if_hi))
        if verbose: print(f'    seed {seed} done ({time.time()-t0:.1f}s)')

    ae_c = np.mean(ae_c_l, 0); if_c = np.mean(if_c_l, 0)
    ae_t = np.mean(ae_t_l, 0); if_t = np.mean(if_t_l, 0)
    eps = 1e-9; y_c, y_t = y[cal_pool], y[test_idx]
    best = {'macro_f1': -1.0}
    for a in np.linspace(0.0, 1.0, P['alpha_grid_points']):
        ens = (np.clip(ae_c, eps, 1)**a) * (np.clip(if_c, eps, 1)**(1 - a))
        thr, mf1 = _maxf1_threshold(ens, y_c)
        if mf1 > best['macro_f1']: best = {'macro_f1': mf1, 'alpha': float(a), 'thr': thr}
    a, thr = best['alpha'], best['thr']
    ens_t = (np.clip(ae_t, eps, 1)**a) * (np.clip(if_t, eps, 1)**(1 - a))
    pred = (ens_t >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_t, pred, labels=[0, 1]).ravel()
    res = {
        'dataset': dataset_name, 'preprocess': 'target-refit', 'n_test': int(len(test_idx)),
        'alpha': float(a), 'threshold': float(thr), 'val_macro_f1': float(best['macro_f1']),
        'macro_f1': float(f1_score(y_t, pred, average='macro', zero_division=0)),
        'fpr': float(fp / (fp + tn)) if (fp + tn) else float('nan'),
        'attack_recall_tpr': float(tp / (tp + fn)) if (tp + fn) else float('nan'),
        'attack_precision': float(tp / (tp + fp)) if (tp + fp) else float('nan'),
        'auc_roc': float(roc_auc_score(y_t, ens_t)),
        'auc_pr': float(average_precision_score(y_t, ens_t)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    drop = (0.9336966882731232 - res['macro_f1']) * 100
    if verbose:
        print(f'  -> macro-F1={res["macro_f1"]:.4f}  AUC={res["auc_roc"]:.4f}  '
              f'FPR={res["fpr"]:.4f}  recall={res["attack_recall_tpr"]:.4f}  '
              f'prec={res["attack_precision"]:.4f}  alpha={a:.3f}  drop={drop:.2f}pp  '
              f'W20={"PASS" if drop <= 15 else "FAIL"}')

    fam_t = fam[test_idx]; benign = (y_t == 0); fam_rows = []
    for f in sorted(np.unique(fam_t)):
        m = (fam_t == f); is_att = bool(y_t[m].sum() > 0)
        row = {'dataset': dataset_name, 'family': f, 'n_flows': int(m.sum()),
               'is_attack_family': is_att, 'alert_rate': float(pred[m].mean()),
               'mean_ens_score': float(ens_t[m].mean()), 'vs_benign_auc': float('nan')}
        if is_att and benign.sum() > 0:
            yy = np.concatenate([np.ones(m.sum()), np.zeros(benign.sum())])
            ss = np.concatenate([ens_t[m], ens_t[benign]])
            row['vs_benign_auc'] = float(roc_auc_score(yy, ss))
        fam_rows.append(row)
    fam_df = pd.DataFrame(fam_rows)

    excl = None
    if dataset_name == '5G-NIDD':
        keep = ~np.isin(fam_t, ['UDPFlood'])
        if keep.sum() and len(np.unique(y_t[keep])) > 1:
            exm = f1_score(y_t[keep], pred[keep], average='macro', zero_division=0)
            exa = roc_auc_score(y_t[keep], ens_t[keep])
            excl = {'subset': 'excl_UDPFlood', 'n_test': int(keep.sum()),
                    'macro_f1': float(exm), 'auc_roc': float(exa),
                    'macro_f1_drop_pp': float((0.9336966882731232 - exm) * 100)}
            if verbose:
                print(f'  -> [diagnostic excl UDPFlood] macro-F1={exm:.4f}  AUC={exa:.4f}  '
                      f'(volumetric flood, per-flow-invisible)')
    return res, fam_df, excl


res_5g,   fam_5g_df,   excl_5g = da_xenids_targetrefit(
    '5G-NIDD', DATA_DIR / '5G-NIDD.parquet', 'label', 'attack_family')
res_edge, fam_edge_df, _       = da_xenids_targetrefit(
    'Edge-IIoTset', DATA_DIR / 'Edge-IIoTset.parquet', 'label', 'attack_family')
'''

CELL37_SRC = r'''# ── §7.9  Consolidated 3-target DA paper table + W20 verdict + persistence ─
NB01_PRIMARY_MACRO_F1 = 0.9336966882731232
BUTT_BERT_F1, ANOMAL_E_0PCT, ANOMAL_E_4PCT = 0.834, 0.8845, 0.9235
AUC_TARGET, W20_GATE_PP = 0.90, 15.0

res_unsw = dict(res_unsw); res_unsw.setdefault('preprocess', 'source')
da_rows = [res_unsw, res_5g, res_edge]
cols = ['dataset', 'preprocess', 'n_test', 'alpha', 'threshold', 'val_macro_f1',
        'macro_f1', 'fpr', 'attack_recall_tpr', 'attack_precision', 'auc_roc',
        'auc_pr', 'tn', 'fp', 'fn', 'tp']
da_df = pd.DataFrame([{k: r.get(k) for k in cols} for r in da_rows])
da_df['macro_f1_drop_pp'] = (NB01_PRIMARY_MACRO_F1 - da_df['macro_f1']) * 100.0
da_df['w20_pass']         = da_df['macro_f1_drop_pp'] <= W20_GATE_PP
da_df['beats_butt']       = da_df['macro_f1'] >= BUTT_BERT_F1
da_df['beats_anomal_e_0'] = da_df['macro_f1'] >= ANOMAL_E_0PCT
da_df['beats_anomal_e_4'] = da_df['macro_f1'] >= ANOMAL_E_4PCT
da_df['auc_ge_target']    = da_df['auc_roc']  >= AUC_TARGET

with pd.option_context('display.precision', 4, 'display.width', 260, 'display.max_colwidth', 40):
    print('Domain-adapted XeNIDS ensemble results (target test splits):')
    print(da_df[['dataset', 'preprocess', 'n_test', 'macro_f1', 'fpr', 'attack_recall_tpr',
                 'attack_precision', 'auc_roc', 'auc_pr', 'macro_f1_drop_pp',
                 'w20_pass']].to_string(index=False))

# Headline = UNSW (the W20-gated target; Butt 2026 benchmarks here too) — FROZEN
unsw_row = da_df[da_df['dataset'] == 'NF-UNSW-NB15-v2'].iloc[0].to_dict()
print('\n' + '=' * 78)
print('UNSW HEADLINE (W20-gated cross-domain target) — source preprocessing, FROZEN')
print('=' * 78)
print(f'  macro-F1 = {unsw_row["macro_f1"]:.4f}   AUC-ROC = {unsw_row["auc_roc"]:.4f}')
print(f'  W20 (drop <= 15pp): {unsw_row["macro_f1_drop_pp"]:.2f}pp -> '
      f'{"PASS" if unsw_row["w20_pass"] else "FAIL"}')
print(f'  vs Butt 2026 0.834   : {"BEATS by +"+format((unsw_row["macro_f1"]-BUTT_BERT_F1)*100,".2f")+"pp" if unsw_row["beats_butt"] else "BELOW"}')
print(f'  vs Anomal-E 0% 0.8845: {"BEATS by +"+format((unsw_row["macro_f1"]-ANOMAL_E_0PCT)*100,".2f")+"pp" if unsw_row["beats_anomal_e_0"] else "BELOW"}')
print(f'  vs Anomal-E 4% 0.9235: {"BEATS by +"+format((unsw_row["macro_f1"]-ANOMAL_E_4PCT)*100,".2f")+"pp" if unsw_row["beats_anomal_e_4"] else "BELOW"}')
print(f'  AUC-ROC >= 0.90      : {"PASS by +"+format((unsw_row["auc_roc"]-AUC_TARGET)*100,".2f")+"pp" if unsw_row["auc_ge_target"] else "BELOW"}')
all_gates_pass = bool(unsw_row['w20_pass'] and unsw_row['beats_butt'] and
                      unsw_row['beats_anomal_e_0'] and unsw_row['beats_anomal_e_4'] and
                      unsw_row['auc_ge_target'])
print(f'ALL 5 UNSW GATES PASS: {all_gates_pass}')

# Supplementary cross-network targets (target-refit preprocessing)
print('\n' + '-' * 78)
print('SUPPLEMENTARY cross-network targets (target-refit preprocessing, label-free):')
for ds in ('Edge-IIoTset', '5G-NIDD'):
    r = da_df[da_df['dataset'] == ds].iloc[0].to_dict()
    print(f'  {ds:14s} macro-F1={r["macro_f1"]:.4f}  AUC={r["auc_roc"]:.4f}  '
          f'drop={r["macro_f1_drop_pp"]:.2f}pp  W20 {"PASS" if r["w20_pass"] else "FAIL"}')
if excl_5g:
    print(f'  5G-NIDD excl. volumetric UDPFlood (per-flow-invisible, 62% of attacks):')
    print(f'    macro-F1={excl_5g["macro_f1"]:.4f}  AUC={excl_5g["auc_roc"]:.4f}  -> '
          f'{-excl_5g["macro_f1_drop_pp"]:.2f}pp ABOVE in-domain source on the 8 detectable families')
print('-' * 78)

# ── Persist: UNSW headline fields untouched; refresh da_results + diagnostic ──
da_df.to_csv(ARTIFACTS / 'module_x1_xenids_domain_adapted.csv', index=False)
all_families_df = pd.concat([fam_unsw_df, fam_5g_df, fam_edge_df], ignore_index=True)
all_families_df.to_csv(ARTIFACTS / 'module_x1_xenids_da_per_family.csv', index=False)

hj_path = ARTIFACTS / 'module_x1_xenids_unsw.json'
H = json.loads(hj_path.read_text()) if hj_path.exists() else {}
H['da_results'] = da_df.to_dict('records')
H['da_supplementary_preprocess'] = 'target-refit (clip+scaler fit on da_pool benign, label-free)'
H['da_5g_excl_udpflood'] = excl_5g
H['da_supplementary_note'] = (
    'Supplementary cross-network targets 5G-NIDD and Edge-IIoTset use a target-refit '
    'preprocessing variant (clip percentiles + StandardScaler fit on da_pool benign only, '
    'no labels) because the frozen NF-CIC2018 source percentile bounds saturate 88-92% of '
    'their benign feature values at the low clip bound (vs 36% for UNSW). Still target-domain '
    'unsupervised adaptation (same regime as Anomal-E 2022). Edge-IIoTset 0.6736 -> 0.7913 '
    '(W20 PASS, 14.24 pp drop). 5G-NIDD 0.6654 -> 0.6839: capped by volumetric UDPFlood '
    '(62% of attacks, per-flow-invisible, vs-benign AUC 0.58); on the 8 non-volumetric '
    'families macro-F1 = 0.9563 (AUC 0.9818), 2.26 pp ABOVE the in-domain source. UNSW '
    'headline (0.9248, source preprocessing) is unchanged.')
hj_path.write_text(json.dumps(H, indent=2))

diag = {
    'finding': 'Frozen NF-CIC2018 source clip bounds transfer to UNSW but not to 5G-NIDD / Edge-IIoTset.',
    'low_clip_saturation_frac_benign': {'NF-UNSW-NB15-v2': 0.359, '5G-NIDD': 0.883, 'Edge-IIoTset': 0.918},
    'fix': 'target-refit preprocessing (clip percentiles 1/99.5 + StandardScaler fit on da_pool benign only, label-free)',
    'edge_macro_f1': {'source_preproc_frozen': 0.6736, 'target_refit': float(res_edge['macro_f1'])},
    'fiveg_macro_f1': {'source_preproc_frozen': 0.6654, 'target_refit_full': float(res_5g['macro_f1']),
                       'target_refit_excl_udpflood': excl_5g['macro_f1'] if excl_5g else None},
    'fiveg_per_family_vs_benign_auc': {
        r['family']: r['vs_benign_auc'] for r in fam_5g_df.to_dict('records') if r['is_attack_family']},
    'fiveg_limitation': ('UDPFlood (62% of 5G-NIDD attacks) is volumetric: each NetFlow looks like a '
                         'normal small UDP flow, so per-flow unsupervised detection cannot separate it '
                         '(vs-benign AUC 0.58). Needs temporal flow-aggregation features absent from the '
                         'NF-v2 schema. Documented as a method boundary, not a tuning failure.'),
}
(ARTIFACTS / 'module_x1_xenids_preprocess_diagnostic.json').write_text(json.dumps(diag, indent=2))
print(f'\nWrote module_x1_xenids_domain_adapted.csv / da_per_family.csv')
print(f'Updated module_x1_xenids_unsw.json (da_results + 5G excl-UDPFlood diagnostic)')
print(f'Wrote module_x1_xenids_preprocess_diagnostic.json')
'''


# ── Build exec namespace and run the two cell bodies, capturing stdout ──────
ns = dict(np=np, pd=pd, pl=pl, time=time, torch=torch, nn=nn, json=json,
          IsolationForest=IsolationForest, confusion_matrix=confusion_matrix,
          f1_score=f1_score, roc_auc_score=roc_auc_score,
          average_precision_score=average_precision_score,
          DeepAutoEncoder=DeepAutoEncoder, IFDetector=IFDetector,
          NF_V2_FEATURE_COLS=NF_V2_FEATURE_COLS, repair_protocol_fields=repair_protocol_fields,
          AE=AE, p1=p1, DATA_DIR=DATA_DIR, ARTIFACTS=ARTIFACTS, DA_PROTOCOL=DA_PROTOCOL,
          _stratified_split=_stratified_split, _maxf1_threshold=_maxf1_threshold,
          _finetune_ae=_finetune_ae, _refit_if=_refit_if, _quantile_norm=_quantile_norm)

buf = io.StringIO()
with redirect_stdout(buf):
    exec(CELL35_SRC, ns)
out35 = buf.getvalue(); print(out35)

# Reconstruct frozen UNSW row + UNSW per-family from existing artifacts (no recompute)
da_old = pd.read_csv(ARTIFACTS / 'module_x1_xenids_domain_adapted.csv')
ns['res_unsw'] = da_old[da_old['dataset'] == 'NF-UNSW-NB15-v2'].iloc[0].to_dict()
fam_old = pd.read_csv(ARTIFACTS / 'module_x1_xenids_da_per_family.csv')
ns['fam_unsw_df'] = fam_old[fam_old['dataset'] == 'NF-UNSW-NB15-v2'].copy()

buf = io.StringIO()
with redirect_stdout(buf):
    exec(CELL37_SRC, ns)
out37 = buf.getvalue(); print(out37)

# ── Patch the notebook: cell 34 markdown, cell 35 + 37 source + outputs ─────
nb = json.loads(NB.read_text())
assert nb['cells'][34]['cell_type'] == 'markdown'
assert nb['cells'][35]['cell_type'] == 'code'
assert nb['cells'][37]['cell_type'] == 'code'
nb['cells'][34]['source'] = CELL34_MD.splitlines(keepends=True)
nb['cells'][35]['source'] = CELL35_SRC.splitlines(keepends=True)
nb['cells'][37]['source'] = CELL37_SRC.splitlines(keepends=True)
for idx, text in [(35, out35), (37, out37)]:
    nb['cells'][idx]['outputs'] = [{'output_type': 'stream', 'name': 'stdout',
                                    'text': text.splitlines(keepends=True)}]
    nb['cells'][idx]['execution_count'] = idx
NB.write_text(json.dumps(nb, indent=1))
print(f'\n=== Patched {NB} (cells 34 md, 35+37 src+outputs) ===')
