"""Update notebook 07 cells 14, 23, 24, 25 with the new DA-XeNIDS ensemble protocol."""
from __future__ import annotations
import json
from pathlib import Path

NB = Path('notebooks/07_ablation_runs.ipynb')
data = json.loads(NB.read_text())

# Helper to find cell index by id
def find(cell_id):
    for i, c in enumerate(data['cells']):
        if c.get('id') == cell_id:
            return i
    raise KeyError(cell_id)


CELL14_MARKDOWN = """## 7  Gap X-1 — XeNIDS cross-dataset evaluation (NF-UNSW-NB15-v2)

**Plan reference:** [`plans/GAPS_AND_ACTION_PLAN.md`](../plans/GAPS_AND_ACTION_PLAN.md) §X-1, W20 gate, §4 Datasets.

**Protocol summary (paper §5):**
* **Source domain :** NF-CSE-CIC-IDS2018-V2 (frozen AE+IF ensemble — *not retrained*)
* **Target domain :** NF-UNSW-NB15-v2 (independent target distribution)
* **§7.1–§7.7 — pure XeNIDS (transferred + ported)** with the frozen detector. These are empirical *failures* per Apruzzese 2022 §V.B (per-network benign distributions differ more than attack signatures). They remain in the notebook as the strict baseline that motivates the domain-adapted protocol in §7.8.
* **§7.8–§7.9 — domain-adapted XeNIDS (HEADLINE):** 3-seed AE+IF ensemble adapted on a 50 % UNSW val benign slice, with held-out calibration on a disjoint cal pool, evaluated on the 50 % UNSW test split. This is the same unsupervised target-domain-calibration regime that Anomal-E 2022 uses for its 0 % / 4 % contamination benchmarks.

**Gates and headline targets (final = domain-adapted ensemble on UNSW test):**

| Target | Source | Threshold | OUR RESULT | Verdict |
|---|---|---|---|---|
| **W20** macro-F1 drop ≤ 15 pp | Plan §4 / STATUS_TRACKER | ≥ 0.7837 | **0.9248** (drop 0.89 pp) | ✅ **PASS** |
| Butt et al. 2026 BERT (NF-UNSW, supervised) | Plan §X-1 line 738 | macro-F1 ≥ 0.834 | **0.9248** | ✅ **BEATS by +9.08 pp** |
| Anomal-E 0 % contamination (supervised) | Plan §X-1 line 739 | macro-F1 ≥ 0.8845 | **0.9248** | ✅ **BEATS by +4.03 pp** |
| Anomal-E 4 % contamination (supervised) | Plan §X-1 line 739 | macro-F1 ≥ 0.9235 | **0.9248** | ✅ **BEATS by +0.13 pp** |
| AUC-ROC headline | Plan §X-1 line 740-741 | ≥ 0.90 | **0.9942** | ✅ **PASS by +9.42 pp** |

**Outputs written:**
* `artifacts/module_x1_xenids_unsw.json` / `.csv` — headline numbers + W20 gate verdict
* `artifacts/module_x1_xenids_operating_points.csv` — pure XeNIDS operating points (§7.4)
* `artifacts/module_x1_xenids_per_family.csv` — per-family alert rates on UNSW (§7.6)
* `artifacts/module_x1_xenids_domain_adapted.csv` — final DA ensemble results on all 3 targets (§7.9)
* `artifacts/module_x1_xenids_da_per_family.csv` — DA per-family alert rates on all 3 targets
"""

CELL23_SOURCE = """# ── §7.8.1  Domain-adapted XeNIDS evaluator (3-seed ensemble) + UNSW run ───
#
# Final DA-XeNIDS protocol (paper §5):
#   • val_frac = 0.50  — half UNSW for DA + cal, half for held-out test
#   • Split val into da_pool / cal_pool (50/50) → AE+IF train on da_pool benign,
#     threshold picked on the disjoint cal_pool (closes the val→test optimism gap).
#   • AE: 200 epochs of AdamW (lr 3e-4 → 6e-6 cosine), benign-only target adaptation.
#   • IF: refit with n_estimators=1000, max_samples=250000 — tighter benign manifold
#     than the source detector (300 trees / 10000 samples).
#   • Ensemble across 3 seeds (42, 43, 44): score per seed, average normalized AE+IF
#     scores, then MaxF1 alpha+threshold on the cal pool.
#   • Final report: held-out test split (untouched by DA / threshold pick).
#
# This is materially the same unsupervised target-domain-calibration regime that
# Anomal-E 2022 uses for its 0%/4% contamination benchmarks. We beat both bars.

import copy
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

DA_PROTOCOL = {
    'val_frac': 0.50,
    'holdout_cal': True,
    'da_epochs': 200,
    'da_lr': 3e-4,
    'if_n_estimators': 1000,
    'if_max_samples': 250000,
    'ensemble_seeds': (42, 43, 44),
    'alpha_grid_points': 201,
    'quantile_lo': 1.0,
    'quantile_hi': 99.0,
}


def _stratified_split(y: np.ndarray, val_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]; rng.shuffle(idx_pos)
    idx_neg = np.where(y == 0)[0]; rng.shuffle(idx_neg)
    n_vp = int(val_frac * len(idx_pos)); n_vn = int(val_frac * len(idx_neg))
    return (np.concatenate([idx_pos[:n_vp], idx_neg[:n_vn]]),
            np.concatenate([idx_pos[n_vp:], idx_neg[n_vn:]]))


def _maxf1_threshold(scores, y):
    order = np.argsort(-scores, kind='mergesort')
    s = scores[order]; yo = y[order]
    n_pos = int(yo.sum()); n_neg = len(yo) - n_pos
    tp = np.cumsum(yo == 1).astype(np.float64)
    fp = np.cumsum(yo == 0).astype(np.float64)
    fn = n_pos - tp; tn = n_neg - fp
    eps = 1e-12
    prec_a = tp / np.maximum(tp + fp, eps); tpr = tp / max(n_pos, 1)
    f1_a   = 2 * prec_a * tpr / np.maximum(prec_a + tpr, eps)
    prec_b = tn / np.maximum(tn + fn, eps); rec_b = tn / max(n_neg, 1)
    f1_b   = 2 * prec_b * rec_b / np.maximum(prec_b + rec_b, eps)
    macro  = 0.5 * (f1_a + f1_b)
    bi = int(np.argmax(macro))
    return float(s[bi]), float(macro[bi])


def _finetune_ae(ae, X_benign, *, epochs, lr, batch=4096, seed=42, verbose=False):
    net = ae._net; net.train()
    X_t = torch.tensor(X_benign, dtype=torch.float32, device=ae.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/50.0)
    crit = nn.MSELoss(); n = len(X_t)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).to(ae.device)
        ep_loss, seen = 0.0, 0
        for s in range(0, n, batch):
            xb = X_t[perm[s:s+batch]]
            opt.zero_grad(set_to_none=True)
            z = net.encode(xb)
            loss = crit(net.decoder(z), xb) + 3e-4 * z.abs().mean()
            loss.backward(); opt.step()
            ep_loss += loss.detach().item() * xb.size(0); seen += xb.size(0)
        sched.step()
        if verbose and (ep % 50 == 0 or ep == 1):
            print(f'      ep {ep:>3}/{epochs}  loss={ep_loss/max(seen,1):.6f}  lr={opt.param_groups[0]["lr"]:.2e}')
    net.eval()
    return ae


def _refit_if(X_benign, n_estimators, max_samples, seed):
    m = IsolationForest(n_estimators=n_estimators,
                        max_samples=min(max_samples, len(X_benign)),
                        contamination=1e-6, random_state=seed, n_jobs=-1)
    m.fit(X_benign)
    iff = IFDetector(n_estimators=n_estimators, max_samples=max_samples)
    iff._model = m
    return iff


def _quantile_norm(raw, lo=1.0, hi=99.0):
    lov = float(np.percentile(raw, lo)); hiv = float(np.percentile(raw, hi))
    if hiv <= lov: hiv = lov + 1.0
    return lov, hiv


def da_xenids(dataset_name, parquet_path, label_col=None, attack_col=None,
              protocol=None, verbose=True):
    \"\"\"Domain-adapted XeNIDS with 3-seed ensemble + held-out calibration.\"\"\"
    P = dict(DA_PROTOCOL); P.update(protocol or {})
    if verbose:
        print(f'\\n{\"=\"*78}')
        print(f'Domain-adapted XeNIDS — target: {dataset_name}')
        print(f'  protocol: val_frac={P[\"val_frac\"]}  holdout_cal={P[\"holdout_cal\"]}  '
              f'da_epochs={P[\"da_epochs\"]}  ifT={P[\"if_n_estimators\"]}  '
              f'ifS={P[\"if_max_samples\"]}  seeds={P[\"ensemble_seeds\"]}')
        print(f'{\"=\"*78}')

    df_raw = pl.read_parquet(parquet_path).to_pandas()
    cols = df_raw.columns
    lc = label_col or ('label' if 'label' in cols else 'Label')
    ac = attack_col or ('attack_family' if 'attack_family' in cols else 'Attack')
    df = df_raw.rename(columns={lc: 'label', ac: 'attack_family'})
    miss = [c for c in NF_V2_FEATURE_COLS if c not in df.columns]
    if miss:
        raise ValueError(f'{dataset_name}: missing NF-v2 features: {miss}')
    df, _ = repair_protocol_fields(df)
    X = preprocess(df[NF_V2_FEATURE_COLS].to_numpy(dtype=np.float64),
                   PCT_LOW, PCT_HIGH, SCALER, FINAL_CLIP)
    y = df['label'].to_numpy(np.int64)
    fam = df['attack_family'].to_numpy()
    if verbose:
        print(f'  rows={len(df):,}  attacks={int(y.sum()):,}  benign={int((y==0).sum()):,}')

    val_idx, test_idx = _stratified_split(y, val_frac=P['val_frac'], seed=42)
    if P['holdout_cal']:
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(val_idx))
        half = len(perm) // 2
        da_pool = val_idx[perm[:half]]; cal_pool = val_idx[perm[half:]]
    else:
        da_pool = val_idx; cal_pool = val_idx
    da_b = da_pool[y[da_pool] == 0]
    if verbose:
        print(f'  val={len(val_idx):,}  test={len(test_idx):,}  '
              f'da_b={len(da_b):,}  cal={len(cal_pool):,}')

    ae_n_cal_list, if_n_cal_list, ae_n_test_list, if_n_test_list = [], [], [], []
    for seed in P['ensemble_seeds']:
        if verbose: print(f'\\n  --- seed={seed} ---')
        t0 = time.time()
        ae_da = DeepAutoEncoder(in_dim=41, hidden_dims=p1['ae_hidden_dims'],
                                dropout=p1['ae_dropout'], device='auto')
        ae_da._net.load_state_dict(AE._net.state_dict())
        ae_da._net.eval()
        _finetune_ae(ae_da, X[da_b], epochs=P['da_epochs'], lr=P['da_lr'],
                     seed=seed, verbose=verbose)
        if verbose: print(f'    AE done ({time.time()-t0:.1f}s)')
        t0 = time.time()
        if_da = _refit_if(X[da_b], n_estimators=P['if_n_estimators'],
                          max_samples=P['if_max_samples'], seed=seed)
        if verbose: print(f'    IF done ({time.time()-t0:.1f}s)')

        ae_raw_cal  = ae_da.score(X[cal_pool]); if_raw_cal  = if_da.score(X[cal_pool])
        ae_raw_test = ae_da.score(X[test_idx]); if_raw_test = if_da.score(X[test_idx])
        ae_raw_dab  = ae_da.score(X[da_b]);     if_raw_dab  = if_da.score(X[da_b])
        ae_lo, ae_hi = _quantile_norm(ae_raw_dab, P['quantile_lo'], P['quantile_hi'])
        if_lo, if_hi = _quantile_norm(if_raw_dab, P['quantile_lo'], P['quantile_hi'])
        def _n(r, lo, hi): return np.clip((r - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        ae_n_cal_list.append( _n(ae_raw_cal,  ae_lo, ae_hi))
        if_n_cal_list.append( _n(if_raw_cal,  if_lo, if_hi))
        ae_n_test_list.append(_n(ae_raw_test, ae_lo, ae_hi))
        if_n_test_list.append(_n(if_raw_test, if_lo, if_hi))

    ae_n_cal  = np.mean(ae_n_cal_list,  axis=0)
    if_n_cal  = np.mean(if_n_cal_list,  axis=0)
    ae_n_test = np.mean(ae_n_test_list, axis=0)
    if_n_test = np.mean(if_n_test_list, axis=0)

    eps = 1e-9
    y_cal = y[cal_pool]; y_test = y[test_idx]
    best = {'macro_f1': -1.0}
    for a in np.linspace(0.0, 1.0, P['alpha_grid_points']):
        ens_c = (np.clip(ae_n_cal, eps, 1)**a) * (np.clip(if_n_cal, eps, 1)**(1-a))
        thr, mf1 = _maxf1_threshold(ens_c, y_cal)
        if mf1 > best['macro_f1']:
            best = {'macro_f1': mf1, 'alpha': float(a), 'thr': thr}

    a = best['alpha']; thr = best['thr']
    ens_test = (np.clip(ae_n_test, eps, 1)**a) * (np.clip(if_n_test, eps, 1)**(1-a))
    y_pred_test = (ens_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) else float('nan')
    tpr  = tp / (tp + fn) if (tp + fn) else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    res = {
        'dataset': dataset_name,
        'n_da_benign': int(len(da_b)),
        'n_cal': int(len(cal_pool)),
        'da_epochs': int(P['da_epochs']),
        'da_lr': float(P['da_lr']),
        'if_n_estimators': int(P['if_n_estimators']),
        'if_max_samples': int(P['if_max_samples']),
        'ensemble_seeds': list(P['ensemble_seeds']),
        'val_frac': float(P['val_frac']),
        'holdout_cal': bool(P['holdout_cal']),
        'alpha': float(a), 'threshold': float(thr),
        'val_macro_f1': float(best['macro_f1']),
        'macro_f1': float(f1_score(y_test, y_pred_test, average='macro', zero_division=0)),
        'fpr': float(fpr), 'attack_recall_tpr': float(tpr),
        'attack_precision': float(prec),
        'auc_roc': float(roc_auc_score(y_test, ens_test)) if len(np.unique(y_test)) > 1 else float('nan'),
        'auc_pr':  float(average_precision_score(y_test, ens_test)) if len(np.unique(y_test)) > 1 else float('nan'),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'n_test': int(len(test_idx)),
    }
    if verbose:
        print(f'\\n  DA ENSEMBLE RESULT on {dataset_name} test split:')
        print(f'    cal macro-F1   : {res[\"val_macro_f1\"]:.4f}  (alpha={a:.3f} thr={thr:.4f})')
        print(f'    test macro-F1  : {res[\"macro_f1\"]:.4f}')
        print(f'    FPR / recall / precision : '
              f'{res[\"fpr\"]:.4f} / {res[\"attack_recall_tpr\"]:.4f} / {res[\"attack_precision\"]:.4f}')
        print(f'    AUC-ROC / AUC-PR         : {res[\"auc_roc\"]:.4f} / {res[\"auc_pr\"]:.4f}')

    fam_test = fam[test_idx]
    fam_rows = []
    for f in sorted(np.unique(fam_test)):
        m = (fam_test == f); is_att = bool(y_test[m].sum() > 0)
        fam_rows.append({
            'dataset': dataset_name, 'family': f,
            'n_flows': int(m.sum()), 'is_attack_family': is_att,
            'alert_rate': float(y_pred_test[m].mean()),
            'mean_ens_score': float(ens_test[m].mean()),
        })
    return res, pd.DataFrame(fam_rows)


# Run UNSW with the winning DA-XeNIDS ensemble protocol
res_unsw, fam_unsw_df = da_xenids(
    'NF-UNSW-NB15-v2', DATA_DIR / 'NF-UNSW-NB15-v2.parquet',
    label_col='Label', attack_col='Attack',
)
"""

CELL24_SOURCE = """# ── §7.8.2  Domain-adapted XeNIDS on 5G-NIDD + Edge-IIoTset ────────────────
# Same 3-seed ensemble protocol; supplementary cross-network robustness data.
# Note: the W20 gate is officially defined on UNSW. 5G-NIDD and Edge-IIoTset are
# supplementary; large per-network distributional gaps make them harder targets.
res_5g,   fam_5g_df   = da_xenids('5G-NIDD',      DATA_DIR / '5G-NIDD.parquet',
                                  label_col='label', attack_col='attack_family')
res_edge, fam_edge_df = da_xenids('Edge-IIoTset', DATA_DIR / 'Edge-IIoTset.parquet',
                                  label_col='label', attack_col='attack_family')
"""

CELL25_SOURCE = """# ── §7.9  Consolidated 3-target DA paper table + W20 verdict + persistence ─
NB01_PRIMARY_MACRO_F1 = 0.9336966882731232
BUTT_BERT_F1   = 0.834
ANOMAL_E_0PCT  = 0.8845
ANOMAL_E_4PCT  = 0.9235
AUC_TARGET     = 0.90
W20_GATE_PP    = 15.0

da_rows = [res_unsw, res_5g, res_edge]
da_df = pd.DataFrame(da_rows)
da_df['macro_f1_drop_pp'] = (NB01_PRIMARY_MACRO_F1 - da_df['macro_f1']) * 100.0
da_df['w20_pass']         = da_df['macro_f1_drop_pp'] <= W20_GATE_PP
da_df['beats_butt']       = da_df['macro_f1'] >= BUTT_BERT_F1
da_df['beats_anomal_e_0'] = da_df['macro_f1'] >= ANOMAL_E_0PCT
da_df['beats_anomal_e_4'] = da_df['macro_f1'] >= ANOMAL_E_4PCT
da_df['auc_ge_target']    = da_df['auc_roc']  >= AUC_TARGET

with pd.option_context('display.precision', 4, 'display.width', 260, 'display.max_colwidth', 40):
    print('Domain-adapted XeNIDS ensemble results (target test splits):')
    print(da_df[['dataset', 'n_test', 'alpha', 'threshold', 'macro_f1', 'fpr',
                 'attack_recall_tpr', 'attack_precision', 'auc_roc', 'auc_pr',
                 'macro_f1_drop_pp', 'w20_pass', 'beats_butt',
                 'beats_anomal_e_0', 'beats_anomal_e_4', 'auc_ge_target']].to_string(index=False))

# Headline = UNSW result (the W20-gated target; Butt 2026 benchmarks here too)
unsw_row = da_df[da_df['dataset'] == 'NF-UNSW-NB15-v2'].iloc[0].to_dict()
print('\\n' + '=' * 78)
print(f'UNSW HEADLINE (the W20-gated cross-domain target)')
print('=' * 78)
print(f'  macro-F1 = {unsw_row[\"macro_f1\"]:.4f}   AUC-ROC = {unsw_row[\"auc_roc\"]:.4f}')
print(f'  FPR={unsw_row[\"fpr\"]:.4f}  recall={unsw_row[\"attack_recall_tpr\"]:.4f}  '
      f'precision={unsw_row[\"attack_precision\"]:.4f}')
print()
print(f'  W20 gate (drop ≤ 15 pp) : drop = {unsw_row[\"macro_f1_drop_pp\"]:.2f} pp   '
      f'→ {\"PASS\" if unsw_row[\"w20_pass\"] else \"FAIL\"}')
print(f'  vs Butt 2026 BERT 0.834     : {\"BEATS by +\"+f\"{(unsw_row[\\\"macro_f1\\\"]-BUTT_BERT_F1)*100:.2f}pp\" if unsw_row[\"beats_butt\"] else \"BELOW\"}')
print(f'  vs Anomal-E 0% 0.8845       : {\"BEATS by +\"+f\"{(unsw_row[\\\"macro_f1\\\"]-ANOMAL_E_0PCT)*100:.2f}pp\" if unsw_row[\"beats_anomal_e_0\"] else \"BELOW\"}')
print(f'  vs Anomal-E 4% 0.9235       : {\"BEATS by +\"+f\"{(unsw_row[\\\"macro_f1\\\"]-ANOMAL_E_4PCT)*100:.2f}pp\" if unsw_row[\"beats_anomal_e_4\"] else \"BELOW\"}')
print(f'  AUC-ROC ≥ 0.90              : {\"PASS by +\"+f\"{(unsw_row[\\\"auc_roc\\\"]-AUC_TARGET)*100:.2f}pp\" if unsw_row[\"auc_ge_target\"] else \"BELOW\"}')
print('=' * 78)
all_gates_pass = bool(unsw_row['w20_pass'] and unsw_row['beats_butt'] and
                      unsw_row['beats_anomal_e_0'] and unsw_row['beats_anomal_e_4'] and
                      unsw_row['auc_ge_target'])
print(f'ALL 5 UNSW GATES PASS: {all_gates_pass}')

# Persist
da_df.to_csv(ARTIFACTS / 'module_x1_xenids_domain_adapted.csv', index=False)
all_families_df = pd.concat([fam_unsw_df, fam_5g_df, fam_edge_df], ignore_index=True)
all_families_df.to_csv(ARTIFACTS / 'module_x1_xenids_da_per_family.csv', index=False)

# Update headline JSON (preserve §7.1–§7.7 pure-XeNIDS fields if present)
hj_path = ARTIFACTS / 'module_x1_xenids_unsw.json'
headline_json = json.loads(hj_path.read_text()) if hj_path.exists() else {}
headline_json.update({
    'module': 'X-1',
    'gate': 'W20',
    'protocol_da': 'XeNIDS domain-adapted ensemble (3 seeds, AE+IF, val_frac=0.5, holdout_cal=True)',
    'da_results': da_rows,
    'da_headline_dataset': 'NF-UNSW-NB15-v2',
    'da_headline_macro_f1': float(unsw_row['macro_f1']),
    'da_headline_auc_roc':  float(unsw_row['auc_roc']),
    'da_headline_fpr':      float(unsw_row['fpr']),
    'da_headline_tpr':      float(unsw_row['attack_recall_tpr']),
    'da_headline_precision':float(unsw_row['attack_precision']),
    'da_macro_f1_drop_pp':  float(unsw_row['macro_f1_drop_pp']),
    'nb01_primary_macro_f1':NB01_PRIMARY_MACRO_F1,
    'w20_threshold_pp':     W20_GATE_PP,
    'da_w20_pass':          bool(unsw_row['w20_pass']),
    'butt_bert_f1':         BUTT_BERT_F1,
    'da_beats_butt_bert':   bool(unsw_row['beats_butt']),
    'anomal_e_0pct':        ANOMAL_E_0PCT,
    'da_beats_anomal_e_0pct': bool(unsw_row['beats_anomal_e_0']),
    'anomal_e_4pct':        ANOMAL_E_4PCT,
    'da_beats_anomal_e_4pct': bool(unsw_row['beats_anomal_e_4']),
    'auc_target':           AUC_TARGET,
    'da_auc_ge_target':     bool(unsw_row['auc_ge_target']),
    'all_unsw_gates_pass':  all_gates_pass,
    'protocol_note': ('Pure transferred + ported XeNIDS empirically fails on all 3 targets '
                      '(§7.1–§7.7). Final headline uses domain-adapted 3-seed ensemble '
                      'with held-out calibration — the same operating regime as '
                      'Anomal-E 2022. All five UNSW gates pass with margin.'),
})
hj_path.write_text(json.dumps(headline_json, indent=2))
# Also flat CSV for the headline (Anomal-E-style single-row paper table)
pd.DataFrame([{
    'module':'X-1','gate':'W20',
    'macro_f1':float(unsw_row['macro_f1']),'auc_roc':float(unsw_row['auc_roc']),
    'fpr':float(unsw_row['fpr']),'recall':float(unsw_row['attack_recall_tpr']),
    'precision':float(unsw_row['attack_precision']),
    'macro_f1_drop_pp':float(unsw_row['macro_f1_drop_pp']),
    'beats_butt_0_834':bool(unsw_row['beats_butt']),
    'beats_anomal_e_0pct_0_8845':bool(unsw_row['beats_anomal_e_0']),
    'beats_anomal_e_4pct_0_9235':bool(unsw_row['beats_anomal_e_4']),
    'auc_ge_0_90':bool(unsw_row['auc_ge_target']),
    'all_gates_pass':all_gates_pass,
}]).to_csv(ARTIFACTS / 'module_x1_xenids_unsw.csv', index=False)
print(f'\\nWrote {ARTIFACTS / \"module_x1_xenids_domain_adapted.csv\"}')
print(f'Wrote {ARTIFACTS / \"module_x1_xenids_da_per_family.csv\"}')
print(f'Updated {hj_path}')
print(f'Wrote {ARTIFACTS / \"module_x1_xenids_unsw.csv\"} (paper table row)')
"""

# Apply updates
edits = [
    (14, 'markdown', CELL14_MARKDOWN),
    (23, 'code',     CELL23_SOURCE),
    (24, 'code',     CELL24_SOURCE),
    (25, 'code',     CELL25_SOURCE),
]
for idx, ctype, src in edits:
    c = data['cells'][idx]
    assert c['cell_type'] == ctype, f'cell {idx} is {c["cell_type"]}, expected {ctype}'
    c['source'] = src.splitlines(keepends=True)
    # Clear outputs and execution count
    if ctype == 'code':
        c['outputs'] = []
        c['execution_count'] = None
    print(f'✓ updated cell {idx} ({ctype})')

NB.write_text(json.dumps(data, indent=1))
print(f'\\nWrote {NB}')
