"""Recompute robust Lipschitz across all 12 A1/A2 cells with quantile=0.95.

Triggered IF the seed43 probe shows quantile=0.95 fixes the regression. Uses the
same protocol as recompute_lipschitz_robust.py but with the new quantile estimator
so a single bursty perturbation doesn't dominate the max.

Output: artifacts/lipschitz_robust_q95_recompute.json
"""
import os
os.environ['CAUSHAP_LIPSCHITZ_ONLY'] = '1'
os.environ['CAUSHAP_LIPSCHITZ_ROBUST_ALL'] = '1'
os.environ['CAUSHAP_LIPSCHITZ_QUANTILE'] = '0.95'

import json
from pathlib import Path
from caushap_nids.experiments.runner import run_single, _load_dataset

OUT = Path('artifacts/lipschitz_robust_q95_recompute.json')
DATASETS = ['5g_nidd', 'edge_iiotset', 'nf_unsw15', 'nf_cic2018']  # fast first
SEEDS = [42, 43, 44]
out = json.load(open(OUT)) if OUT.exists() else {}

for ds in DATASETS:
    for s in SEEDS:
        need = [c for c in ['A1_stl_only', 'A2_causal_shap'] if f'{c}/{ds}/{s}' not in out]
        if not need:
            print(f'skip {ds}/{s} (already done)', flush=True)
            continue
        pre = _load_dataset(ds, s, Path('data'))
        for cfg in need:
            r = run_single(cfg, ds, s, n_explain=200, verbose=False, _preloaded=pre)
            fa = r.faithfulness or {}
            out[f'{cfg}/{ds}/{s}'] = {
                'status': r.status,
                'lip_x0': fa.get('lipschitz'),
                'lip_robust': fa.get('lipschitz_robust'),
                'per_instance': fa.get('lipschitz_per_instance', []),
                'quantile_mode': fa.get('lipschitz_quantile_mode', 0.95),
                'n': fa.get('n_explained'),
            }
            json.dump(out, open(OUT, 'w'), indent=2)
            print(f'done {cfg}/{ds}/{s}: x0={fa.get("lipschitz")} robust={fa.get("lipschitz_robust")}',
                  flush=True)
print('ALL DONE', flush=True)
