import os
os.environ['CAUSHAP_LIPSCHITZ_ONLY'] = '1'   # skip CF block; detection+faithfulness identical to matrix
import json
from pathlib import Path
from caushap_nids.experiments.runner import run_single, _load_dataset

OUT = Path('artifacts/lipschitz_robust_recompute.json')
DATASETS = ['5g_nidd', 'edge_iiotset', 'nf_unsw15', 'nf_cic2018']  # fast first, cic2018 (slow) last
SEEDS = [42, 43, 44]
out = json.load(open(OUT)) if OUT.exists() else {}

for ds in DATASETS:
    for s in SEEDS:
        need = [c for c in ['A1_stl_only', 'A2_causal_shap'] if f'{c}/{ds}/{s}' not in out]
        if not need:
            print(f'skip {ds}/{s} (already done)', flush=True)
            continue
        pre = _load_dataset(ds, s, Path('data'))           # load once, reuse for A1+A2
        for cfg in need:
            r = run_single(cfg, ds, s, n_explain=200, verbose=False, _preloaded=pre)
            fa = r.faithfulness or {}
            out[f'{cfg}/{ds}/{s}'] = {
                'status': r.status,
                'lip_x0': fa.get('lipschitz'),          # single-point (validation vs matrix)
                'lip_robust': fa.get('lipschitz_robust'),  # 20-instance average (the real number)
                'n': fa.get('n_explained'),
            }
            json.dump(out, open(OUT, 'w'), indent=2)       # incremental save (resume-safe)
            print(f'done {cfg}/{ds}/{s}: x0={fa.get("lipschitz")} robust={fa.get("lipschitz_robust")}', flush=True)
print('ALL DONE', flush=True)
