"""Patch the captured cell outputs from run_module_x1_da.py into notebook 07."""
from __future__ import annotations
import json
from pathlib import Path

NB = Path('notebooks/07_ablation_runs.ipynb')
OUT = Path('scripts/_nb07_cell_outputs.json')

data = json.loads(NB.read_text())
captured = json.loads(OUT.read_text())

# Map index → captured key
for idx_str, text in captured.items():
    idx = int(idx_str)
    cell = data['cells'][idx]
    assert cell['cell_type'] == 'code', f'cell {idx} not code'
    # Replace outputs with a single stream output
    cell['outputs'] = [{
        'output_type': 'stream',
        'name': 'stdout',
        'text': text.splitlines(keepends=True),
    }]
    cell['execution_count'] = idx  # arbitrary monotonic count
    print(f'✓ patched cell {idx} outputs ({len(text)} chars)')

NB.write_text(json.dumps(data, indent=1))
print(f'\nWrote {NB}')
