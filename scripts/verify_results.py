"""Verify archived source provenance and numerical summaries without rerunning experiments."""
import hashlib,json,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
for folder in sorted((root/'results').glob('round[0-9][0-9]')):
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()==digest,(folder,name)
rows=json.loads((root/'results/round05/raw.json').read_text())
summary=json.loads((root/'results/round05/summary.json').read_text())
assert len(rows)==144
for item in summary:
    selected=[r for r in rows if all(r[k]==item[k] for k in ['resolution','channels','method'])]
    assert len(selected)==3
    assert all(len(r['samples_ms'])==11 for r in selected)
    assert math.isclose(item['median_of_seed_medians_ms'],float(np.median([r['median_ms'] for r in selected])),rel_tol=1e-12)
    assert item['max_error']==max(r['max_error_weighted_mps'] for r in selected)
print('Verified five source snapshots and final raw-result summaries.')
