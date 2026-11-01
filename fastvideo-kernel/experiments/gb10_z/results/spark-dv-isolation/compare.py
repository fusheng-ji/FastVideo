"""Compare each logical tensor element by raw bits and compute numerical differences."""
import argparse
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--artifact-root', type=Path, required=True)
p.add_argument('--comparison', choices=('control0', 'control1', 'main', 'save-recompute'), required=True)
a = p.parse_args()
root = a.artifact_root
names = ('O', 'M', 'dQ', 'dK', 'STE', 'dV')
pairs = {'control0': ('split', 'split'), 'control1': ('split', 'legacy'),
         'main': ('split', 'default'), 'save-recompute': ('save', 'recompute')}
left, right = pairs[a.comparison]
left_meta = json.loads((root / left / 'metadata.json').read_text())
right_meta = json.loads((root / right / 'metadata.json').read_text())
assert left_meta['status'] == right_meta['status'] == 'completed'
def input_fingerprints(metadata):
    result = {}
    for case in metadata['cases']:
        key = (case['n'], case['seed'])
        if key in result:
            assert result[key] == case['input_sha256']
        result[key] = case['input_sha256']
    assert len(result) == 6
    return result
assert input_fingerprints(left_meta) == input_fingerprints(right_meta)
rows = []
for n in (2112, 31200):
    for seed in (0, 11, 42):
        x = torch.load(root / left / f'{n}-{seed}-0.pt', map_location='cpu', weights_only=True)
        y = torch.load(root / right / f'{n}-{seed}-{1 if a.comparison == "control0" else 0}.pt', map_location='cpu', weights_only=True)
        results = {}
        for name in names:
            u, v = x[name].contiguous().flatten(), y[name].contiguous().flatten()
            assert u.shape == v.shape and u.dtype == v.dtype and torch.isfinite(u).all() and torch.isfinite(v).all()
            different = 0
            max_abs = sum_sq = ref_sq = 0.0
            for start in range(0, u.numel(), 1_000_000):
                aa, bb = u[start:start + 1_000_000], v[start:start + 1_000_000]
                different += int((aa.view(torch.uint8).reshape(-1, aa.element_size()) != bb.view(torch.uint8).reshape(-1, bb.element_size())).any(dim=1).sum())
                ref, val = aa.double(), bb.double()
                delta = val - ref
                max_abs = max(max_abs, float(delta.abs().max()))
                if name == 'dV':
                    sum_sq += float(torch.dot(delta, delta)); ref_sq += float(torch.dot(ref, ref))
            row = {'bitwise_equal': different == 0, 'different_elements': different, 'max_abs_difference': max_abs}
            if name == 'dV':
                row['relative_l2_vs_reference'] = (sum_sq / ref_sq)**0.5 if ref_sq else (0.0 if sum_sq == 0 else None)
            results[name] = row
        rows.append({'n': n, 'seed': seed, 'tensors': results})
report = {'comparison': a.comparison, 'reference': left, 'candidate': right, 'input_fingerprints_equal': True, 'cases': rows}
(root / f'{a.comparison}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
if a.comparison == 'control0' and not all(v['bitwise_equal'] for row in rows for v in row['tensors'].values()):
    raise SystemExit('Control 0 is non-deterministic; stop downstream experiments')
