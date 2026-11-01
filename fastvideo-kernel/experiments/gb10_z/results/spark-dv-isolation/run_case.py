"""Collect fixed-input QAT outputs from the editable worktree selected by the caller."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from fastvideo_kernel.triton_kernels import attn_qat_train as kernel

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=('split', 'legacy', 'default', 'save', 'recompute'), required=True)
p.add_argument('--artifact-root', type=Path, required=True)
p.add_argument('--repo', type=Path, required=True)
a = p.parse_args()
repo = a.repo.resolve()
source = Path(kernel.__file__).resolve()
assert source.is_relative_to(repo / 'fastvideo-kernel')
metadata = json.loads(importlib.metadata.distribution('fastvideo-kernel').read_text('direct_url.json'))
assert metadata['dir_info']['editable'] and metadata['url'] == 'file://' + str(repo / 'fastvideo-kernel')
assert torch.cuda.get_device_capability() == (12, 1)
assert 'FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV' not in os.environ
os.environ.pop('FASTVIDEO_ATTN_QAT_SM121_FWD_DV', None)
os.environ.pop('FASTVIDEO_ATTN_QAT_SM121_DV_STATS', None)
if a.mode == 'legacy':
    os.environ['FASTVIDEO_ATTN_QAT_SM121_FWD_DV'] = '0'
if a.mode in ('save', 'recompute'):
    os.environ['FASTVIDEO_ATTN_QAT_SM121_DV_STATS'] = a.mode
output = a.artifact_root / a.mode
output.mkdir(parents=True, exist_ok=False)
metadata = {'mode': a.mode, 'import_path': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'cases': [], 'status': 'running'}

def flush():
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

flush()
try:
    for n in (2112, 31200):
        for seed in (0, 11, 42):
            key = f'{n}-{seed}'
            input_file = a.artifact_root / 'inputs' / f'{key}.pt'
            if a.mode == 'split':
                torch.manual_seed(seed)
                source_tensors = [torch.randn((1, 3, n, 128), device='cuda', dtype=torch.bfloat16) for _ in range(4)]
                input_file.parent.mkdir(parents=True, exist_ok=True)
                assert not input_file.exists()
                torch.save([x.cpu() for x in source_tensors], input_file)
                del source_tensors
            cpu = torch.load(input_file, map_location='cpu', weights_only=True)
            original = [x.to('cuda') for x in cpu]
            del cpu
            q, k, v, do = original
            assert not kernel._use_consumer_blackwell_joined_qat_pv(q.device)
            if a.mode != 'split':
                selected = kernel._use_sm121_forward_consistent_dv(q, k, v, False, True, True, False, False, True, False)
                assert selected == (a.mode != 'legacy'), (a.mode, selected)
                if selected:
                    assert kernel._sm121_dv_stats_mode() == ('recompute' if a.mode == 'recompute' else 'save')
            for repeat in range(2 if a.mode == 'split' else 1):
                q1, k1, v1 = [x.clone().requires_grad_(True) for x in (q, k, v)]
                out = kernel.attention(q1, k1, v1, False, 128**-0.5, True, False, True, True, False, True, True, False, False, False)
                ste, m = out.grad_fn.saved_tensors[3:5]
                dq, dk, dv = torch.autograd.grad(out, (q1, k1, v1), do)
                tensors = {name: t.detach().cpu().contiguous() for name, t in zip(('O', 'M', 'dQ', 'dK', 'STE', 'dV'), (out, m, dq, dk, ste, dv))}
                assert all(torch.isfinite(t).all().item() for t in tensors.values()), (a.mode, key, repeat)
                target = output / f'{key}-{repeat}.pt'
                torch.save(tensors, target)
                metadata['cases'].append({'n': n, 'seed': seed, 'repeat': repeat, 'input_sha256': hashlib.sha256(input_file.read_bytes()).hexdigest()})
                flush()
                del tensors, out, ste, m, dq, dk, dv, q1, k1, v1
            del original, q, k, v, do
    metadata['status'] = 'completed'
except BaseException as error:
    metadata.update(status='failed', error_type=type(error).__name__, error=str(error))
    raise
finally:
    flush()
