"""Synthetic local-spill and whole-layer fallback checks; separate fresh attempts."""
import argparse
from contextlib import ExitStack, nullcontext
import fcntl
import json
from pathlib import Path

from spark_measure import digest, qat, sha_tensor, CODE, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, action='append', required=True)
    parser.add_argument('--scope', choices=['one-tile', 'all-tiles'], required=True)
    a = parser.parse_args()
    if a.output.exists() or len(set(a.lock)) != 2:
        parser.error('Require a fresh output directory and two distinct locks')
    a.output.mkdir(parents=True)
    report = dict(domain='spark-new-replication-v1', kind='overflow', scope=a.scope,
                  status='started', source_sha256=digest(__file__),
                  kernel_sha256=digest(REPO / 'fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py'), rows=[])
    try:
        with ExitStack() as stack:
            for path in a.lock:
                handle = stack.enter_context(path.open('a+'))
                fcntl.flock(handle, fcntl.LOCK_EX)
            import torch
            import portable
            portable.configure()
            from fastvideo_kernel.triton_kernels import attn_qat_train as kernel
            from local_policy import kernel_policy, statistics_metadata
            torch.manual_seed(1000)
            torch.use_deterministic_algorithms(True, warn_only=False)
            assert kernel._sm121_dv_stats_mode() == 'save'
            shape = (1, 1, 8192, 128)
            q = torch.zeros(shape, device='cuda', dtype=torch.bfloat16)
            q[:, :, :32 if a.scope == 'one-tile' else 8192] = .01
            values = torch.logspace(-1.7, 2.7, 256, device='cuda').repeat_interleave(32)
            k = values[None, None, :, None].expand(shape).to(torch.bfloat16).contiguous()
            v = torch.randn_like(q)
            do = torch.randn_like(q)
            expected = None
            for policy in ['dense', 'Z']:
                leaves = [x.clone().requires_grad_() for x in (q, k, v)]
                with kernel_policy(kernel, 'Z') if policy == 'Z' else nullcontext():
                    out = qat(kernel, *leaves)
                    metadata = statistics_metadata(out.grad_fn)
                    grads = torch.autograd.grad(out, leaves, do)
                    torch.cuda.synchronize()
                    assert all(torch.isfinite(t).all().item() for t in (out, *grads)), 'Nonfinite overflow result'
                    hashes = {name: sha_tensor(t, torch) for name, t in zip(['O', 'dQ', 'dK', 'dV'], [out, *grads])}
                    report['rows'].append(dict(policy=policy, statistics=metadata, tensor_sha256=hashes))
                    if policy == 'dense':
                        expected = hashes
                    else:
                        assert hashes == expected, 'Overflow route differs from dense'
                        assert metadata['attempted_spill_tile_count'] > 0, 'No overflow was generated'
                        if a.scope == 'one-tile':
                            assert metadata['local_replay'] and metadata['spill_tile_count'] == 1, 'Expected exactly one spill tile'
                        else:
                            assert metadata['overflow_fallback'] and metadata['format'] == 'dense', 'Expected safe whole-layer fallback'
                    del out, grads, leaves
            report['status'] = 'completed'
            report['bitwise_equal'] = True
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        raise
    finally:
        (a.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
