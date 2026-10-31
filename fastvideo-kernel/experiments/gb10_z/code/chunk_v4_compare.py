"""Strict comparisons of formal-entrypoint chunk runs; timing is not an equality field."""
import argparse
import json
from pathlib import Path


def compare(a, b, steps=2):
    failures = []
    if a.get('optimizer_chunk_size') != 0 or a.get('optimizer_class') != 'AdamW':
        failures.append('dense_optimizer_selection')
    if b.get('optimizer_chunk_size') != 64 or b.get('optimizer_class') != '_ChunkedForeachAdamW':
        failures.append('chunk_optimizer_selection')
    for key in ('status', 'parameter_tensors', 'initial_parameter_sha256',
                'initial_method_rng_sha256', 'initial_cpu_rng_sha256', 'initial_cuda_rng_sha256'):
        if a.get(key) != b.get(key) or key not in a or key not in b:
            failures.append(key)
    if a.get('status') != 'completed' or b.get('status') != 'completed':
        failures.append('incomplete')
    if a.get('parameter_tensors') != 825 or b.get('parameter_tensors') != 825:
        failures.append('parameter_coverage')
    if len(a.get('updates', [])) != steps or len(b.get('updates', [])) != steps:
        failures.append('update_count')
    for x, y in zip(a.get('updates', []), b.get('updates', [])):
        step = str(x['step'])
        for key in ('step', 'parameter_sha256', 'method_rng_sha256'):
            if key not in x or key not in y or x[key] != y[key]:
                failures.append(f'{step}:{key}')
        if len(x.get('parameter_sha256', {})) != 825 or len(y.get('parameter_sha256', {})) != 825:
            failures.append(f'{step}:hash_coverage')
        for key in ('total_loss', 'finetune_loss'):
            if x['metrics'].get(key) != y['metrics'].get(key) or key not in x['metrics']:
                failures.append(f'{step}:{key}')
        gx = a.get('tracker_metrics', {}).get(step)
        gy = b.get('tracker_metrics', {}).get(step)
        if not gx or gx != gy:
            failures.append(f'{step}:gradient_norm')
    return {'status': 'failed' if failures else 'passed', 'failures': failures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('dense', type=Path)
    p.add_argument('chunk', type=Path)
    a = p.parse_args()
    result = compare(json.loads(a.dense.read_text()), json.loads(a.chunk.read_text()), steps=a.steps)
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'passed')


if __name__ == '__main__':
    main()
