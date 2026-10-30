"""Source-only compatibility gate; no Torch import, model loading or GPU calls."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

KERNEL = 'fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py'
FA2 = 'fastvideo/attention/utils/flash_attn_default.py'
REQUIRED = ('_sm121_forward_consistent_dv_enabled', '_sm121_dv_stats_mode',
            '_sm121_dv_stats_shape', '_attn_bwd_dv_forward_weights')


def inspect_source(source):
    tree = ast.parse(source)
    definitions = {n.name: n.lineno for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    saves = [{'line': n.lineno, 'arguments': [ast.unparse(a) for a in n.args]}
             for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'save_for_backward']
    return {'sha256': hashlib.sha256(source.encode()).hexdigest(),
            'required_definitions': {name: definitions.get(name) for name in REQUIRED},
            'context_saves': saves, 'contains_dv_maxima': 'dv_maxima' in source}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite; use a fresh output file'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(a.repo), *args], text=True)
    baseline = 'cf6327b7f4d8f1326717f5be9ef9da74fb067560'
    target = 'ce33886087f5aa3db0283c90e993adcba29259e3'
    current = inspect_source(git('show', target+':'+KERNEL))
    previous = inspect_source(git('show', baseline+':'+KERNEL))
    fa2 = git('show', target+':'+FA2)
    calls = []
    for node in ast.walk(ast.parse(fa2)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_fa2_backward':
            calls.append({'line': node.lineno, 'deterministic': next((ast.unparse(k.value) for k in node.keywords if k.arg=='deterministic'), None)})
    available = all(current['required_definitions'].values()) and current['contains_dv_maxima']
    result = {'domain': 'spark-production-chunk-preflight-v1', 'target_commit': target,
              'historical_reference_commit': baseline, 'status': 'passed' if available else 'blocked',
              'basis': 'source compatibility, not a GPU runtime failure',
              'target_kernel': current, 'historical_kernel': previous,
              'fa2_backward_calls': calls,
              'fa2_global_determinism_policy_present': 'are_deterministic_algorithms_enabled' in fa2,
              'historical_experiment_scripts_present': bool(git('ls-tree','-r','--name-only',target,'fastvideo-kernel/experiments/gb10_z').strip()),
              'environment': {name: importlib.metadata.version(name) for name in ('torch','triton','flash-attn','numpy')},
              'gpu_training_runs': 0, 'services_changed': False, 'production_patches_applied': False,
              'next_action': 'Obtain an explicitly approved baseline with GB10 saved dense statistics and the FA2 determinism policy; do not transplant automatically'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])


if __name__ == '__main__':
    main()
