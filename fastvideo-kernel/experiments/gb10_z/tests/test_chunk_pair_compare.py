"""Coverage and mismatch checks for paired formal-entrypoint reports."""
import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('chunk_pair_compare', Path(__file__).parents[1] / 'code/chunk_pair_compare.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(chunk=0):
    hashes = {str(i): 'a' * 64 for i in range(825)}
    return {'optimizer_chunk_size': chunk, 'optimizer_class': '_ChunkedForeachAdamW' if chunk else 'AdamW', 'status': 'completed', 'parameter_tensors': 825,
            'initial_parameter_sha256': hashes,
            'initial_method_rng_sha256': {'0': 'x'}, 'initial_cpu_rng_sha256': {'0': 'y'},
            'initial_cuda_rng_sha256': {'0': 'z'},
            'updates': [{'step': i, 'parameter_sha256': hashes.copy(), 'method_rng_sha256': {'0': str(i)},
                         'metrics': {'total_loss': 1., 'finetune_loss': 1., 'step_time_sec': 10.}}
                        for i in range(1, 5)],
            'tracker_metrics': {str(i): {'grad_norm/student': .5} for i in range(1, 5)}}


def test_equal_and_timing_is_not_arithmetic():
    a = report(); b = report(64)
    b['updates'][0]['metrics']['step_time_sec'] = 11.
    assert module.compare(a, b)['status'] == 'passed'


def test_parameter_mismatch_fails():
    a = report(); b = report(64)
    b['updates'][1]['parameter_sha256']['0'] = 'b' * 64
    assert '2:parameter_sha256' in module.compare(a, b)['failures']


def test_missing_gradient_norm_fails():
    a = report(); b = report(64)
    del b['tracker_metrics']['3']
    assert '3:gradient_norm' in module.compare(a, b)['failures']


def test_partial_matrix_fails():
    a = report(); b = report(64)
    b['updates'].pop()
    assert 'update_count' in module.compare(a, b)['failures']


def test_rng_mismatch_fails():
    a = report(); b = report(64)
    b['initial_method_rng_sha256'] = {'0': 'other'}
    assert 'initial_method_rng_sha256' in module.compare(a, b)['failures']


def test_disabled_chunk_cannot_pass():
    assert 'chunk_optimizer_selection' in module.compare(report(), report())['failures']
