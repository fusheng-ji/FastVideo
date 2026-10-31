"""Short-budget completeness checks for the v4 comparison tool."""
import importlib.util
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root = Path(__file__).parents[1]
compare = load(root / 'code/chunk_v4_compare.py', 'v4_compare').compare
report = load(Path(__file__).with_name('test_chunk_pair_compare.py'), 'old_fixtures').report


def short(chunk):
    result = report(chunk)
    result['updates'] = result['updates'][:2]
    return result


def test_two_step_pair():
    assert compare(short(0), short(64), steps=2)['status'] == 'passed'


def test_incomplete_pair_rejected():
    assert 'update_count' in compare(short(0), short(64), steps=4)['failures']


def test_disabled_chunk_rejected():
    assert 'chunk_optimizer_selection' in compare(short(0), short(0), steps=2)['failures']


def test_second_step_difference_rejected():
    a, b = short(0), short(64)
    b['updates'][1]['parameter_sha256']['4'] = 'different'
    assert '2:parameter_sha256' in compare(a, b, steps=2)['failures']
