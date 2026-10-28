"""CPU checks for explicit path selection and manifest rejection."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from portable import parse_args, digest, verify_sources


def test_run_arguments_are_not_consumed_by_wrapper(tmp_path):
    args = parse_args(['run', '--source-manifest', str(tmp_path / 'source.json'),
                      '--script', 'control_train.py', '--lock', str(tmp_path / 'a'),
                      '--lock', str(tmp_path / 'b'), '--', '--seed', '41001'])
    assert args.arguments == ['--seed', '41001']


def test_missing_inputs_rejected(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(['prepare', '--source-manifest', str(tmp_path / 'source.json')])


def test_two_distinct_locks_required(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(['run', '--source-manifest', str(tmp_path / 'source.json'),
                    '--script', 'fa2_replay.py', '--lock', 'same', '--lock', 'same'])


def test_hash_match_and_changed_source(tmp_path):
    source = tmp_path / 'module.py'
    source.write_text('pass\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(acceptance_scope='gb10-z-portable-calibration-v1',
                                        sources={str(source): digest(source)}, inputs={})))
    verify_sources(manifest)
    source.write_text('changed\n')
    with pytest.raises(ValueError, match='fingerprint'):
        verify_sources(manifest)


def test_old_domain_rejected(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(acceptance_scope='old', sources={})))
    with pytest.raises(ValueError, match='manifest'):
        verify_sources(manifest)


def test_private_entrypoint_imports_are_self_contained():
    import importlib
    import torch
    for name in ('probe_model', 'compare_nonstrict', 'control_train', 'report_calibration', 'fa2_replay'):
        importlib.import_module(name)
    assert not torch.cuda.is_initialized()
