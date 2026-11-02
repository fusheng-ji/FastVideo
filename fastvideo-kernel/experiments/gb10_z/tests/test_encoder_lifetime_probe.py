"""CPU checks for the encoder lifetime probe: its patches, modes and referrer reports, without CUDA or FastVideo."""

import gc
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import encoder_lifetime_probe as elp

STUB_SOURCE = '''
import gc
import torch


class TextEncoderLoader:

    def load(self, path, args):
        module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        if CYCLE:
            module.bound_forward = module.forward
        return module


def _release_encoder_memory():
    gc.collect()


def encode_negative_prompt(tc, *, prompt, device, dtype):
    loader = TextEncoderLoader()
    text_encoder = loader.load("text_encoder", None).to(device).eval()
    with torch.no_grad():
        outputs = text_encoder(torch.ones(1, 8, device=device))
        embeds = outputs.to(dtype)
    del text_encoder, outputs
    _release_encoder_memory()
    return embeds, embeds
'''


@pytest.fixture
def stub_modules(monkeypatch):

    def install(cycle):
        for name in ('fastvideo', 'fastvideo.train', 'fastvideo.train.models', 'fastvideo.train.models.wan',
                     'fastvideo.train.utils'):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        negative_prompt = types.ModuleType('fastvideo.train.utils.negative_prompt')
        negative_prompt.CYCLE = cycle
        exec(STUB_SOURCE, negative_prompt.__dict__)
        monkeypatch.setitem(sys.modules, negative_prompt.__name__, negative_prompt)
        wan = types.ModuleType('fastvideo.train.models.wan.wan')
        wan.encode_negative_prompt = negative_prompt.encode_negative_prompt
        monkeypatch.setitem(sys.modules, wan.__name__, wan)
        return wan

    yield install
    gc.enable()


def run(stub_modules, mode, cycle):
    wan = stub_modules(cycle)
    gc.collect()
    probe = elp.LifetimeProbe(None, [])
    report = {}
    elp._install(probe, mode, report)
    with pytest.raises(elp.ProbeDone):
        wan.encode_negative_prompt(None, prompt='x', device=torch.device('cpu'), dtype=torch.float32)
    return probe, {c['name']: c['live_tensors'] for c in probe.checkpoints}


@pytest.mark.parametrize('mode', elp.MODES)
def test_every_mode_records_the_checkpoints_and_stops(stub_modules, mode):
    probe, live = run(stub_modules, mode, cycle=True)
    expected = ['after_load', 'after_forward', 'after_del_inside_function', 'after_return', 'after_gc_collect',
                'after_empty_cache']
    if mode == 'fixed':
        expected.insert(3, 'after_release_helper_inside_function')
    assert [c['name'] for c in probe.checkpoints] == expected
    assert probe.checkpoints[0]['tracked_tensors'] == 4
    assert live['after_gc_collect'] == 0


@pytest.mark.parametrize('mode', ('natural', 'gc-disabled', 'cycle'))
def test_a_cycle_survives_return_and_dies_at_the_collection(stub_modules, mode):
    probe, live = run(stub_modules, mode, cycle=True)
    assert live['after_del_inside_function'] == 4
    assert live['after_return'] == 4
    assert live['after_gc_collect'] == 0
    tree = probe.referrers['after_return']['tree']
    assert tree[0]['qualname'] == 'Sequential.forward'
    assert tree[0]['referred_by'][0]['keys'] == ['bound_forward']


def test_gc_disabled_mode_keeps_the_collector_off_until_the_collection(stub_modules):
    probe, _ = run(stub_modules, 'gc-disabled', cycle=True)
    enabled = {c['name']: c['gc_enabled'] for c in probe.checkpoints}
    assert not enabled['after_load'] and not enabled['after_return']
    assert enabled['after_gc_collect']


def test_cycle_mode_finds_the_module_in_the_garbage(stub_modules):
    probe, _ = run(stub_modules, 'cycle', cycle=True)
    assert probe.cycle['encoder_module_in_garbage']
    assert [r['qualname'] for r in probe.cycle['garbage_referrers_of_module']] == ['Sequential.forward']
    assert gc.garbage == []


def test_fixed_mode_releases_inside_the_function(stub_modules):
    _, live = run(stub_modules, 'fixed', cycle=True)
    assert live['after_del_inside_function'] == 4
    assert live['after_release_helper_inside_function'] == 0


def test_without_a_cycle_del_is_enough(stub_modules):
    probe, live = run(stub_modules, 'natural', cycle=False)
    assert live['after_del_inside_function'] == 0
    assert 'after_return' not in probe.referrers


def test_an_external_reference_is_named(stub_modules):
    wan = stub_modules(False)
    negative_prompt = sys.modules['fastvideo.train.utils.negative_prompt']
    kept = {}
    original = negative_prompt.TextEncoderLoader.load

    def load_and_keep(self, path, args):
        kept['encoder'] = original(self, path, args)
        return kept['encoder']

    negative_prompt.TextEncoderLoader.load = load_and_keep
    probe = elp.LifetimeProbe(None, [])
    elp._install(probe, 'natural', {})
    with pytest.raises(elp.ProbeDone):
        wan.encode_negative_prompt(None, prompt='x', device=torch.device('cpu'), dtype=torch.float32)
    live = {c['name']: c['live_tensors'] for c in probe.checkpoints}
    assert live['after_gc_collect'] == 4
    assert probe.referrers['after_gc_collect']['tree'][0]['keys'] == ['encoder']
    kept.clear()
