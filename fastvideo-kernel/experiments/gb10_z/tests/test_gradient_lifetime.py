"""Exercise the complete replay loop on CPU; no native FA2 or CUDA work."""
import gc
import importlib.util
import pytest
from pathlib import Path
import sys
import types
import weakref

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import fa2_replay as replay
from fa2_extract import extract
from test_recovery import capture


@pytest.mark.parametrize('frozen_control', [False])
def test_every_round_releases_all_gradient_buffers(tmp_path, monkeypatch, frozen_control):
    target = replay
    source = tmp_path / 'source.pt'
    torch.save(capture(), source)
    extract(source, tmp_path / 'input')
    module = types.ModuleType('flash_attn.flash_attn_interface')
    calls = []

    def native(*args, **kwargs):
        calls.append(kwargs['deterministic'])
        for value in args[6:9]:
            value.zero_()

    module._flash_attn_backward = native
    monkeypatch.setitem(sys.modules, 'flash_attn.flash_attn_interface', module)
    for name in ('init', 'reset_peak_memory_stats', 'synchronize', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    original_strided = torch.empty_strided
    original_like = torch.empty_like
    references = []

    def cpu_input(shape, stride, **kwargs):
        kwargs['device'] = 'cpu'
        return original_strided(shape, stride, **kwargs)

    def tracked_gradient(*args, **kwargs):
        tensor = original_like(*args, **kwargs)
        references.append(weakref.ref(tensor))
        return tensor

    def sample(label, *args):
        gc.collect()
        return dict(label=label, live_gradients=sum(ref() is not None for ref in references))

    monkeypatch.setattr(torch, 'empty_strided', cpu_input)
    monkeypatch.setattr(torch, 'empty_like', tracked_gradient)
    monkeypatch.setattr(target, 'memory', sample)
    output = tmp_path / 'output'
    output.mkdir()
    report = dict(memory=[], modes=[])
    target.run(tmp_path / 'input', output, report)
    release = [m for m in report['memory'] if m['label'].startswith('after-release')]
    assert len(release) == 20
    assert all(row['live_gradients'] == int(frozen_control) for row in release)
    assert report['memory'][-1]['live_gradients'] == int(frozen_control)
    assert calls == [False] * 10 + [True] * 10
    assert report['accepted']
