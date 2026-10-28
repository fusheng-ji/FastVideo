import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import calibration_runtime as runtime
from control_train import require_recipe
from probe_helpers import capture_rng, rng_identity
from common import tensor_hash


@pytest.mark.parametrize('seed', [41001, 41002, 41003])
def test_recipe(seed):
    require_recipe(SimpleNamespace(seed=seed, factor=1, samples=20, tokens=8192))


@pytest.mark.parametrize('change', [dict(seed=1000), dict(factor=4), dict(samples=21), dict(tokens=31200)])
def test_recipe_rejects_expansion(change):
    args=dict(seed=41001, factor=1, samples=20, tokens=8192);args.update(change)
    with pytest.raises(ValueError):require_recipe(SimpleNamespace(**args))


@pytest.mark.parametrize('fail', [False, True])
def test_evaluation_restores_all_rng_modes_and_sampler(tmp_path, monkeypatch, fail):
    fake_cuda=[torch.get_rng_state().clone()]
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [x.clone() for x in fake_cuda])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda states: fake_cuda.__setitem__(slice(None), states))
    monkeypatch.setattr(runtime, 'STATE', dict(phase='idle', rep=None))
    monkeypatch.setattr(runtime, 'OUTPUT', tmp_path)
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout());model[1].eval()
    sampler=lambda: None
    method=SimpleNamespace(cuda_generator=torch.Generator().manual_seed(41001),student=SimpleNamespace(_sample_timesteps=sampler))
    before=rng_identity(capture_rng(torch,np,method.cuda_generator),tensor_hash)
    modes=[m.training for m in model.modules()]
    @runtime.evaluation_guard(method, model)
    def evaluate(step):
        assert runtime.STATE['phase']=='evaluation'
        random.random();np.random.rand();torch.rand(2);torch.rand(2,generator=method.cuda_generator)
        fake_cuda[0]=torch.zeros_like(fake_cuda[0]);model.eval();method.student._sample_timesteps=lambda: 2
        if fail:raise RuntimeError('evaluation failed')
        return 3
    if fail:
        with pytest.raises(RuntimeError,match='evaluation failed'):evaluate(10)
    else:assert evaluate(10)==3
    assert rng_identity(capture_rng(torch,np,method.cuda_generator),tensor_hash)==before
    assert [m.training for m in model.modules()]==modes
    assert method.student._sample_timesteps is sampler
    assert runtime.STATE['phase']=='idle'


def test_evaluation_cannot_interrupt_microbatch(monkeypatch):
    monkeypatch.setattr(runtime,'STATE',dict(phase='backward',rep={}))
    @runtime.evaluation_guard(None,None)
    def evaluate(step):raise AssertionError('must not execute')
    with pytest.raises(RuntimeError,match='inside a training microbatch'):evaluate(0)


def test_saved_state_does_not_keep_full_storage(tmp_path):
    runtime.save_named(tmp_path/'state',[('view',torch.arange(100)[:4])],torch)
    rows=json.loads((tmp_path/'state/manifest.json').read_text())
    value=torch.load(tmp_path/'state'/rows[0]['file'],weights_only=True)
    assert value.untyped_storage().nbytes()==4*8


def test_missing_final_gradient_fails(tmp_path):
    with pytest.raises(RuntimeError,match='Missing final tensor'):
        runtime.save_named(tmp_path/'state',[('missing',None)],torch)


def test_evaluation_hooks_do_not_create_training_events():
    from model_trace import ModelTrace
    phase=['evaluation']
    model=SimpleNamespace(blocks=[torch.nn.Linear(2,2) for _ in range(30)])
    trace=ModelTrace(model,8192,lambda x: {},lambda x:x,lambda *a:{},lambda:phase[0],lambda *a,**k:'evaluation-output')
    try:
        model.blocks[0](torch.ones(1,2))
        assert trace.attention(None)=='evaluation-output'
        assert trace.active=={}
    finally:trace.close()


@pytest.mark.parametrize('field', ['batch_sha256','rng_before','timesteps','events'])
def test_pairing_rejects_changed_input_rng_or_event_order(field):
    import copy
    from compare_calibration import check_micro_pair
    event=lambda name:dict(phase='backward',kind='parameter_gradient_arrival',module=name,call_id=name)
    a=dict(input_state=dict(batch_sha256='same',rng_before={'cpu':'same'}),timesteps={'sha256':'same'},rng_after={'cpu':'same'},events=[event('a'),event('b')])
    b=copy.deepcopy(a)
    if field in b['input_state']:b['input_state'][field]='changed'
    elif field=='events':b['events'].reverse()
    else:b[field]={'sha256':'changed'}
    with pytest.raises(RuntimeError):check_micro_pair(a,b)


def test_numeric_trajectory_difference_is_observation_not_pairing_failure():
    from compare_calibration import check_micro_pair
    a=dict(input_state=dict(batch_sha256='same',rng_before={'cpu':'same'},parameters_sha256='left'),timesteps={'sha256':'same'},rng_after={'cpu':'same'},events=[],loss=1.0)
    b=dict(a,input_state=dict(a['input_state'],parameters_sha256='right'),loss=1.1)
    check_micro_pair(a,b)


def test_report_domain_cannot_be_mixed():
    from compare_calibration import check_pair
    from local_policy import SCOPE
    fields=('source_fingerprint','config_fingerprint','initial_parameters','initial_buffers_sha256','initial_rng','initial_optimizer','initial_scheduler')
    a={k:'same' for k in fields};a['acceptance_scope']=SCOPE
    b=dict(a,acceptance_scope='old-strict-control')
    with pytest.raises(RuntimeError,match='configuration/state'):check_pair(a,b)


@pytest.mark.parametrize('change', ['none','missing','noise','timestep','duplicate'])
def test_validation_budget_and_identity(tmp_path,monkeypatch,change):
    code=tmp_path/'code';code.mkdir()
    import portable
    monkeypatch.setattr(portable, '_ACTIVE', SimpleNamespace(validation_manifest=tmp_path/'validation-inputs.json'))
    declared=[dict(id=str(i),noise_seed=3100+i) for i in range(32)]
    (tmp_path/'validation-inputs.json').write_text(json.dumps(dict(rows=declared)))
    rows=[dict(event='eval-sample',step=step,split='heldout',draw='fixed',timestep=[i],**r) for step in (0,10,20) for i,r in enumerate(declared)]
    if change=='missing':rows.pop()
    elif change=='noise':rows[-1]['noise_seed']=999
    elif change=='timestep':rows[-1]['timestep']=[999]
    elif change=='duplicate':rows[-1]['id']='0'
    run=tmp_path/'run';run.mkdir()
    (run/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (run/'evaluation-isolation.jsonl').write_text(''.join(json.dumps(dict(step=s,rng_restored=True))+'\n' for s in (0,10,20)))
    if change=='none':runtime.validate_evaluations(run)
    else:
        with pytest.raises(RuntimeError):runtime.validate_evaluations(run)
