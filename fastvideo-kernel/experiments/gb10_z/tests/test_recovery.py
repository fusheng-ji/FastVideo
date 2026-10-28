import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import pytest
import torch
from common import span, digest, tensor_hash
from fa2_extract import extract


@pytest.mark.parametrize('shape,stride,expected', [([2,3],[3,1],24),([2,3],[1,2],24),([2,3],[5,1],32),
                                                    ([1,3],[999,1],12),([0,3],[3,1],0),([],[],4)])
def test_span(shape,stride,expected):
    assert span(shape,stride,4)==expected


@pytest.mark.parametrize('shape,stride', [([2,3],[0,1]),([2,3],[1,1]),([2],[-1]),([-1],[1]),([2],[True]),([2],[])])
def test_invalid(shape,stride):
    with pytest.raises(ValueError):span(shape,stride,4)


def capture():
    inputs={name:torch.arange(6,dtype=torch.float32).reshape(2,3) for name in ('dout','q','k','v','out','lse')}
    # Views backed by oversized storage must not pull that storage into the small artifact.
    inputs['q']=torch.arange(1000,dtype=torch.float32)[:6].reshape(2,3)
    return dict(schema_version=1,flash=[dict(function='_fa2_backward',layer=29,kwargs=dict(deterministic=False,dropout_p=0.),
        inputs=inputs,metadata={k:dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype)) for k,t in inputs.items()},
        gradients={name:torch.zeros(2,3) for name in ('dq','dk','dv')})])


def test_independent_storage_and_exact_serialization(tmp_path):
    p=tmp_path/'source.pt';torch.save(capture(),p)
    result=extract(p,tmp_path/'minimal')
    assert result['accepted'] and not result['cuda_initialized']
    assert result['tensors']['q']['original_storage_bytes']==4000
    assert result['tensors']['q']['storage_bytes']==24
    for name,row in result['tensors'].items():
        value=torch.load(tmp_path/'minimal'/row['file'],weights_only=True,mmap=True)
        assert value.untyped_storage().nbytes()==row['logical_bytes']
        assert tensor_hash(value)==row['raw_sha256']


def test_missing_input_fails(tmp_path):
    data=capture();del data['flash'][0]['inputs']['q'];p=tmp_path/'source.pt';torch.save(data,p)
    with pytest.raises(ValueError,match='Missing'):extract(p,tmp_path/'minimal')
    assert not (tmp_path/'minimal/manifest.json').exists()


def test_streaming_hash(tmp_path,monkeypatch):
    p=tmp_path/'large';p.write_bytes(b'a'*(2**20+1))
    def forbidden(*a,**k):raise AssertionError('read_bytes forbidden')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    assert digest(p)==hashlib.sha256(b'a'*(2**20+1)).hexdigest()


def test_replay_cleanup_contract():
    import ast
    source=ast.parse((Path(__file__).resolve().parents[1]/'code/fa2_replay.py').read_text())
    calls=[n for n in ast.walk(source) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr=='read_bytes' for n in calls)
    assert any(isinstance(n,ast.Try) and n.finalbody for n in ast.walk(source))
    assert "del grads" in (Path(__file__).resolve().parents[1]/'code/fa2_replay.py').read_text()


def test_allocation_failure_runs_cleanup(tmp_path,monkeypatch):
    import types
    import fa2_replay as replay
    source=tmp_path/'source.pt';torch.save(capture(),source)
    extract(source,tmp_path/'input')
    module=types.ModuleType('flash_attn.flash_attn_interface')
    module._flash_attn_backward=lambda *a,**k:None
    monkeypatch.setitem(sys.modules,'flash_attn.flash_attn_interface',module)
    for name in ('init','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    cleaned=[]
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:cleaned.append(True))
    monkeypatch.setattr(replay,'memory',lambda label,*args:dict(label=label))
    def oom(*args,**kwargs):raise torch.OutOfMemoryError('injected allocation failure')
    monkeypatch.setattr(torch,'empty_strided',oom)
    report=dict(memory=[],modes=[])
    output=tmp_path/'output';output.mkdir()
    with pytest.raises(torch.OutOfMemoryError,match='injected'):
        replay.run(tmp_path/'input',output,report)
    assert cleaned==[True]
    assert report['memory'][-1]['label']=='cleanup'
    assert 'before-allocate-dout' in [x['label'] for x in report['memory']]
