import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('peak_trace', Path(__file__).parents[1]/'code/peak_trace.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def test_trace_transient_peak():
    start = {'segments': [{'blocks': [{'address': 10, 'size': 512, 'state': 'active_allocated'}]}], 'device_traces': [[]]}
    end = {'device_traces': [[{'action': 'alloc', 'addr': 20, 'size': 1024}, {'action': 'free_requested', 'addr': 20, 'size': 1024}]]}
    peak, blocks = trace.reconstruct(start, end)
    assert peak == 1536 and len(blocks) == 2
    result = trace.summarize(start, end, {'10': {'category': 'parameters_master_fp32', 'bytes': 512}})
    assert sum(result['category_bytes'].values()) == peak


def test_reuse_does_not_accumulate_lifetimes():
    start = {'segments': [], 'device_traces': [[]]}
    end = {'device_traces': [[{'action': 'alloc', 'addr': 10, 'size': 100}, {'action': 'free_requested', 'addr': 10, 'size': 100}, {'action': 'free_completed', 'addr': 10, 'size': 100}, {'action': 'alloc', 'addr': 10, 'size': 200}]]}
    assert trace.reconstruct(start, end)[0] == 200


def test_duplicate_address_rejected():
    import pytest
    start = {'segments': [], 'device_traces': [[]]}
    end = {'device_traces': [[{'action': 'alloc', 'addr': 10, 'size': 100}]*2]}
    with pytest.raises(AssertionError):
        trace.reconstruct(start, end)


def test_gradient_release_cannot_pass_finite_coverage():
    import torch
    p = torch.nn.Parameter(torch.ones(4))
    p.sum().backward()
    p.grad = None
    assert not all(x.grad is not None and torch.isfinite(x.grad).all() for x in [p])


def test_lower_precision_gradient_assignment_rejected():
    import pytest
    import torch
    p = torch.nn.Parameter(torch.ones(4))
    p.sum().backward()
    with pytest.raises(RuntimeError):
        p.grad = p.grad.bfloat16()


def test_chunked_adam_matches_full_foreach_cpu():
    import importlib
    import torch
    path = Path(__file__).parents[1]/'code/peak_chunked.py'
    spec = importlib.util.spec_from_file_location('peak_chunked', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adam = importlib.import_module('torch.optim.adam')
    original = adam._multi_tensor_adam
    torch.manual_seed(1000)
    left = [torch.nn.Parameter(torch.randn(31+i)) for i in range(130)]
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    a = torch.optim.AdamW(left, lr=1e-6, foreach=True)
    b = torch.optim.AdamW(right, lr=1e-6, foreach=True)
    for _ in range(4):
        for p,q in zip(left,right):
            p.grad=torch.randn_like(p);q.grad=p.grad.clone()
        a.step()
        adam._multi_tensor_adam=module.chunked(original)
        try:
            b.step()
        finally:
            adam._multi_tensor_adam=original
        assert all(torch.equal(p,q) for p,q in zip(left,right))
        assert all(torch.equal(a.state[p][key],b.state[q][key]) for p,q in zip(left,right) for key in ('exp_avg','exp_avg_sq','step'))


def test_storage_inventory_deduplicates_master_parameter_aliases():
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('peak_memory', Path(__file__).parents[1]/'code/peak_memory.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Storage:
        def __init__(self, address):self.address=address
        def data_ptr(self):return self.address
        def nbytes(self):return 16
    class Tensor:
        device=SimpleNamespace(type='cuda')
        dtype='float32'
        grad=None
        def __init__(self,address):self.address=address
        def untyped_storage(self):return Storage(self.address)
    p=Tensor(100);p.grad=Tensor(200)
    optimizer=SimpleNamespace(state={'p': {'exp_avg':Tensor(300),'exp_avg_sq':Tensor(400)}})
    result=module.inventory([p,p],optimizer)
    assert len(result)==4
    assert sum(v['bytes'] for v in result.values())==64
    assert result[100]['category']=='parameters_master_fp32'


def test_trace_exports_basename_not_local_source_path():
    start={'segments':[],'device_traces':[[]]}
    end={'device_traces':[[{'action':'alloc','addr':1,'size':512,'frames':[{'filename':'private-root/package/adam.py','line':12,'name':'sqrt'}]}]]}
    result=trace.summarize(start,end,{})
    assert result['other_allocation_origins']=={'adam.py:12:sqrt':512}


def test_telemetry_alignment_is_independent_of_analysis_timezone(tmp_path):
    spec = importlib.util.spec_from_file_location('peak_report', Path(__file__).parents[1]/'code/peak_report.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=tmp_path/'telemetry.csv'
    p.write_text('timestamp,temperature.gpu,clocks.current.sm [MHz],utilization.gpu [%],power.draw [W]\n1970/01/01 01:00:00.100,40,1794,95,30\n')
    result=module.clocks(p,[{'start_unix':0,'end_unix':1}],0)
    assert result['telemetry_utc_offset_seconds']==3600
    assert result['clock_p10_p50_p90']==[1794,1794,1794]
