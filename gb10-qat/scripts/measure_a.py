"""Joined (switch=1) vs split (switch=0) P@V on this GPU with attn_qat_train from PYTHONPATH."""
import json, math, os, sys
import torch
from fastvideo_kernel.triton_kernels import attn_qat_train as kernel

def run(inputs, grad_out, switch):
    os.environ["FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV"] = switch
    q, k, v = [t.clone().requires_grad_(True) for t in inputs]
    o = kernel.attention(q, k, v, False, 1.0 / math.sqrt(128), True, False, True, True, False, True, True, False, False, False)
    ste, m = o.grad_fn.saved_tensors[3:]
    return [o.detach(), *torch.autograd.grad(o, (q, k, v), grad_out), ste, m]

result = {}
for lq, lk in ((2112, 2112), (2112, 2080), (8192, 8192)):
    torch.manual_seed(0)
    inputs = [torch.randn((1, 3, n, 128), device="cuda", dtype=torch.bfloat16) for n in (lq, lk, lk)]
    g = torch.randn((1, 3, lq, 128), device="cuda", dtype=torch.bfloat16)
    split, joined = run(inputs, g, "0"), run(inputs, g, "1")
    for name, a, b in zip(("O", "dQ", "dK", "dV", "STE", "M"), joined, split):
        a, b = a.double(), b.double()
        result[f"{lq}/{lk}|{name}"] = dict(rel_l2=float((a - b).norm() / b.norm().clamp_min(1e-30)),
                                          mismatch=float((a != b).double().mean()))
json.dump(result, open(sys.argv[1], "w"))
