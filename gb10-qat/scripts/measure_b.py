"""Quantize a partially masked tile with nvfp4_utils from PYTHONPATH; dump per-padding scales/outputs."""
import json, sys
import torch, triton, triton.language as tl
from fastvideo_kernel.triton_kernels.nvfp4_utils import MXFP_BLOCK_SIZE, _compute_dequant, _compute_quant_and_scale

TILE = 32
GROUPS = TILE // MXFP_BLOCK_SIZE.value

@triton.jit
def quantize_masked(X, VALID, OUT, SCALE, TILE: tl.constexpr, GROUPS: tl.constexpr, GLOBAL: tl.constexpr, TWO: tl.constexpr):
    rows = tl.arange(0, TILE); columns = tl.arange(0, TILE)
    values = tl.load(X + rows[:, None] * TILE + columns[None, :])
    valid = tl.load(VALID + rows[:, None] * TILE + columns[None, :])
    packed, scale, decode = _compute_quant_and_scale(values, valid, use_global_sf=GLOBAL, two_level_quant_P=TWO)
    out = _compute_dequant(packed, scale, decode, TILE, TILE, tl.bfloat16)
    tl.store(OUT + rows[:, None] * TILE + columns[None, :], out)
    tl.store(SCALE + rows[:, None] * GROUPS + tl.arange(0, GROUPS)[None, :], scale.to(tl.float32))

result = {}
for mode, (g, two) in {"per_group": (False, False), "global": (True, False), "two_level": (False, True)}.items():
    torch.manual_seed(42)
    source = torch.rand((TILE, TILE), device="cuda")
    valid = torch.zeros_like(source, dtype=torch.bool); valid[:29, :23] = True
    invalid_groups = ~valid.reshape(TILE, GROUPS, -1).any(-1)
    ref = None
    for padding in (0.0, 1e4, float("nan")):
        x = torch.where(valid, source, padding)
        out = torch.empty_like(source, dtype=torch.bfloat16); sc = torch.empty((TILE, GROUPS), device="cuda")
        quantize_masked[(1,)](x, valid, out, sc, TILE, GROUPS, g, two, num_warps=4)
        out = out.float()
        if ref is None: ref = out
        d = (out[valid] - ref[valid]).abs()
        result[f"{mode}|{padding}"] = dict(
            nonfinite=int((~torch.isfinite(out)).sum()) + int((~torch.isfinite(sc)).sum()),
            leaked_scales=int((sc[invalid_groups] != 0).sum()),
            valid_changed=int(((d > 0) | d.isnan()).sum()),
            scales=sc.cpu().tolist())
json.dump(result, open(sys.argv[1], "w"))
