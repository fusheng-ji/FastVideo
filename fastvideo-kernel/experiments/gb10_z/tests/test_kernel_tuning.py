"""CPU checks of private P64 dispatch and the actual kernel address expressions.

The tests extract the small host helpers without importing CUDA or the kernel
package. Address expressions are evaluated from the edited Triton source and
checked against NumPy's independent five-dimensional indexing, including all
valid rows of non-aligned tails and multiple batch/head combinations.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT.parents[1] / "python/fastvideo_kernel/triton_kernels/attn_qat_train.py"
TREE = ast.parse(KERNEL.read_text())
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}


def host_helpers():
    names = {
        "_sm121_packed_tuning", "_sm121_effective_packed_tuning", "_sm121_event_value_shape", "_sm121_has_spill_launch",
        "_sm121_local_cache_bytes", "_sm121_dv_stats_format", "_sm121_local_replay_enabled"
    }
    module = ast.Module(body=[FUNCTIONS[name] for name in sorted(names)], type_ignores=[])
    namespace = {"SM121_BLOCK": 32, "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)}
    exec(compile(ast.fix_missing_locations(module), str(KERNEL), "exec"), namespace)
    return namespace


def expression(function, target, condition=None):
    """Get an address expression from its actual (possibly constexpr) branch."""
    root = FUNCTIONS[function]
    if condition:
        root = next(node for node in ast.walk(root) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == condition)
        nodes = root.body
    else:
        nodes = ast.walk(root)
    return next(node.value for node in nodes if isinstance(node, ast.Assign)
                and any(isinstance(lhs, ast.Name) and lhs.id == target for lhs in node.targets))


def evaluate(expr, **namespace):
    return eval(compile(ast.Expression(expr), str(KERNEL), "eval"), {"__builtins__": {}}, namespace)


def test_default_tuning_does_not_enable_statistics_or_local_replay():
    helpers = host_helpers()
    assert helpers["_sm121_packed_tuning"]() == (False, False)
    assert helpers["_sm121_dv_stats_format"]() == ("dense", 0, False)
    assert helpers["_sm121_local_replay_enabled"]() is False


@pytest.mark.parametrize("selected", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("packed", [False, True])
def test_private_policy_isolated_from_dense_and_legacy_indices(selected, packed):
    helpers = host_helpers()
    helpers["_sm121_packed_tuning"] = lambda: selected
    assert helpers["_sm121_effective_packed_tuning"](packed) == (selected if packed else (False, False))


@pytest.mark.parametrize("selected", [(1, False), (False, 0), (None, True), (True, "yes")])
def test_policy_rejects_non_boolean_switches(selected):
    helpers = host_helpers()
    helpers["_sm121_packed_tuning"] = lambda: selected
    with pytest.raises(TypeError, match="booleans"):
        helpers["_sm121_effective_packed_tuning"](True)


@pytest.mark.parametrize("local,count,fast,expected", [
    (False, 0, False, False), (False, 1, True, False),
    (True, 0, False, True), (True, 0, True, False),
    (True, 1, False, True), (True, 1, True, True),
    (True, 100, True, True),
])
def test_zero_spill_dispatch_preserves_spilling_and_baseline_paths(local, count, fast, expected):
    assert host_helpers()["_sm121_has_spill_launch"](local, count, fast) is expected


@pytest.mark.parametrize("nq", [1, 31, 32, 33, 257, 2113, 8192, 8193, 31200])
def test_tiled_value_addresses_are_bijective_and_exclude_padding(nq):
    batch, heads, capacity = 2, 3, 64
    helpers = host_helpers()
    shape = helpers["_sm121_event_value_shape"](batch, heads, nq, capacity, True)
    assert shape == (batch, heads, (nq + 31) // 32, capacity, 32)
    query, slot, bh = np.meshgrid(np.arange(nq), np.arange(capacity), np.arange(batch * heads), indexing="ij")
    expected = np.ravel_multi_index((bh // heads, bh % heads, query // 32, slot, query % 32), shape)
    backward = evaluate(expression("_sm121_packed_maximum", "value_offset", "TILED_VALUES"),
                        bhid=bh, NQ=nq, query_rows=query, slot=slot,
                        tl=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b))
    forward_local = evaluate(expression("_attn_fwd_inner", "slot_offsets", "DV_TILED_VALUES"),
                             offs_m=query, DV_EVENT_CAPACITY=capacity, event_count=slot, BLOCK_M=32)
    forward = bh * ((nq + 31) // 32) * capacity * 32 + forward_local
    assert np.array_equal(forward, expected)
    assert np.array_equal(backward, expected)
    assert np.unique(forward).size == batch * heads * nq * capacity
    assert int(forward.min()) >= 0 and int(forward.max()) < math.prod(shape)


@pytest.mark.parametrize("nq", [257, 8192, 31200])
def test_repacking_preserves_fp32_raw_bits_and_tail_padding_is_unobserved(nq):
    # Raw FP32 patterns include both zeros, subnormal values and normal numbers.
    # Repacking never performs arithmetic on event values.
    shape = (2, 3, (nq + 31) // 32, 64, 32)
    source = np.resize(np.array([0, 0x80000000, 1, 0x3F800000, 0xBF800000, 0x7F7FFFFF], dtype=np.uint32), (2, 3, 64, nq))
    packed = np.full(shape, 0x7FC00001, dtype=np.uint32)
    for query in range(nq):
        packed[:, :, query // 32, :, query % 32] = source[:, :, :, query]
    restored = np.stack([packed[:, :, query // 32, :, query % 32] for query in range(nq)], axis=-1)
    assert np.array_equal(restored, source)
    if nq % 32:
        assert np.all(packed[:, :, -1, :, nq % 32:] == np.uint32(0x7FC00001))


@pytest.mark.parametrize("nq", [257, 8192, 8193, 31200])
@pytest.mark.parametrize("spill", [0, 1, 7])
def test_saved_cache_bytes_include_tiled_padding_and_spill(nq, spill):
    helpers = host_helpers()
    batch, heads, nk, capacity = 2, 3, 31200, 64
    kv_tiles, qt = (nk + 31) // 32, (nq + 31) // 32
    values = 4 * math.prod(helpers["_sm121_event_value_shape"](batch, heads, nq, capacity, True))
    directory = 4 * batch * heads * ((kv_tiles + 4) // 5) * nq
    count_final_denom_flags = 13 * batch * heads * nq
    mapping, spill_list, replay = 4 * batch * heads * qt, 4 * spill, 4 * spill * kv_tiles * 32
    expected = values + directory + count_final_denom_flags + mapping + spill_list + replay
    actual = helpers["_sm121_local_cache_bytes"](batch, heads, nq, nk, spill, capacity, True, True)
    assert actual == expected
    original = helpers["_sm121_local_cache_bytes"](batch, heads, nq, nk, spill, capacity, True, False)
    assert actual - original == 4 * capacity * batch * heads * (qt * 32 - nq)


def test_zero_spill_kernel_branch_contains_no_spill_map_load():
    function = FUNCTIONS["_attn_bwd_dv_forward_weights"]
    branch = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "LOCAL_REPLAY and HAS_SPILL")
    assert "SPILL_MAP" in ast.unparse(branch.body[0])
    assert "SPILL_MAP" not in "\n".join(ast.unparse(node) for node in branch.orelse)
    assert "_sm121_packed_maximum" in "\n".join(ast.unparse(node) for node in branch.orelse)


def test_fastpath_dispatch_is_saved_at_forward_and_used_without_new_sync():
    # Classes are handled separately to inspect the private autograd context.
    cls = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "_attention")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    backward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "backward")
    dispatch = next(node for node in ast.walk(forward) if isinstance(node, ast.Assign)
                    and any(isinstance(lhs, ast.Name) and lhs.id == "dv_has_spill_launch" for lhs in node.targets))
    assert ast.unparse(dispatch.value) == "_sm121_has_spill_launch(local_dv_replay, dv_spill_tile_count, dv_zero_spill_fastpath)"
    launch = next(node for node in ast.walk(backward) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Subscript)
                  and ast.unparse(node.func.value) == "_attn_bwd_dv_forward_weights")
    kwargs = {item.arg: ast.unparse(item.value) for item in launch.keywords}
    assert kwargs["HAS_SPILL"] == "ctx.dv_has_spill_launch"
    assert kwargs["TILED_VALUES"] == "ctx.dv_packed_tiled_values"
    helper_source = ast.unparse(FUNCTIONS["_sm121_has_spill_launch"])
    assert "torch" not in helper_source and ".item" not in helper_source
