# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the optional chunked foreach AdamW (``optimizer.foreach_chunk_size``)."""
import functools

import pytest
import torch
import torch.optim.adam as torch_adam

from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler
from fastvideo.train.utils.training_config import OptimizerConfig, TrainingLoopConfig


def _params(seed=0, count=10):
    generator = torch.Generator().manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(3 + i, 5, generator=generator)) for i in range(count)]
    for p in params:
        p.grad = torch.randn(p.shape, generator=generator)
    return params


def _build(params, chunk):
    config = OptimizerConfig(learning_rate=1e-3, weight_decay=0.01, foreach_chunk_size=chunk)
    return build_optimizer_and_scheduler(params=params,
                                         optimizer_config=config,
                                         loop_config=TrainingLoopConfig(max_train_steps=10),
                                         learning_rate=1e-3,
                                         betas=(0.9, 0.999),
                                         scheduler_name="constant")[0]


def _run(chunk, steps=3):
    params = _params()
    optimizer = _build(params, chunk)
    for _ in range(steps):
        optimizer.step()
        for p in params:
            p.grad = p.detach().clone() * 0.5 + 0.1
    return params, optimizer


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_chunked_foreach_matches_default_bitwise(chunk):
    reference, ref_opt = _run(0)
    chunked, opt = _run(chunk)
    for a, b in zip(reference, chunked):
        assert torch.equal(a, b)
        assert torch.equal(ref_opt.state[a]["exp_avg"], opt.state[b]["exp_avg"])
        assert torch.equal(ref_opt.state[a]["exp_avg_sq"], opt.state[b]["exp_avg_sq"])


def test_chunked_foreach_bounds_list_length_and_restores_global():
    original = torch_adam._multi_tensor_adam
    seen = []

    @functools.wraps(original)
    def spy(params, *args, **kwargs):
        seen.append(len(params))
        return original(params, *args, **kwargs)

    torch_adam._multi_tensor_adam = spy
    try:
        params = _params(count=10)
        optimizer = _build(params, 4)
        optimizer.param_groups[0]["foreach"] = True
        optimizer.step()
        assert seen == [4, 4, 2]
        assert torch_adam._multi_tensor_adam is spy
    finally:
        torch_adam._multi_tensor_adam = original


def test_default_keeps_plain_adamw():
    assert type(_build(_params(), 0)) is torch.optim.AdamW


def test_negative_chunk_size_rejected():
    with pytest.raises(ValueError):
        _build(_params(), -1)


def test_chunking_with_fused_rejected():
    config = OptimizerConfig(learning_rate=1e-3, foreach_chunk_size=8, fused=True)
    with pytest.raises(ValueError):
        build_optimizer_and_scheduler(params=_params(),
                                      optimizer_config=config,
                                      loop_config=TrainingLoopConfig(max_train_steps=10),
                                      learning_rate=1e-3,
                                      betas=(0.9, 0.999),
                                      scheduler_name="constant")
