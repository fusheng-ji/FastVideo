# SPDX-License-Identifier: Apache-2.0
"""The negative-prompt text encoder must be freed as soon as encoding is done."""
import gc
import weakref

import torch

from fastvideo.train.utils.negative_prompt import _release_encoder_memory


class _Cyclic(torch.nn.Module):
    """A module that references itself, as hooks and wrappers around encoders can."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.self_ref = self


def test_release_frees_cyclic_encoder_immediately() -> None:
    gc.collect()
    gc.disable()
    try:
        module = _Cyclic()
        ref = weakref.ref(module.weight)
        del module
        assert ref() is not None  # without a collection the cycle keeps the weights alive
        _release_encoder_memory()
        assert ref() is None
    finally:
        gc.enable()
