# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch
import torch.optim.adam as _torch_adam
from torch.distributed.tensor import DTensor

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        OptimizerConfig,
        TrainingLoopConfig,
    )

_FOREACH_ARGS = ("params", "grads", "exp_avgs", "exp_avg_sqs", "max_exp_avg_sqs", "state_steps")


def _chunked_multi_tensor_adam(original, chunk_size: int):
    """Run ``original`` on consecutive tensor lists of at most ``chunk_size``; the per-tensor math is unchanged."""

    def run(*args, **kwargs):
        lists, rest = args[:len(_FOREACH_ARGS)], args[len(_FOREACH_ARGS):]
        for offset in range(0, len(lists[0]), chunk_size):
            original(*(x[offset:offset + chunk_size] for x in lists), *rest, **kwargs)

    return run


class _ChunkedForeachAdamW(torch.optim.AdamW):
    """AdamW whose foreach update is split into bounded lists to cap its temporary tensors.

    PyTorch's foreach AdamW materializes parameter-sized temporaries for every tensor at once. Chunking keeps
    the foreach arithmetic and results unchanged while lowering the peak. It relies on the private
    ``torch.optim.adam._multi_tensor_adam``; the swap is scoped to this optimizer's ``step``.
    """

    def __init__(self, params, *, foreach_chunk_size: int, **kwargs):
        if foreach_chunk_size <= 0:
            raise ValueError(f"foreach_chunk_size must be positive, got {foreach_chunk_size}")
        original = getattr(_torch_adam, "_multi_tensor_adam", None)
        if original is None or tuple(inspect.signature(original).parameters)[:len(_FOREACH_ARGS)] != _FOREACH_ARGS:
            raise RuntimeError("foreach_chunk_size needs torch.optim.adam._multi_tensor_adam with the expected "
                               f"signature; unsupported torch {torch.__version__}")
        self._foreach_chunk_size = foreach_chunk_size
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        original = _torch_adam._multi_tensor_adam
        _torch_adam._multi_tensor_adam = _chunked_multi_tensor_adam(original, self._foreach_chunk_size)
        try:
            return super().step(closure)
        finally:
            _torch_adam._multi_tensor_adam = original


def build_optimizer_and_scheduler(
    *,
    params: list[torch.nn.Parameter],
    optimizer_config: OptimizerConfig,
    loop_config: TrainingLoopConfig,
    learning_rate: float,
    betas: tuple[float, float],
    scheduler_name: str,
    weight_decay: float | None = None,
) -> tuple[torch.optim.Optimizer, object]:
    """Build an AdamW optimizer and LR scheduler.

    Returns ``(optimizer, lr_scheduler)`` so the caller can store them
    as method-level attributes.
    """
    if not params:
        raise ValueError("No trainable parameters passed to "
                         "build_optimizer_and_scheduler")

    adamw_kwargs: dict[str, Any] = dict(
        lr=float(learning_rate),
        betas=betas,
        weight_decay=float(optimizer_config.weight_decay if weight_decay is None else weight_decay),
        eps=float(optimizer_config.eps),
    )
    if bool(getattr(optimizer_config, "fused", False)):
        adamw_kwargs["fused"] = True
    chunk_size = int(optimizer_config.foreach_chunk_size)
    if chunk_size < 0:
        raise ValueError(f"optimizer.foreach_chunk_size must be >= 0, got {chunk_size}")
    if chunk_size and adamw_kwargs.get("fused"):
        raise ValueError("optimizer.foreach_chunk_size cannot be combined with optimizer.fused")
    if chunk_size:
        optimizer = _ChunkedForeachAdamW(params, foreach_chunk_size=chunk_size, **adamw_kwargs)
    else:
        optimizer = torch.optim.AdamW(params, **adamw_kwargs)

    if str(scheduler_name) == "multistep_with_warmup":
        milestones = list(optimizer_config.lr_milestones)
        if not milestones:
            raise ValueError("multistep_with_warmup requires training.optimizer.lr_milestones")
        warmup_steps = int(optimizer_config.lr_warmup_steps)
        if warmup_steps <= 0:
            raise ValueError("multistep_with_warmup requires lr_warmup_steps > 0")

        def warmup(current_step: int) -> float:
            return (current_step + 1) / (warmup_steps + 1)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup,
        )
        step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(optimizer_config.lr_gamma),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = get_scheduler(
            str(scheduler_name),
            optimizer=optimizer,
            num_warmup_steps=int(optimizer_config.lr_warmup_steps),
            num_training_steps=int(loop_config.max_train_steps),
            num_cycles=int(optimizer_config.lr_num_cycles),
            power=float(optimizer_config.lr_power),
            min_lr_ratio=float(optimizer_config.min_lr_ratio),
            last_epoch=-1,
        )

    return optimizer, scheduler


def seed_adamw_parameter_state(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
) -> None:
    """Create AdamW state using PyTorch's fused/capturable device contract."""
    if optimizer.state.get(parameter):
        return
    owner_group = next(
        (group for group in optimizer.param_groups if any(candidate is parameter for candidate in group["params"])),
        None,
    )
    if owner_group is None:
        raise ValueError("Cannot seed optimizer state for an unowned parameter")
    step_on_parameter = bool(owner_group.get("capturable", False) or owner_group.get("fused", False))
    step_device = parameter.device if step_on_parameter else torch.device("cpu")
    optimizer.state[parameter] = {
        "step": torch.zeros((), dtype=torch.float32, device=step_device),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }


def clip_grad_norm_if_needed(
    module: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    if max_grad_norm <= 0.0:
        return 0.0
    grad_norm = (clip_grad_norm_while_handling_failing_dtensor_cases(
        [p for p in module.parameters()],
        max_grad_norm,
        foreach=None,
    ))
    return (float(grad_norm.item()) if grad_norm is not None else 0.0)


@torch.no_grad()
def measure_grad_norm(module: torch.nn.Module) -> float:
    """Measure the global L2 gradient norm without changing gradients.

    This uses the same DTensor-aware reduction as gradient clipping, but it
    deliberately does not apply a clip coefficient. In particular, an
    overflowing norm must remain diagnostic evidence rather than multiplying
    otherwise-finite gradients by ``inf / inf``.
    """
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        return 0.0
    grad_norm = torch.nn.utils.get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=None,
    )
    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    return float(grad_norm.item())


@torch.no_grad()
def sanitize_non_finite_gradients(module: torch.nn.Module) -> int:
    """Replace non-finite local gradient values using ``torch.nan_to_num``.

    FSDP2 exposes gradients as DTensors. Mutating their local tensor avoids a
    collective while applying the same per-rank sanitation as FastGen's
    gradient callback.
    """
    non_finite_count = 0
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        local_grad = grad._local_tensor if isinstance(grad, DTensor) else grad
        count = int((~torch.isfinite(local_grad)).sum().item())
        if count:
            non_finite_count += count
            torch.nan_to_num_(local_grad)
    return non_finite_count
