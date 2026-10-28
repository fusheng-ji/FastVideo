"""Optimizer-window semantics shared by the local GB10 training experiment.

The caller supplies an already-zeroed method. Loss scaling remains owned by
``method.backward(..., grad_accum_rounds=...)``, as in the canonical trainer.
No model, CUDA, environment or public FastVideo interface is changed here.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class WindowResult:
    """One completed optimizer update, after all microbatch gradients accumulate."""

    losses: dict[str, float]
    gradient_norm: float
    diagnostics: dict[str, Any]
    microbatches: int


class NonFiniteWindow(RuntimeError):
    """A nonfinite loss or gradient prevents the window's optimizer update."""


def empty_cache_metrics() -> dict[str, Any]:
    """Host metadata accounting only; no tensor unpack, transfer or synchronization."""
    return dict(attention_forward_calls=0, saved_statistics_calls=0, saved_cache_bytes=0,
                dense_cache_bytes=0, overflow_fallback_calls=0, formats={},
                maximum_saved_cache_bytes_per_attention=0, maximum_dense_cache_bytes_per_attention=0,
                minimum_saved_cache_reduction_per_attention=None, saved_cache_reduction=None)


def record_saved_cache(cache, context, q_shape, k_shape):
    cache['attention_forward_calls'] += 1
    if (context is None or not getattr(context, 'forward_consistent_dv', False)
            or getattr(context, 'dv_stats_mode', None) != 'save'
            or not hasattr(context, 'dv_stats_storage_bytes')):
        return
    batch, heads, nq, _ = q_shape
    nk = k_shape[2]
    dense_bytes = batch * heads * nq * (((nk + 31) // 32) * 4 + 4)
    saved_bytes = int(context.dv_stats_storage_bytes)
    if saved_bytes < 0 or dense_bytes <= 0:
        raise ValueError('Invalid saved statistics byte metadata')
    cache['saved_statistics_calls'] += 1
    cache['saved_cache_bytes'] += saved_bytes
    cache['dense_cache_bytes'] += dense_bytes
    cache['overflow_fallback_calls'] += bool(getattr(context, 'dv_overflow_fallback', False))
    cache['maximum_saved_cache_bytes_per_attention'] = max(
        cache['maximum_saved_cache_bytes_per_attention'], saved_bytes)
    cache['maximum_dense_cache_bytes_per_attention'] = max(
        cache['maximum_dense_cache_bytes_per_attention'], dense_bytes)
    reduction = 1 - saved_bytes / dense_bytes
    old_minimum = cache['minimum_saved_cache_reduction_per_attention']
    cache['minimum_saved_cache_reduction_per_attention'] = reduction if old_minimum is None else min(old_minimum, reduction)
    cache['saved_cache_reduction'] = 1 - cache['saved_cache_bytes'] / cache['dense_cache_bytes']
    label = str(getattr(context, 'dv_stats_format', 'dense'))
    cache['formats'][label] = cache['formats'].get(label, 0) + 1


def combine_cache_metrics(records):
    combined = empty_cache_metrics()
    for row in records:
        for key in ('attention_forward_calls', 'saved_statistics_calls', 'saved_cache_bytes',
                    'dense_cache_bytes', 'overflow_fallback_calls'):
            combined[key] += row[key]
        for key in ('maximum_saved_cache_bytes_per_attention', 'maximum_dense_cache_bytes_per_attention'):
            combined[key] = max(combined[key], row[key])
        minimum = row['minimum_saved_cache_reduction_per_attention']
        if minimum is not None:
            previous = combined['minimum_saved_cache_reduction_per_attention']
            combined['minimum_saved_cache_reduction_per_attention'] = minimum if previous is None else min(previous, minimum)
        for label, count in row['formats'].items():
            combined['formats'][label] = combined['formats'].get(label, 0) + count
    if combined['dense_cache_bytes']:
        combined['saved_cache_reduction'] = 1 - combined['saved_cache_bytes'] / combined['dense_cache_bytes']
    return combined


def accumulation_window(
    method: Any,
    batches: Sequence[Any],
    *,
    iteration: int,
    rounds: int,
    clip_gradients: Callable[[], float],
    inspect_gradients: Callable[[], Mapping[str, Any]] | None = None,
    before_microstep: Callable[[int, Any], None] | None = None,
    before_backward: Callable[[int, Any, Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    after_microstep: Callable[[int, Any, Mapping[str, Any], Mapping[str, Any]], None] | None = None,
) -> WindowResult:
    """Average losses and gradients, then clip/step/schedule/zero exactly once.

    ``batches`` must contain a complete window. Releasing each output graph
    before the next forward bounds activation memory independently of rounds.
    Any failure clears partially accumulated gradients without applying them.
    """
    if rounds < 1 or len(batches) != rounds:
        raise ValueError("A complete positive accumulation window is required")
    loss_sums: dict[str, float] = {}
    try:
        for index, batch in enumerate(batches):
            if before_microstep is not None:
                before_microstep(index, batch)
            losses, outputs, _ = method.single_train_step(batch, iteration)
            for name, tensor in losses.items():
                value = float(tensor.detach().item())
                if not math.isfinite(value):
                    raise NonFiniteWindow(f"Nonfinite {name} in microbatch {index}")
                loss_sums[name] = loss_sums.get(name, 0.0) + value
            if before_backward is not None:
                before_backward(index, batch, losses, outputs)
            method.backward(losses, outputs, grad_accum_rounds=rounds)
            if after_microstep is not None:
                after_microstep(index, batch, losses, outputs)
            del losses, outputs
        diagnostics = dict(inspect_gradients()) if inspect_gradients is not None else {}
        gradient_norm = float(clip_gradients())
        if not math.isfinite(gradient_norm):
            raise NonFiniteWindow("Nonfinite accumulated gradient norm")
        method.optimizers_schedulers_step(iteration)
    except BaseException:
        method.optimizers_zero_grad(iteration)
        raise
    method.optimizers_zero_grad(iteration)
    return WindowResult(
        losses={name: value / rounds for name, value in loss_sums.items()},
        gradient_norm=gradient_norm,
        diagnostics=diagnostics,
        microbatches=rounds,
    )
