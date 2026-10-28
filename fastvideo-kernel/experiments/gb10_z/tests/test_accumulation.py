"""CPU regression checks for real optimizer-window semantics, without CUDA imports."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from accumulation import (NonFiniteWindow, accumulation_window, combine_cache_metrics,
                                  empty_cache_metrics, record_saved_cache)
from train_entry import parse_args


class ToyMethod:
    def __init__(self, *, learning_rate=0.2):
        self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float64))
        self.optimizer = torch.optim.SGD([self.weight], lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0 / (step + 1))
        self.calls = []
        self.backward_factors = []
        self.optimizer.zero_grad(set_to_none=True)

    def single_train_step(self, batch, iteration):
        self.calls.append(("forward", iteration))
        features, target = batch
        loss = (torch.dot(self.weight, features) - target).square()
        return {"total_loss": loss}, {}, {}

    def backward(self, losses, outputs, *, grad_accum_rounds):
        self.calls.append(("backward", grad_accum_rounds))
        self.backward_factors.append(grad_accum_rounds)
        (losses["total_loss"] / grad_accum_rounds).backward()

    def optimizers_schedulers_step(self, iteration):
        self.calls.append(("optimizer", iteration))
        self.optimizer.step()
        self.calls.append(("scheduler", iteration))
        self.scheduler.step()

    def optimizers_zero_grad(self, iteration):
        self.calls.append(("zero", iteration))
        self.optimizer.zero_grad(set_to_none=True)


def batches():
    return [(torch.tensor([i / 4, 1 - i / 5], dtype=torch.float64), torch.tensor(i / 10, dtype=torch.float64))
            for i in range(1, 5)]


def test_four_microbatches_match_average_gradient_and_one_sgd_update():
    method = ToyMethod()
    reference_weight = method.weight.detach().clone().requires_grad_()
    reference_losses = [(torch.dot(reference_weight, x) - y).square() for x, y in batches()]
    reference_loss = torch.stack(reference_losses).mean()
    reference_loss.backward()
    expected_gradient = reference_weight.grad.clone()
    expected_weight = reference_weight.detach() - 0.2 * expected_gradient
    seen_gradients = []

    def clip():
        method.calls.append(("clip", 1))
        seen_gradients.append(method.weight.grad.detach().clone())
        return method.weight.grad.norm().item()

    result = accumulation_window(method, batches(), iteration=1, rounds=4, clip_gradients=clip)
    torch.testing.assert_close(seen_gradients[0], expected_gradient, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(method.weight, expected_weight, rtol=1e-14, atol=1e-14)
    assert result.losses["total_loss"] == pytest.approx(reference_loss.item())
    assert result.microbatches == 4
    assert method.backward_factors == [4, 4, 4, 4]
    assert [name for name, _ in method.calls] == ["forward", "backward"] * 4 + [
        "clip", "optimizer", "scheduler", "zero"]
    assert method.weight.grad is None
    assert method.scheduler.last_epoch == 1


def test_ga_one_preserves_a_single_unscaled_update():
    method = ToyMethod()
    x, y = batches()[0]
    original = method.weight.detach().clone().requires_grad_()
    loss = (torch.dot(original, x) - y).square()
    loss.backward()
    result = accumulation_window(method, [(x, y)], iteration=1, rounds=1,
                                 clip_gradients=lambda: method.weight.grad.norm().item())
    torch.testing.assert_close(method.weight, original.detach() - 0.2 * original.grad)
    assert method.backward_factors == [1]
    assert result.losses["total_loss"] == loss.item()


def test_clip_applies_to_the_accumulated_gradient_once():
    method = ToyMethod()
    original = method.weight.detach().clone()
    clipped_gradient = []

    def clip():
        value = torch.nn.utils.clip_grad_norm_([method.weight], max_norm=0.03)
        clipped_gradient.append(method.weight.grad.detach().clone())
        return value.item()

    result = accumulation_window(method, batches(), iteration=1, rounds=4, clip_gradients=clip)
    assert len(clipped_gradient) == 1
    assert result.gradient_norm > 0.03
    assert clipped_gradient[0].norm().item() <= 0.03
    torch.testing.assert_close(method.weight, original - 0.2 * clipped_gradient[0])


def test_incomplete_window_is_rejected_before_forward_or_optimizer():
    method = ToyMethod()
    with pytest.raises(ValueError, match="complete"):
        accumulation_window(method, batches()[:3], iteration=1, rounds=4, clip_gradients=lambda: 0.0)
    assert method.calls == []


def test_nonfinite_microbatch_discards_partial_window():
    method = ToyMethod()
    original = method.weight.detach().clone()
    bad_batches = batches()
    bad_batches[2] = (bad_batches[2][0], torch.tensor(float("nan"), dtype=torch.float64))
    with pytest.raises(NonFiniteWindow, match="microbatch 2"):
        accumulation_window(method, bad_batches, iteration=1, rounds=4, clip_gradients=lambda: 0.0)
    torch.testing.assert_close(method.weight, original)
    assert method.weight.grad is None
    assert not any(name in ("optimizer", "scheduler") for name, _ in method.calls)


def test_nonfinite_accumulated_norm_does_not_step():
    method = ToyMethod()
    original = method.weight.detach().clone()
    with pytest.raises(NonFiniteWindow, match="gradient"):
        accumulation_window(method, batches(), iteration=1, rounds=4, clip_gradients=lambda: float("inf"))
    torch.testing.assert_close(method.weight, original)
    assert method.weight.grad is None
    assert method.scheduler.last_epoch == 0


def test_microstep_callbacks_run_in_sample_order_with_raw_losses():
    method = ToyMethod()
    events = []
    result = accumulation_window(
        method, batches(), iteration=7, rounds=4,
        clip_gradients=lambda: method.weight.grad.norm().item(),
        inspect_gradients=lambda: {"gradient": method.weight.grad.detach().clone()},
        before_microstep=lambda index, batch: events.append(("before", index)),
        after_microstep=lambda index, batch, losses, outputs: events.append(
            ("after", index, losses["total_loss"].item())),
    )
    assert [(event[0], event[1]) for event in events] == [item for i in range(4)
                                                       for item in [("before", i), ("after", i)]]
    assert result.losses["total_loss"] == pytest.approx(sum(event[2] for event in events if event[0] == "after") / 4)
    assert torch.isfinite(result.diagnostics["gradient"]).all()


def test_callback_failure_clears_partial_gradients_without_update():
    method = ToyMethod()
    original = method.weight.detach().clone()

    def fail_after_second(index, batch, losses, outputs):
        if index == 1:
            raise RuntimeError("capture failure")

    with pytest.raises(RuntimeError, match="capture failure"):
        accumulation_window(method, batches(), iteration=1, rounds=4,
                            clip_gradients=lambda: 0.0, after_microstep=fail_after_second)
    torch.testing.assert_close(method.weight, original)
    assert method.weight.grad is None
    assert not any(name == "optimizer" for name, _ in method.calls)


def test_sample_matched_cli_defaults_have_distinct_update_budgets():
    common = ["--variant", "save", "--seed", "2001", "--run-dir", "/tmp/unused-cli-test"]
    ga1 = parse_args(["--artifact-root", "/tmp/test-artifacts", *common, "--grad-accum-steps", "1"])
    ga4 = parse_args(["--artifact-root", "/tmp/test-artifacts", *common, "--grad-accum-steps", "4"])
    assert ga1.samples == ga4.samples == 800
    assert ga1.samples // ga1.grad_accum_steps == 800
    assert ga4.samples // ga4.grad_accum_steps == 200
    assert ga1.lr == ga4.lr == 1e-6
    assert ga1.eval_every_samples == ga4.eval_every_samples == 200


def test_first_window_verification_cli_can_be_enabled_without_cuda_imports():
    args = parse_args(["--artifact-root", "/tmp/test-artifacts", "--variant", "save", "--seed", "2001", "--samples", "4",
                       "--verify-first-window", "--run-dir", "/tmp/unused-cli-test"])
    assert args.verify_first_window
    assert args.samples == args.grad_accum_steps == 4


def test_backward_phase_callback_runs_after_forward_before_backward():
    method = ToyMethod()

    def before_backward(index, batch, losses, outputs):
        method.calls.append(("set-backward-phase", index))

    accumulation_window(method, batches(), iteration=1, rounds=4, before_backward=before_backward,
                        clip_gradients=lambda: method.weight.grad.norm().item())
    assert [name for name, _ in method.calls[:12]] == ["forward", "set-backward-phase", "backward"] * 4


def test_saved_cache_metadata_accounts_for_cross_attention_without_tensor_unpack():
    class Context:
        forward_consistent_dv = True
        dv_stats_mode = 'save'
        dv_stats_format = 'dense'
        dv_overflow_fallback = False
        dv_stats_storage_bytes = 2 * 3 * 257 * (8 * 4 + 4)

        @property
        def saved_tensors(self):
            raise AssertionError('Metadata accounting must not unpack checkpoint-saved tensors')

    cache = empty_cache_metrics()
    record_saved_cache(cache, Context(), (2, 3, 257, 128), (2, 3, 241, 128))
    assert cache['saved_cache_bytes'] == cache['dense_cache_bytes'] == Context.dv_stats_storage_bytes
    assert cache['saved_statistics_calls'] == cache['attention_forward_calls'] == 1
    assert cache['saved_cache_reduction'] == 0.0


def test_cache_accumulation_reports_weighted_reduction_and_fallback_calls():
    dense = 12 * 31200 * (975 * 4 + 4)
    compact_context = SimpleNamespace(forward_consistent_dv=True, dv_stats_mode='save',
                                      dv_stats_format='change_points', dv_overflow_fallback=False,
                                      dv_stats_storage_bytes=dense // 4)
    replay_context = SimpleNamespace(forward_consistent_dv=True, dv_stats_mode='save',
                                     dv_stats_format='dense', dv_overflow_fallback=True,
                                     dv_stats_storage_bytes=dense)
    first, second = empty_cache_metrics(), empty_cache_metrics()
    record_saved_cache(first, compact_context, (1, 12, 31200, 128), (1, 12, 31200, 128))
    record_saved_cache(second, replay_context, (1, 12, 31200, 128), (1, 12, 31200, 128))
    result = combine_cache_metrics([first, second])
    assert result['saved_statistics_calls'] == 2
    assert result['overflow_fallback_calls'] == 1
    assert result['formats'] == {'change_points': 1, 'dense': 1}
    assert result['saved_cache_reduction'] == pytest.approx(0.375)
    assert result['maximum_saved_cache_bytes_per_attention'] == dense
    assert result['minimum_saved_cache_reduction_per_attention'] == 0.0


def test_legacy_or_recompute_cache_is_explicitly_unavailable():
    cache = empty_cache_metrics()
    record_saved_cache(cache, None, (1, 12, 8192, 128), (1, 12, 8192, 128))
    record_saved_cache(cache, SimpleNamespace(forward_consistent_dv=True, dv_stats_mode='recompute'),
                       (1, 12, 8192, 128), (1, 12, 8192, 128))
    assert cache['attention_forward_calls'] == 2
    assert cache['saved_statistics_calls'] == 0
    assert cache['saved_cache_reduction'] is None


@pytest.mark.parametrize("extra", [["--samples", "799"], ["--eval-every-samples", "201"], ["--lr", "nan"]])
def test_cli_rejects_truncated_windows_and_nonfinite_learning_rate(extra):
    with pytest.raises(SystemExit) as error:
        parse_args(["--artifact-root", "/tmp/test-artifacts", "--variant", "save", "--seed", "2001", "--run-dir", "/tmp/unused-cli-test", *extra])
    assert error.value.code == 2
