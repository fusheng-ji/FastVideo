"""Paired ordinary-mode calibration; no statistical equivalence certification."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any
from common import atomic, digest
from local_policy import SCOPE
from probe_helpers import event_key


def read_rows(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_pair(left: dict[str, Any], right: dict[str, Any]) -> None:
    for field in ('acceptance_scope', 'source_fingerprint', 'config_fingerprint', 'initial_parameters',
                  'initial_buffers_sha256', 'initial_rng', 'initial_optimizer', 'initial_scheduler'):
        if left.get(field) != right.get(field) or field not in left:
            raise RuntimeError('Unpaired initial configuration/state: ' + field)
    if left['acceptance_scope'] != SCOPE:
        raise RuntimeError('Wrong calibration domain')


def check_micro_pair(a: Any, b: Any) -> None:
    for key in ('batch_sha256', 'rng_before'):
        if a['input_state'][key] != b['input_state'][key]:
            raise RuntimeError('Unpaired training input/RNG: ' + key)
    if a['timesteps'] != b['timesteps'] or a['rng_after'] != b['rng_after']:
        raise RuntimeError('Training timesteps or RNG consumption differ')
    if [event_key(x) for x in a['events']] != [event_key(x) for x in b['events']]:
        raise RuntimeError('Event order changed; comparison must not sort events')


def compare_tensors(a: Path, b: Path, output: Path) -> Any:
    import torch
    from tensor_evidence import metrics
    torch.set_num_threads(4)
    left = json.loads((a / 'manifest.json').read_text())
    right = json.loads((b / 'manifest.json').read_text())
    if len(left) != 825 or [right_label['name']
                            for right_label in left] != [right_label['name'] for right_label in right]:
        raise RuntimeError('Missing final parameter/gradient evidence')
    rows = []
    for x, y in zip(left, right, strict=True):
        paths = (a / x['file'], b / y['file'])
        if any(digest(p) != row['sha256'] for p, row in zip(paths, (x, y), strict=True)):
            raise RuntimeError('Final tensor evidence changed')
        tensors = [torch.load(p, weights_only=True, map_location='cpu', mmap=True) for p in paths]
        result = metrics(*tensors)
        if not result['finite']:
            raise RuntimeError('Nonfinite final tensor')
        rows.append(dict(name=x['name'], **result))
        del tensors
    atomic(output, rows)
    absolute = math.sqrt(sum(row['absolute_l2']**2 for row in rows))
    reference = math.sqrt(sum(row['reference_l2']**2 for row in rows))
    return dict(byte_equal=all(row['byte_equal'] for row in rows),
                absolute_l2=absolute,
                relative_l2=absolute / reference if reference else None,
                max_absolute=max(row['max_absolute'] for row in rows))


def compare_seed(root: Path, seed: int) -> dict[str, Any]:
    directories = {label: root / 'runs' / f'{seed}-{label}' for label in ('A', 'B', 'Z')}
    audits = {label: json.loads((p / 'audit.json').read_text()) for label, p in directories.items()}
    micros = {label: read_rows(p / 'audit-microsteps.jsonl') for label, p in directories.items()}
    evals: dict[str, Any] = {}
    for label, report in audits.items():
        expected_policy = 'Z' if label == 'Z' else 'dense'
        if (not report.get('accepted') or report['status'] != 'complete' or report['repeat_id'] != label
                or report['variant'] != expected_policy or report['microbatches'] != 20
                or report['optimizer_updates'] != 20 or report['configuration']['seed'] != seed):
            raise RuntimeError('Incomplete or mislabeled calibration run')
        evals[label] = [
            right_label for right_label in read_rows(directories[label] / 'metrics.jsonl')
            if right_label['event'] == 'eval-sample'
        ]
        if len(evals[label]) != 96 or len(micros[label]) != 20:
            raise RuntimeError('Incomplete calibration budget')
    out = root / 'comparisons' / str(seed)
    out.mkdir(parents=True, exist_ok=False)
    tensor_results = {}
    for left_label, right_label in (('A', 'B'), ('A', 'Z'), ('B', 'Z')):
        check_pair(audits[left_label], audits[right_label])
        for a, b in zip(micros[left_label], micros[right_label], strict=True):
            check_micro_pair(a, b)

        def identity(rows: Any) -> Any:
            return [{k: row[k] for k in ('step', 'id', 'noise_seed', 'timestep', 'split', 'draw')} for row in rows]

        if identity(evals[left_label]) != identity(evals[right_label]):
            raise RuntimeError('Validation input pairing changed')
        tensor_results[left_label + '-' + right_label] = {
            kind:
            compare_tensors(directories[left_label] / ('final-' + kind), directories[right_label] / ('final-' + kind),
                            out / f'{left_label}-{right_label}-{kind}.json')
            for kind in ('parameters', 'gradients')
        }
    means = {
        label: {
            str(step): sum(right_label['loss'] for right_label in rows if right_label['step'] == step) / 32
            for step in (0, 10, 20)
        }
        for label, rows in evals.items()
    }
    if any(not math.isfinite(v) or v <= 0 for values in means.values() for v in values.values()):
        raise RuntimeError('Validation loss must be positive and finite for log ratios')
    result = dict(execution_complete=True,
                  paired_checks_passed=True,
                  seed=seed,
                  means=means,
                  dense_repeat_log_ratio=math.log(means['A']['20'] / means['B']['20']),
                  Z_dense_log_ratio=math.log(means['Z']['20'] / ((means['A']['20'] + means['B']['20']) / 2)),
                  final_tensor_comparisons=tensor_results,
                  statistical_equivalence=None,
                  independent_unit='seed',
                  acceptance_scope=SCOPE)
    atomic(out / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    compare_seed(args.root, args.seed)
