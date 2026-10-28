"""Descriptive comparisons; execution validity is never statistical equivalence."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from compare_probes_v2 import event_key, repetition_errors
from flash_control import validate_calls
from common import atomic as atomic_json, digest
from local_policy import SCOPE, require_report_domain
from tensor_evidence import record_metrics


def validate_document(doc: dict[str, Any]) -> None:
    require_report_domain(doc)
    if doc.get('status') != 'complete' or doc.get('schema_version') != 2:
        raise ValueError('Incomplete or incompatible probe')
    if not doc.get('source_fingerprint') or not doc.get('config_fingerprint'):
        raise ValueError('Missing source/config identity')
    if doc.get('model', {}).get('optimizer_updates') != 0 or doc['model'].get('gradient_clipping_calls') != 0:
        raise ValueError('Backward-only diagnostic unexpectedly updated parameters')
    repetitions = doc.get('repetitions', [])
    if len(repetitions) not in (1, 2):
        raise ValueError('Missing repetitions')
    validate_calls(doc.get('flash_calls', []), 30 * len(repetitions))
    for index, rep in enumerate(repetitions):
        errors = repetition_errors(rep, str(index))
        if rep['index'] != index or len(rep.get('parameter_gradients', {})) != 825:
            errors.append('Repetition index/825-parameter coverage mismatch')
        if errors:
            raise ValueError('; '.join(errors[:8]))
        if doc['variant'] == 'Z' and not rep['compact_path_exercised']:
            raise ValueError('Dense fallback alone cannot cover Z')
        for record in required_evidence(rep):
            item = record.get('evidence')
            if not item or item['sha256'] != record['sha256'] or digest(Path(item['path'])) != item['sha256']:
                raise ValueError('Missing or changed tensor evidence')


def required_evidence(rep: Any) -> Any:
    yield from rep['parameter_gradients'].values()
    yield rep['loss']
    yield from rep['prediction'].values()
    for event in rep['events']:
        # Additional boundary captures supply their own PT evidence.
        if 'gradient' in event.get('tensors', {}) and not event['kind'].startswith('boundary_'):
            yield event['tensors']['gradient']


def paired_state(left: Any, right: Any) -> Any:
    for key in ('input_state', 'rng_after', 'parameters_after_sha256', 'buffers_after_sha256', 'batch_after_sha256'):
        if left[key] != right[key]:
            raise ValueError('Unpaired input/state: ' + key)
    if left['timesteps']['sha256'] != right['timesteps']['sha256']:
        raise ValueError('Noise timesteps changed')


def tensor_mapping(rep: Any) -> Any:
    result = {'loss': rep['loss']}
    result.update({'prediction/' + k: v for k, v in rep['prediction'].items()})
    result.update({'parameter/' + k: v for k, v in rep['parameter_gradients'].items()})
    for event in rep['events']:
        for name, value in event['tensors'].items():
            key = 'event/' + '/'.join(str(x) for x in event_key(event)) + '/' + name
            if key in result:
                raise ValueError('Duplicate event tensor key')
            result[key] = value
    return result


def causal_summary(left: Any, right: Any) -> Any:
    """Follow output-to-input block order, never infer causality from hook time."""
    a, b = tensor_mapping(left), tensor_mapping(right)
    candidates = []
    for layer in reversed(range(30)):

        def same(kind: Any, layer: int = layer) -> Any:
            keys = [k for k in a if k.startswith('event/backward/' + kind + '/blocks.' + str(layer) + '/')]
            return bool(keys) and all(k in b and a[k]['sha256'] == b[k]['sha256'] for k in keys)

        if same('block_output_grad') and not same('block_input_grad'):
            candidates.append(
                dict(layer=layer, boundary='block-output-to-block-input', establishes_operator_cause=False))
    return candidates


def compare(paths: list[Path], output: Path) -> Any:
    import torch
    torch.set_num_threads(4)
    docs = [json.loads(p.read_text()) for p in paths]
    if len(docs) != 6:
        raise ValueError('Expected three processes per format')
    for doc in docs:
        validate_document(doc)
    for key in ('source_fingerprint', 'config_fingerprint', 'tokens', 'seed', 'backend_settings', 'model'):
        if any(doc[key] != docs[0][key] for doc in docs):
            raise ValueError('Incompatible model report: ' + key)
    for policy in ('dense', 'Z'):
        policy_docs = [doc for doc in docs if doc['variant'] == policy]
        if len(policy_docs) != 3 or any(len(doc['repetitions']) != 2 for doc in policy_docs):
            raise ValueError('Incomplete repetition matrix')
    if len({doc['process_nonce'] for doc in docs}) != 6:
        raise ValueError('Independent process identities were reused')
    runs = [(doc, rep) for doc in docs for rep in doc['repetitions']]
    rows, pair_summary = [], []
    fields = [
        'pair', 'group', 'process_relation', 'tensor', 'byte_equal', 'finite', 'changed_elements', 'elements',
        'max_absolute', 'relative_l2', 'relative_l2_status', 'metric_available', 'reason', 'absolute_l2', 'reference_l2'
    ]
    with (output / 'tensor-differences.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, ((ld, left), (rd, right)) in enumerate(itertools.combinations(runs, 2)):
            paired_state(left, right)
            group = '/'.join(sorted((ld['variant'], rd['variant']), key=lambda x: x != 'dense'))
            relation = 'within-process' if ld['process_nonce'] == rd['process_nonce'] else 'cross-process'
            a, b = tensor_mapping(left), tensor_mapping(right)
            if set(a) != set(b):
                raise ValueError('Missing or extra boundary/parameter tensor')
            event_order_equal = [event_key(e) for e in left['events']] == [event_key(e) for e in right['events']]
            differing = []
            pair_max = 0.
            # Dictionary iteration preserves left observation order; event sequence is independently compared.
            for key in a:
                result = record_metrics(a[key], b[key])
                writer.writerow(dict(pair=number, group=group, process_relation=relation, tensor=key, **result))
                if not result['byte_equal']:
                    differing.append(key)
                if result.get('relative_l2') is not None:
                    pair_max = max(pair_max, result['relative_l2'])
            record = dict(pair=number,
                          group=group,
                          process_relation=relation,
                          left=dict(process=ld['process_nonce'], repetition=left['index']),
                          right=dict(process=rd['process_nonce'], repetition=right['index']),
                          tensor_byte_equal=not differing,
                          event_order_equal=event_order_equal,
                          differing_tensors=differing,
                          max_tensor_relative_l2=pair_max,
                          dataflow_candidates=causal_summary(left, right))
            pair_summary.append(record)
            rows.append(dict(pair=number, group=group, max_relative_l2=pair_max, changed_tensors=len(differing)))
            stream.flush()
            atomic_json(output / 'comparison-progress.json', pair_summary)
    result = dict(acceptance_scope=SCOPE,
                  execution_complete=True,
                  accepted=True,
                  execution_valid=True,
                  default_model_acceptance=False,
                  statistical_equivalence=None,
                  observed_byte_repeatable=all(x['tensor_byte_equal'] and x['event_order_equal'] for x in pair_summary),
                  tokens=docs[0]['tokens'],
                  processes=6,
                  backwards=12,
                  pairs=pair_summary,
                  note='Pairs share repetitions and are not independent statistical samples',
                  sources=[dict(path=str(p), sha256=digest(p)) for p in paths])
    atomic_json(output / 'result.json', result)
    plot(rows, output)
    return result


def plot(rows: Any, output: Any) -> Any:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ('dense/dense', 'Z/Z', 'dense/Z')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, group in enumerate(groups):
        points = [r for r in rows if r['group'] == group]
        axes[0].scatter([index] * len(points), [r['max_relative_l2'] for r in points], alpha=.5)
        axes[1].scatter([index] * len(points), [r['changed_tensors'] for r in points], alpha=.5)
    for axis in axes:
        axis.set_xticks(range(3), groups)
    axes[0].set_ylabel('Maximum tensor relative L2')
    axes[1].set_ylabel('Byte-different tensors')
    fig.suptitle('Ordinary-mode diagnostic; overlapping pairs, no equivalence certification')
    fig.tight_layout()
    fig.savefig(output / 'repeatability.png', dpi=160)
    fig.savefig(output / 'repeatability.svg')
    plt.close(fig)

    import re
    values: dict[tuple[str, int], list[float]] = {(group, layer): [] for group in groups for layer in range(30)}
    with (output / 'tensor-differences.csv').open() as stream:
        for row in csv.DictReader(stream):
            match = re.search(r'blocks\.(\d+)(?:/|\.)', row['tensor'])
            if match and row['relative_l2'] and int(match[1]) < 30:
                values[row['group'], int(match[1])].append(float(row['relative_l2']))
    fig, axis = plt.subplots(figsize=(11, 4))
    for group in groups:
        axis.plot(range(30), [max(values[group, layer], default=0.) for layer in range(30)], label=group)
    axis.set_xlabel('Wan layer')
    axis.set_ylabel('Maximum observed tensor relative L2')
    axis.set_title('Layerwise observed differences; no statistical equivalence bound')
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'layerwise-differences.png', dpi=160)
    plt.close(fig)
