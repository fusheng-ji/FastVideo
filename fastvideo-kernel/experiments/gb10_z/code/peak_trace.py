"""Estimate the transient peak from local allocator allocation-event sizes.

Allocation events omit extra bytes in unsplit reused blocks. This estimate is
not the exact allocated peak; peak_capture.py obtains a live snapshot for that.
Only aggregate bytes and source basenames are exported. Raw snapshots are trusted
local pickle files and must never be uploaded with results.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle


def reconstruct(start, end):
    live = {b['address']: {'size': b['size'], 'frames': b.get('frames', [])}
            for s in start['segments'] for b in s['blocks'] if b['state'] == 'active_allocated'}
    total = sum(b['size'] for b in live.values())
    peak, peak_live = total, dict(live)
    # Trace begins before the initial snapshot: skip already accounted events.
    offset = len(start['device_traces'][0])
    for event in end['device_traces'][0][offset:]:
        action = event['action']
        if action == 'alloc':
            assert event['addr'] not in live, 'Duplicate live address'
            live[event['addr']] = event
            total += event['size']
        elif action == 'free_requested':
            old = live.pop(event['addr'])
            total -= old['size']
        if total > peak:
            peak, peak_live = total, dict(live)
    return peak, peak_live


def summarize(start, end, addresses):
    peak, blocks = reconstruct(start, end)
    categories, origins = Counter(), Counter()
    for address, block in blocks.items():
        known = addresses.get(str(address))
        # Rounded allocator size may exceed tensor storage slightly.
        if known and known['bytes'] <= block['size'] < known['bytes'] + 512:
            categories[known['category']] += known['bytes']
            categories['allocator_rounding'] += block['size'] - known['bytes']
        else:
            categories['other_live_and_temporaries'] += block['size']
            frames = block.get('frames', [])
            frame = frames[0] if frames else {}
            origin = '{}:{}:{}'.format(Path(frame.get('filename', 'unrecorded')).name, frame.get('line', 0), frame.get('name', 'unrecorded'))
            origins[origin] += block['size']
    assert sum(categories.values()) == peak
    return dict(trace_peak_before_unsplit_block_correction=peak, category_bytes=dict(categories), other_allocation_origins=dict(origins.most_common()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite'
    def load(name):
        with (a.input / f'step-{a.step}-{name}.pickle').open('rb') as stream:
            return pickle.load(stream)
    result = summarize(load('start'), load('optimizer'), json.loads((a.input / f'step-{a.step}-addresses.json').read_text()))
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
