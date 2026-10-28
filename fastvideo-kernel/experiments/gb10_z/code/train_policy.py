"""Private P64 train entry policy; no production config or API additions."""
from contextlib import contextmanager

from local_policy import kernel_policy


@contextmanager
def kernel_options(module, store_mode='tma', stats_format='dense', event_capacity=64, force_short=False):
    if store_mode != 'tma' or stats_format not in ('dense', 'packed_slots') or force_short:
        raise ValueError('Frozen model entry requires TMA and the selected full-length statistics')
    if stats_format == 'packed_slots' and event_capacity != 64:
        raise ValueError('P64 requires exactly 64 event values')
    with kernel_policy(module, 'Z' if stats_format == 'packed_slots' else 'dense') as audit:
        yield {'statistics': audit, 'stores': None}
