"""Diagnostic-only snapshot at the observed foreach AdamW sqrt allocation.

Wraps a private experiment, not production code. The original operation executes
once with unchanged arguments and returns the original tensors. Snapshot overhead
is excluded from the independent paired timing suite.
"""
import json
import sys


def main():
    import torch
    import peak_memory
    import spark_peak
    original_recorder = peak_memory.PhaseRecorder
    original_sqrt = torch._foreach_sqrt
    original_set_device = torch.cuda.set_device
    active = {}

    def set_device(*args, **kwargs):
        result = original_set_device(*args, **kwargs)
        torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=200000)
        return result

    class Recorder(original_recorder):
        def begin(self, step):
            super().begin(step)
            active['recorder'] = self
            active['phase'] = 'start'

        def mark(self, phase):
            active['phase'] = phase
            return super().mark(phase)

    def sqrt(*args, **kwargs):
        result = original_sqrt(*args, **kwargs)
        recorder = active.get('recorder')
        if recorder and recorder.enabled and active.get('phase') == 'clip':
            snapshot = torch.cuda.memory._snapshot()
            addresses = peak_memory.inventory(recorder.parameters, recorder.optimizer)
            for tensor in result:
                storage = peak_memory.local(tensor).untyped_storage()
                assert storage.data_ptr() not in addresses
                addresses[storage.data_ptr()] = {'category': 'adam_sqrt_temporaries', 'bytes': storage.nbytes()}
            categories = {}
            other_origins = {}
            for segment in snapshot['segments']:
                for block in segment['blocks']:
                    if block['state'] != 'active_allocated':
                        continue
                    known = addresses.get(block['address'])
                    if known:
                        assert block['size'] >= known['bytes']
                        for category, size in ((known['category'], known['bytes']), ('allocator_rounding', block['size']-known['bytes'])):
                            categories[category] = categories.get(category, 0) + size
                    else:
                        categories['other_live_tensors'] = categories.get('other_live_tensors', 0) + block['size']
                        frames = block.get('frames', [])
                        frame = frames[0] if frames else {}
                        from pathlib import Path
                        origin = '{}:{}:{}'.format(Path(frame.get('filename', 'unrecorded')).name, frame.get('line', 0), frame.get('name', 'unrecorded'))
                        other_origins[origin] = other_origins.get(origin, 0) + block['size']
            report = dict(other_allocation_origins=other_origins, category_bytes=categories, allocated=torch.cuda.memory_allocated(), peak_allocated=torch.cuda.max_memory_allocated(), step=recorder.step)
            assert sum(categories.values()) == report['allocated']
            assert report['allocated'] == report['peak_allocated'], 'This interception did not capture the true peak'
            torch.cuda.memory._dump_snapshot(str(recorder.output/f'step-{recorder.step}-actual-peak.pickle'))
            (recorder.output/f'step-{recorder.step}-peak-attribution.json').write_text(json.dumps(report, indent=2)+'\n')
        return result

    assert '--history' in sys.argv, 'This entrypoint is only for attribution, never paired timing'
    peak_memory.PhaseRecorder = Recorder
    torch._foreach_sqrt = sqrt
    torch.cuda.set_device = set_device
    try:
        spark_peak.main()
    finally:
        torch._foreach_sqrt = original_sqrt
        torch.cuda.set_device = original_set_device
        torch.cuda.memory._record_memory_history(enabled=None)
        peak_memory.PhaseRecorder = original_recorder


if __name__ == '__main__':
    main()
