"""Experimental bounded AdamW foreach lists, preserving per-tensor operations."""
import importlib
import json
from pathlib import Path


def chunked(original, capacity=64):
    assert isinstance(capacity, int) and capacity > 0

    def run(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, *args, **kwargs):
        run.invocations += 1
        run.tensor_counts.append(len(params))
        assert all(len(x) == len(params) for x in (grads, exp_avgs, exp_avg_sqs, state_steps))
        assert len(max_exp_avg_sqs) in (0, len(params))
        for offset in range(0, len(params), capacity):
            inputs = [x[offset:offset+capacity] for x in (params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)]
            original(*inputs, *args, **kwargs)
    run.invocations = 0
    run.tensor_counts = []
    return run


def main():
    import spark_peak
    adam = importlib.import_module('torch.optim.adam')
    original = adam._multi_tensor_adam
    original_parse = spark_peak.parse_args
    selected = {}

    def parse():
        args = original_parse()
        assert args.variant == 'baseline', 'Chunking is the only modified factor'
        args.variant = 'optimizer-chunk64'
        selected['args'] = args
        return args

    spark_peak.parse_args = parse
    wrapped = chunked(original)
    adam._multi_tensor_adam = wrapped
    try:
        spark_peak.main()
    finally:
        adam._multi_tensor_adam = original
        spark_peak.parse_args = original_parse
        if 'args' in selected:
            path = selected['args'].output/'result.json'
            if path.exists():
                result = json.loads(path.read_text())
                result['chunk_capacity'] = 64
                result['foreach_invocations'] = wrapped.invocations
                result['foreach_tensor_counts'] = wrapped.tensor_counts
                if result['status'] == 'completed':
                    assert wrapped.invocations == selected['args'].warmup + selected['args'].measure
                result['wrapper_sha256'] = spark_peak.digest(__file__)
                result['optimizer_implementation_sha256'] = spark_peak.digest(adam.__file__)
                path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
