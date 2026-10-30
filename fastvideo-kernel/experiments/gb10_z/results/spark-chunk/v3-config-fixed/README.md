# YAML chunk setting fixed: three paired runs passed

Production fix: `f8739ad56b06aea9ea60937f2eacd058e7b753a6`.
`load_run_config` now passes `training.optimizer.foreach_chunk_size` into
OptimizerConfig. The default remains zero. The optimizer implementation, kernel,
FA2 policy, and training arithmetic were not changed.

Tests: 35 configuration/optimizer tests passed, including three new YAML-to-factory
cases (zero, 64, and fused conflict). Default parsing is also asserted. Pre-commit
passed for the changed files in an equivalent temporary worktree with a valid
Python package directory name. In the original checkout, mypy rejected the hyphenated
checkout name before type checking; all other hooks passed. The checked source
file was byte-identical. Historical experiment CPU suites were not rerun.

## Numerical and memory results

Six independent production-entrypoint runs: seed 1000, 8192 tokens, G1,
full checkpointing, strict determinism, dense saved statistics, TMA,
BF16 compute / FP32 parameters, LR 1e-6, no warmup, clip 1.0. Each run
performed four real updates. The single-row parquet cycles the same original
first sample as the historical private baseline. See input-receipt.json for hashes.
No input IDs or tensor data are published.

Run order was frozen with seed 20261030: 64/0, 0/64, 0/64. Every requested
chunk value matched both the parsed configuration and actual optimizer class.
All three comparisons passed: initial parameter/RNG hashes, per-step loss,
gradient norm, method RNG state, and all 825 post-update parameter hashes.
All three dense runs also matched the historical four-update dense endpoint
at all 825 parameters. There were no retries or failed pairs.

| Measurement | chunk 0 | chunk 64 |
| --- | ---: | ---: |
| Peak allocated, each round (GiB) | 48.672398 | 45.438960 |
| Peak reserved, each round (GiB) | 49.937500 | 47.994141 |
| Median of the three run medians, steps 2–4 (seconds) | 9.855083 | 9.829143 |

Each pair saved exactly 3,471,878,144 allocated bytes (3.233438 GiB).
These are formal-entrypoint peaks, not a reproduction of the historical private
loop's absolute 26.8/23.6 GiB values. The formal path initializes negative
conditioning with the restored text encoder; this batch did not re-attribute
that path's allocations. It demonstrates a similar absolute memory reduction
without claiming the old absolute peak or its percentage reduction.

The timing values are diagnostic Trainer `step_time_sec`, using updates 2–4;
update 1 remains in the raw data. Hashing at step-end is outside that timer.
This is not a CUDA-event or phase-by-phase benchmark and does not establish a
speed improvement. Telemetry covers the entire process, including model loading
and hashing; its busy-sample median clock was 1794 MHz in every run. One run
also recorded a 2398 MHz sample. No samples or runs were discarded to favor timing.
Raw clock/temperature/power telemetry is retained with sequential sample indices.

## Reproduction

Use the existing project environment and complete matching model. First create
an isolated input directory containing exactly the original row 0, preserving
its Arrow schema and tensor values (keep the generated parquet local):

```python
from pathlib import Path
import pyarrow.parquet as pq

source = Path("SOURCE_PARQUET")
output = Path("NEW_ONE_ROW_DIRECTORY")
output.mkdir(exist_ok=False)
row = pq.read_table(source).slice(0, 1)
pq.write_table(row, output / "data.parquet")
assert pq.read_table(output / "data.parquet").equals(row)
```

Execute each command in a new process and empty output directory, in the order
specified by schedule.json. Coordinate exclusive GPU use through both scheduler
locks and restore competing services after every run.

```bash
python fastvideo-kernel/experiments/gb10_z/code/chunk_entry_preflight.py \
  --steps 4 --chunk 0 --model "$MODEL" --data "$ONE_ROW_DIRECTORY" \
  --output "$NEW_DENSE_OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
python fastvideo-kernel/experiments/gb10_z/code/chunk_entry_preflight.py \
  --steps 4 --chunk 64 --model "$MODEL" --data "$ONE_ROW_DIRECTORY" \
  --output "$NEW_CHUNK_OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
python fastvideo-kernel/experiments/gb10_z/code/chunk_pair_compare.py \
  "$NEW_DENSE_OUTPUT/observation.json" "$NEW_CHUNK_OUTPUT/observation.json"
```

Generated YAML explicitly uses `training.tracker.trackers: [none]` and the
production optimizer setting; no optimizer monkeypatch is used by the harness.
Actual optimizer identity is recorded. Source fingerprints and per-run environment
versions are included. Runtime: Torch 2.12.0+cu130, Triton 3.7.0,
FlashAttention 2.8.3+cu130torch2.12, GB10 SM121.

The active compatibility service and manual GPU hold were stopped and restored
per run. vLLM, Duty, and Ollama were inactive. The thermal guard and clock policy
were unchanged. Both GPU locks were held for each training run.

## Scope and retained failures

This accepts only the three four-update pairs at 8192/G1. It does not certify
31200 phase attribution, 200-step equality, or an optimal chunk size. Those remain
unexecuted in this batch. No production default changed, no PR opened. Previous
missing-input, ignored-option and mismatched-data results remain untouched.
All uploaded files are scalar/hash evidence; no model weights, input parquet,
video IDs, raw parameter tensors or machine paths are included. SHA256SUMS
covers this published directory; it does not replace hashes of original raw logs.
