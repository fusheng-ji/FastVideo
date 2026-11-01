# GB10 retest: encoder release and chunked AdamW

Integration commit `21b2fdac3beec4f2c588a8c93dc9cc46b0275258`, parent `376a1f6476d1f209440a6746f7209986a39494bf`; no production files changed. Control commit `d1235c66a`. These are new measurements on GB10 SM121, Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12, driver 580.173.02.

CPU tests with GPU visible, idle, and both locks held: train utilities 36 passed, GB10 experiment 130 passed, FA2 policy 18 passed. Formal entrypoint smoke: one update, chunk=0, `training.tracker.trackers: [none]`, 825 parameter hashes.

Queue status: **completed**. Encoder result: control refs 194, release refs 0; allocated 28739451904 to 6015810560 bytes (difference 21.163040 GiB). Comparisons and failures are recorded in `summary.json`. Each run used a fresh directory, both locks, seed 1000, and restored previously active competitor services after exit. A one-row parquet is repeated; input fingerprints are included. No weight, video, raw capture or machine path is included.

## Recorded results

| Run | Updates | Allocated peak GiB | Reserved peak GiB | First phase observing peak |
|---|---:|---:|---:|---|
| smoke | 1 | 27.51424 | 28.91602 | optimizer_end step 1 |
| encoder-control | 2 | 48.67240 | 49.93750 | optimizer_end step 1 |
| encoder-release | 2 | 27.51424 | 30.87305 | optimizer_end step 1 |
| B-round-0-chunk-0 | 4 | 27.51424 | 30.87305 | optimizer_end step 1 |
| B-round-0-chunk-64 | 4 | 24.28059 | 26.97266 | backward_end step 2 |
| B-round-1-chunk-64 | 4 | 24.28059 | 26.97266 | backward_end step 2 |
| B-round-1-chunk-0 | 4 | 27.51424 | 30.87305 | optimizer_end step 1 |
| B-round-2-chunk-0 | 4 | 27.51424 | 30.87305 | optimizer_end step 1 |
| B-round-2-chunk-64 | 4 | 24.28059 | 26.97266 | backward_end step 2 |
| C-round-0-chunk-0 | 2 | 32.87558 | 42.79688 | backward_end step 2 |
| C-round-0-chunk-64 | 2 | 32.87558 | 37.70508 | backward_end step 2 |
| C-round-1-chunk-64 | 2 | 32.87558 | 37.70508 | backward_end step 2 |
| C-round-1-chunk-0 | 2 | 32.87558 | 42.79688 | backward_end step 2 |
| C-round-2-chunk-0 | 2 | 32.87558 | 42.79688 | backward_end step 2 |
| C-round-2-chunk-64 | 2 | 32.87558 | 37.70508 | backward_end step 2 |
| D-chunk-64 | 200 | 24.28059 | 26.97266 | backward_end step 2 |
| D-chunk-0 | 200 | 27.51424 | 30.87305 | optimizer_end step 1 |

Phase peak counters are cumulative; phase is where the peak was first observed. GPU clocks were recorded, not fixed. Step times include observation costs and are not certified speedups.

B (8192 if B, 31200 if C) allocated/reserved GiB saved by chunk=64 per pair: [(3.233652, 3.900391), (3.233652, 3.900391), (3.233652, 3.900391)]. Historical comparison: allocated 27.514 to 24.281 GiB; historical saving 3.233 GiB. Saving differences (measured minus historical), GiB: [0.000652, 0.000652, 0.000652].

C (8192 if B, 31200 if C) allocated/reserved GiB saved by chunk=64 per pair: [(0.0, 5.091797), (0.0, 5.091797), (0.0, 5.091797)]. Historical comparison: allocated 32.876 GiB both; reserved 42.797 to 37.705 GiB. Saving differences versus historical allocated 0 and reserved 5.092 GiB: [(0.0, -0.000203), (0.0, -0.000203), (0.0, -0.000203)].

Pair and 200-update comparison results: `[{"status": "passed", "failures": [], "stage": "B", "round": 0}, {"status": "passed", "failures": [], "stage": "B", "round": 1}, {"status": "passed", "failures": [], "stage": "B", "round": 2}, {"status": "passed", "failures": [], "stage": "C", "round": 0}, {"status": "passed", "failures": [], "stage": "C", "round": 1}, {"status": "passed", "failures": [], "stage": "C", "round": 2}, {"status": "passed", "failures": [], "stage": "D"}]`. D compares every-step loss and gradient norm, and all 825 parameter hashes every 20 updates. If a recorded checkpoint differs, the exact first differing unobserved update is unknown. The optional size sweep was not run.

## Reproduction

Provide `$MODEL`, `$ONE_ROW_8192`, `$ONE_ROW_31200`, `$FRESH_OUTPUT`, `$LOCK_A`, `$LOCK_B` and `$PYTHON` for the local machine. For every invocation, stop competing Duty/Ollama/compatibility services after recording their state and restore them on exit. Acquire both GPU locks, verify GPU idle, and record `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,power.draw --format=csv --loop=1`. Run CPU checks with GPU visible:

```bash
$PYTHON -m pytest fastvideo/tests/train/utils/test_negative_prompt_release.py fastvideo/tests/train/utils/test_config.py fastvideo/tests/train/utils/test_optimizer_foreach_chunk.py -q --confcutdir=fastvideo/tests/train
$PYTHON -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
$PYTHON -m pytest fastvideo/tests/attention/test_fa2_determinism_policy.py -q --confcutdir=fastvideo/tests/attention
```

The entrypoint acquires the two locks; call it from the corresponding checkout with the 8192 or 31200 single-row input:

```bash
$PYTHON fastvideo-kernel/experiments/gb10_z/code/chunk_v5_entry.py --tokens "$TOKENS" --steps "$STEPS" --chunk "$CHUNK" --hash-every "$HASH_EVERY" --model "$MODEL" --data "$ONE_ROW_DATA" --output "$FRESH_OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

Smoke uses `8192/1/0/1`; release uses `8192/2/0/1`; B uses `8192/4/{0,64}/1`; C uses `31200/2/{0,64}/1`; D uses `8192/200/{0,64}/20`. For the control checkout at `d1235c66a`, replace `chunk_v5_entry.py` by `chunk_v4_entry.py` and use `8192/2/0/1`. Each script constructs a YAML with `training.tracker.trackers: [none]` and the requested optimizer chunk. Comparison order is in summary.json. No failed job is retried.

Verify sanitized artifacts with `sha256sum -c SHA256SUMS`.
