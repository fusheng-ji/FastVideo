# GB10 encoder release and AdamW chunk validation v5

Base `01ac554f8041aaa367497eeb30aa93965ce642c6`. Environment: GB10 SM121, torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12. Runs use the formal training entrypoint, strict determinism, seed 1000, and YAML `training.tracker.trackers: [none]`. A single parquet row is repeated.

The initial CPU collection failed because `CUDA_VISIBLE_DEVICES` was set to an empty string and the transitive Triton import needed to discover a CUDA driver. It was a test environment issue, not a finding against the change. Re-running the specified negative-prompt, config, and chunk optimizer tests with GPU visible, idle and both GPU locks held gave 36 passed (14 deprecation warnings). The previous GB10 experiment and FA2 policy tests gave 130 and 18 passed respectively.

An initial service preflight attempted to stop an absent temporary hold unit and exited before launching training. The state-based service control was corrected; the actual A run used a fresh directory. A post-run comparator then incorrectly compared timing fields with v3 and stopped before B. The 825 parameter hashes, loss and gradient norms actually match v3 for the first two updates; the comparator was corrected to compare numerical fields and its earlier failure was retained locally.

A: after negative encoding, 0 of 194 encoder tensor weakrefs remained alive. Allocated memory was 6,015,810,560 bytes, down 22,723,641,344 bytes (21.163 GiB) from v4 at the same phase. No peak snapshot was attempted. Two updates completed; the allocated training peak was about 27.51 GiB.

B, C and D are ongoing; this initial receipt does not claim results for them. E is optional and not yet run. No production files changed in this validation. The branch started from the user's production commit.

Reproduce A with a fresh output directory and both GPU locks. Before each run, stop competing services and restore their previous states after exit while keeping the thermal guard active. Collect GPU clock telemetry. Supply the corresponding model and single-row data as local inputs:

```bash
python fastvideo-kernel/experiments/gb10_z/code/chunk_v5_entry.py --tokens 8192 --steps 2 --chunk 0 --hash-every 1 --model "$MODEL" --data "$ONE_ROW_DATA" --output "$FRESH_OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

No weights, video, raw capture or machine-specific paths are included.
