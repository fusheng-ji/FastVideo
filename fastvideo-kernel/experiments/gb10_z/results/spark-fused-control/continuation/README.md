# GB10 fused AdamW control

Base commit `45ee2455bb17f49e10f8163207ec9efbc79501ed`. New GB10 SM121 replication; Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12, CUDA 13.0, driver 580.173.02. Step 0 CPU checks: 36, 130 and 18 passed; formal entrypoint baseline smoke completed one update. Each run used YAML `training.tracker.trackers: [none]`, seed 1000 and a single repeated parquet row. Each output has its own directory. Source and input fingerprints, actual optimizer class and `fused`/`foreach` flags are recorded in JSON.

Prior feasibility and early rounds are recorded in the parent directory. Continuation runs use v2 public optimizer hooks: optimizer peak counters are reset at each optimizer step, with pre-reset peaks retained. Reported training peaks combine all segments after `on_train_start`; startup peaks (including text encoder load) and whole-run peaks are separate. Phase counters in memory.json are cumulative only between resets. Timings include instrumentation/synchronization and clocks were recorded, not fixed; do not infer a speedup.

| Run | Updates | Training peak allocated GiB | Training peak reserved GiB | Limiting phase | Optimizer step peak GiB | Median step s | Median optimizer s |
|---|---:|---:|---:|---|---:|---:|---:|
| previous-pair-8192-r2-F0 | 4 | 27.51424 | 30.87305 | optimizer_end | 27.51424 | 9.9539 | 0.5612 |
| pair-8192-r2-FU | 4 | 24.28196 | 26.71680 | backward_end | 22.22846 | 11.1062 | 0.2024 |
| pair-31200-r0-F64 | 2 | 32.87558 | 37.70508 | backward_end | 24.67871 | 63.4874 | 3.2098 |
| pair-31200-r0-FU | 2 | 32.88015 | 37.39648 | backward_end | 24.20411 | 60.2815 | 1.2727 |
| pair-31200-r0-F0 | 2 | 32.87558 | 42.79688 | backward_end | 29.48989 | 67.3990 | 7.2902 |
| pair-31200-r1-FU | 2 | 32.88015 | 37.39648 | backward_end | 24.20411 | 63.6821 | 1.2595 |
| pair-31200-r1-F0 | 2 | 32.87558 | 42.79688 | backward_end | 29.48989 | 72.8771 | 8.3726 |
| pair-31200-r1-F64 | 2 | 32.87558 | 37.70508 | backward_end | 24.67871 | 62.6119 | 3.1018 |
| pair-31200-r2-FU | 2 | 32.88015 | 37.39648 | backward_end | 24.20411 | 61.2242 | 1.2682 |
| pair-31200-r2-F64 | 2 | 32.87558 | 37.70508 | backward_end | 24.67871 | 65.9260 | 1.4526 |
| pair-31200-r2-F0 | 2 | 32.87558 | 42.79688 | backward_end | 29.48989 | 66.3779 | 5.0617 |
| long-8192-F0 | 200 | 27.51424 | 30.87305 | optimizer_end | 27.51424 | 11.5155 | 0.5614 |
| long-8192-FU | 200 | 24.28196 | 26.71680 | backward_end | 22.22846 | 11.2112 | 0.1989 |
| long-8192-F64 | 200 | 24.28059 | 26.97266 | backward_end | 22.70307 | 11.5012 | 0.5570 |

Queue status: completed_with_prior_failure; failures: []; earlier infrastructure failure: [{"name": "pair-8192-r2-F64", "arm": "F64", "exit_code": 1, "reason": "attempt failed; no retry"}]; F0/F64 bitwise comparisons: [{"status": "passed", "failures": [], "stage": "8192-round-2", "service_protocol_valid": false}, {"status": "passed", "failures": [], "stage": "31200-round-0", "service_protocol_valid": true}, {"status": "passed", "failures": [], "stage": "31200-round-1", "service_protocol_valid": true}, {"status": "passed", "failures": [], "stage": "31200-round-2", "service_protocol_valid": true}, {"status": "passed", "failures": [], "stage": "long-8192", "service_protocol_valid": true}]; 200-update FU/F0 divergence: {"loss": {"first_different_step": 2, "max_relative_difference": 0.04826568132660418, "zero_reference_mismatch_steps": []}, "grad_norm": {"first_different_step": 2, "max_relative_difference": 0.6788698491935565, "zero_reference_mismatch_steps": []}}. FU parameter hashes are deliberately not compared with F0. These observations are not convergence evidence.

8192-token median training allocated peak GiB: {}. See per-round values and limiting phases above.

31200-token median training allocated peak GiB: {'F0': 32.87557649612427, 'F64': 32.87557649612427, 'FU': 32.88015079498291}. See per-round values and limiting phases above.

Service restoration exceptions (vLLM not active in immediate post-run receipt): []. The wrapper was changed during the batch to attempt restoration against the originally active vLLM state and check it after a delay. The compatibility and thermal guard states are recorded separately in per-run JSON.

## Commands

The original reference commands are in `SPARK_RETEST_PRS.md`. Use the training worktree as cwd, set `$MODEL`, `$ONE_ROW_8192`, `$ONE_ROW_31200`, `$OUTPUT`, `$LOCK_A`, `$LOCK_B`, and ensure each output path is new. Before each run stop Duty, Ollama, and compatibility services; restore their previous state after exit and preserve the thermal guard. Collect clock/temperature/power telemetry at one-second intervals. The entrypoint acquires both locks.

```bash
python -m pytest fastvideo/tests/train/utils/test_negative_prompt_release.py fastvideo/tests/train/utils/test_config.py fastvideo/tests/train/utils/test_optimizer_foreach_chunk.py -q --confcutdir=fastvideo/tests/train
python -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo/tests/attention/test_fa2_determinism_policy.py -q --confcutdir=fastvideo/tests/attention
python fastvideo-kernel/experiments/gb10_z/code/chunk_v5_entry.py --tokens 8192 --steps 1 --chunk 0 --hash-every 1 --model "$MODEL" --data "$ONE_ROW_8192" --output "$OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
python fastvideo-kernel/experiments/gb10_z/results/spark-fused-control/fused_entry.py --tokens "$TOKENS" --steps 1 --arm FU --hash-every 1 --model "$MODEL" --data "$ONE_ROW_DATA" --output "$OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
python fastvideo-kernel/experiments/gb10_z/results/spark-fused-control/fused_entry_v2.py --tokens "$TOKENS" --steps "$STEPS" --arm "$ARM" --hash-every 1 --model "$MODEL" --data "$ONE_ROW_DATA" --output "$OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

Feasibility uses the first fused entry at both lengths. Formal arm order is in summary.json: 8192/4 steps and 31200/2 steps for three rounds, then 8192/200 steps. Fresh process and output per run. Raw weights, videos, tensors, local paths, hostnames and video identifiers are not shipped. All failures retain a sanitized exception. Verify with `sha256sum -c SHA256SUMS`.
