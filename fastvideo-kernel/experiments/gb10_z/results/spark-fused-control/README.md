# GB10 fused AdamW control

Base commit `45ee2455bb17f49e10f8163207ec9efbc79501ed`. New GB10 SM121 replication; Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12, CUDA 13.0, driver 580.173.02. Step 0 CPU checks: 36, 130 and 18 passed; formal entrypoint baseline smoke completed one update. Each run used YAML `training.tracker.trackers: [none]`, seed 1000 and a single repeated parquet row. Each output has its own directory. Source and input fingerprints, actual optimizer class and `fused`/`foreach` flags are recorded in JSON.

FU one-update feasibility used the v1 observer: optimizer timing includes scheduler and the private observer. Formal runs use v2 public optimizer hooks: optimizer peak counters are reset at each optimizer step, with pre-reset peaks retained. Reported training peaks combine all segments after `on_train_start`; startup peaks (including text encoder load) and whole-run peaks are separate. Phase counters in memory.json are cumulative only between resets. Timings include instrumentation/synchronization and clocks were recorded, not fixed; do not infer a speedup.

| Run | Updates | Training peak allocated GiB | Training peak reserved GiB | Limiting phase | Optimizer step peak GiB | Median step s | Median optimizer s |
|---|---:|---:|---:|---|---:|---:|---:|
| feasibility-8192-FU | 1 | 22.22846 | 23.30664 | optimizer_end | 22.22846 | 12.8642 | 2.6356 |
| feasibility-31200-FU | 1 | 24.20404 | 24.75781 | optimizer_end | 24.20404 | 61.6475 | 2.3687 |
| pair-8192-r0-FU | 4 | 24.28196 | 26.71680 | backward_end | 22.22846 | 9.5978 | 0.1919 |
| pair-8192-r0-F0 | 4 | 27.51424 | 30.87305 | optimizer_end | 27.51424 | 9.9274 | 0.5624 |
| pair-8192-r0-F64 | 4 | 24.28059 | 26.97266 | backward_end | 22.70307 | 9.9664 | 0.5560 |
| pair-8192-r1-FU | 4 | 24.28196 | 26.71680 | backward_end | 22.22846 | 9.6085 | 0.1923 |
| pair-8192-r1-F0 | 4 | 27.51424 | 30.87305 | optimizer_end | 27.51424 | 9.9417 | 0.5623 |
| pair-8192-r1-F64 | 4 | 24.28059 | 26.97266 | backward_end | 22.70307 | 10.0497 | 0.5593 |
| pair-8192-r2-F64 | 4 | 24.28059 | 26.97266 | backward_end | 22.70307 | 9.9659 | 0.5569 |

Queue status: failed; failures: [{"name": "pair-8192-r2-F64", "arm": "F64", "exit_code": 1, "reason": "attempt failed; no retry"}]; F0/F64 bitwise comparisons: [{"status": "passed", "failures": [], "stage": "8192-round-0"}, {"status": "passed", "failures": [], "stage": "8192-round-1"}]; 200-update FU/F0 divergence: null. FU parameter hashes are deliberately not compared with F0. These observations are not convergence evidence.

8192-token median training allocated peak GiB: {'F64': 24.280588626861572}. See per-round values and limiting phases above.

31200-token median training allocated peak GiB: {}. See per-round values and limiting phases above.

Service restoration exceptions (vLLM not active in immediate post-run receipt): ['pair-8192-r0-F64', 'pair-8192-r1-FU', 'pair-8192-r1-F0', 'pair-8192-r1-F64', 'pair-8192-r2-F64']. The wrapper was changed during the batch to attempt restoration against the originally active vLLM state and check it after a delay. The compatibility and thermal guard states are recorded separately in per-run JSON.

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

## Service restoration and continuation

The initial coordinator was itself misidentified by the vLLM process guard, so vLLM refused to restart after several runs. The 8192 round-2 F64 training completed but its wrapper failed on restoration. A first continuation used a coordinator that avoids that guard; round-2 F0 training completed and vLLM became active, but the wrapper returned failure because optional `reset-failed` was called on an unloaded unit. Neither failed attempt was retried. Their four-step results compare equal numerically, but the third 8192 round is not accepted as a compliant paired run. Earlier service exceptions are recorded per run. A corrected wrapper and coordinator continue only previously unattempted configurations; see continuation-preflight.json.

## Continuation result

Continuation queue status: **completed_with_prior_failure**. Detailed runs are in `continuation/`; aggregated memory by length and arm, all comparisons and the 200-step FU/F0 loss and gradient-norm deviations are in `final-summary.json`. The two original wrapper failures remain failures and were not retried. 8192 three-round service-protocol acceptance remains blocked; the results for those runs are descriptive.

8192 token training peak allocated GiB per round: {'F0': [27.514241, 27.514241, 27.514241], 'F64': [24.280589, 24.280589, 24.280589], 'FU': [24.281959, 24.281959, 24.281959]}.

31200 token training peak allocated GiB per round: {'F0': [32.875576, 32.875576, 32.875576], 'F64': [32.875576, 32.875576, 32.875576], 'FU': [32.880151, 32.880151, 32.880151]}.

200-update FU versus F0: {"loss": {"first_different_step": 2, "max_relative_difference": 0.04826568132660418, "zero_reference_mismatch_steps": []}, "grad_norm": {"first_different_step": 2, "max_relative_difference": 0.6788698491935565, "zero_reference_mismatch_steps": []}}. F0/F64 comparisons: [{"status": "passed", "failures": [], "stage": "8192-round-2", "service_protocol_valid": false}, {"status": "passed", "failures": [], "stage": "31200-round-0", "service_protocol_valid": true}, {"status": "passed", "failures": [], "stage": "31200-round-1", "service_protocol_valid": true}, {"status": "passed", "failures": [], "stage": "31200-round-2", "service_protocol_valid": true}, {"status": "passed", "failures": [], "stage": "long-8192", "service_protocol_valid": true}]. These describe this single repeated sample and do not establish convergence.

Observed FU minus F64 8192 median allocated peak: 0.001370 GiB. The pre-stated 0.2 GiB threshold would place this in the within threshold category, but the 8192 service protocol failure prevents treating that reading as fully accepted.
