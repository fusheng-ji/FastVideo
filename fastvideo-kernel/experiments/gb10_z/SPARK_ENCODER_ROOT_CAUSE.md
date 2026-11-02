# Spark task: why does the negative-prompt text encoder survive `del`?

You are the agent on the DGX Spark (GB10, SM121). The probe is written and CPU-tested; your job is to run it exactly as below and
report. Do not write another probe and do not edit production code. Locks, services and data handling follow the Rules in
`SPARK_RETEST_PRS.md`. Everything you measure is a new replication.

## Question

`pr/release-neg-prompt-encoder` adds `gc.collect()` and `torch.cuda.empty_cache()` after `del text_encoder, ...` in
`encode_negative_prompt`. Earlier batches showed 194 encoder tensors (21.163 GiB allocated) alive after encoding without it, none with it.
Why they stayed alive is unknown:

| H | Explanation | Signature in the probe output |
|---|---|---|
| H1 | Reference cycle, freed only by the cyclic collector | alive at `after_return`, dead at `after_gc_collect` |
| H2 | A live reference outside the function (global, cache, attribute, hook) | still alive at `after_gc_collect`; `referrers.after_gc_collect` names the holder |
| H2' | A reference inside the function frame (for example the loader) | alive at `after_del_inside_function`, dead at `after_return` |
| H3 | Allocator caching, not object lifetime | tensors dead at `after_del_inside_function`; only `reserved_bytes` stays high |

`memory_allocated` counts live tensors only and `torch.cuda.empty_cache()` cannot change it. The probe records both counters at every
checkpoint, so it also answers which part of the PR's helper does the work.

## What the probe does

`fastvideo-kernel/experiments/gb10_z/code/encoder_lifetime_probe.py` runs the unmodified formal entrypoint `run_training_from_config`
with the 8192-token calibration recipe (`configs/calibration-8192.yaml`, one training step, trackers `none`), with the same distributed
initialisation, seeding and deterministic settings as `chunk_v5_entry.py`. So the DiT, the VAE and the encoder load in the same order as in
the formal run. It replaces three names only:

- `negative_prompt.TextEncoderLoader`: subclass that takes weak references to every encoder parameter and buffer after load;
- `negative_prompt._release_encoder_memory`: records the state right after `del`, then runs the original helper only in mode `fixed`;
- `wan.encode_negative_prompt`: after the real function returns, records the state, runs `gc.collect()`, records, runs
  `torch.cuda.empty_cache()`, records, and stops the run with `ProbeDone` before the first training step.

Checkpoints, in order: `after_load`, `after_forward`, `after_del_inside_function`, (`fixed` only: `after_release_helper_inside_function`),
`after_return`, `after_gc_collect`, `after_empty_cache`. Each has `live_tensors` of `tracked_tensors`, `encoder_module_live`,
`allocated_bytes`, `reserved_bytes`, `gc_enabled`. When tensors are alive at `after_return` or `after_gc_collect`, `referrers.<checkpoint>`
holds the tree of objects referring to the encoder module (three levels: types, dict keys, owner classes, frame function and line).

| mode | release helper | purpose |
|---|---|---|
| `fixed` | original PR helper runs | confirms the PR fix in the same harness |
| `natural` | no-op | reproduces the code before the PR |
| `gc-disabled` | no-op; cyclic collector off from encoder load until the explicit collection | rules out a background collection |
| `cycle` | no-op; explicit collection uses `DEBUG_SAVEALL` | `cycle` lists what the collector freed and what in that garbage refers to the encoder module |

## Step 0: preconditions

1. Fetch and check out `validate/encoder-root-cause` at the commit given in the instruction, working tree clean. Merge it into your
   results branch `spark/gb10-encoder-root-cause` (only new files and this document differ); keep the failed first attempt as it is.
2. Do not set `CUDA_VISIBLE_DEVICES=''`. Run the CPU checks; all must pass:

```bash
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_encoder_lifetime_probe.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo/tests/train/utils/test_negative_prompt_release.py -q --confcutdir=fastvideo/tests/train
```

3. `MODEL` is the Wan2.1-T2V-1.3B Diffusers directory and `ONE_ROW_8192` the one-row 8192-token parquet used by the earlier batches
   (`chunk_v5_entry.py --data`). `LOCK_A` and `LOCK_B` are the two GPU lock paths. `OUT` is a new directory for this attempt.

## Step 1: run six trials, in this order, each a fresh process

Stop competing services before each trial and restore them after, as in the earlier batches.

```bash
P=fastvideo-kernel/experiments/gb10_z/code/encoder_lifetime_probe.py
n=0
for trial in fixed natural natural natural gc-disabled cycle; do
  n=$((n + 1))
  python "$P" --mode "$trial" --model "$MODEL" --data "$ONE_ROW_8192" \
    --output "$OUT/$n-$trial" --lock "$LOCK_A" --lock "$LOCK_B" || { echo "trial $n-$trial failed"; break; }
done
```

Each trial writes `result.json` (sanitized; commit it), and `run.yaml`, `rendezvous` and on failure `traceback.local.txt` (these contain
local paths; do not commit them). A trial loads the models and encodes once; no training step runs. If a trial fails, stop the loop, keep
its `result.json`, record the error type and sanitized message, and report. Do not retry and do not change the probe.

## Step 2: read the result (stated before the run; do not tune toward it)

Use the `natural` trials for the hypothesis. All three must agree on the checkpoint at which the tensors die; otherwise report the
variation and stop.

- `fixed`: `after_release_helper_inside_function` and `after_return` show 0 live tensors. If not, the PR fix does not work here;
  report that first.
- `natural` alive at `after_return` and dead at `after_gc_collect`: **H1**. Quote `referrers.after_return` for the objects that close the
  cycle, and `cycle.garbage_referrers_of_module` from the `cycle` trial.
- `natural` alive at `after_del_inside_function` and dead at `after_return`: **H2'**; returning from the function is enough.
- `natural` alive at `after_gc_collect`: **H2**. Quote `referrers.after_gc_collect`.
- `natural` dead already at `after_del_inside_function`, while earlier batches saw 194 live tensors: **H3** or a changed code path;
  state the discrepancy.
- `gc-disabled` agrees with `natural`; if it differs, report both.
- Memory: for one `natural` trial report `allocated_bytes` and `reserved_bytes` at `after_return`, `after_gc_collect` and
  `after_empty_cache`, the change of each between checkpoints in GiB, and `tracked_bytes`.

## Report

Under `fastvideo-kernel/experiments/gb10_z/results/spark-encoder-root-cause/attempt-2/`: each trial's `result.json` in a directory named
like its output directory (`1-fixed`, `2-natural`, ...), and a `README.md` with versions, the commit, the exact commands, a table of
`live_tensors`, `allocated_bytes` and `reserved_bytes` at every checkpoint for all six trials, and failures as failures. No weights, paths,
hostnames or video IDs. Commit to `spark/gb10-encoder-root-cause` and push. Answer:

1. Which of H1, H2, H2', H3 holds, with the referrer evidence.
2. At which checkpoint does `allocated` drop by the encoder size, and does `empty_cache()` change `allocated` or only `reserved`?
3. Does the PR's helper (`fixed`) release the encoder inside the function?
