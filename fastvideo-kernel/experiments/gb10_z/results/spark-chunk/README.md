# Chunked AdamW validation v2: preflight blocked

Base: `a36a7eb0a892746f8173c26eee86c9a3f8f6f7a3`. This is a new replication;
no historical hardware acceptance is inherited. Kernel and FA2 files are byte-identical
to `cf6327b7f4d8f1326717f5be9ef9da74fb067560`; saved forward-consistent dV is present.
Default dense/TMA behavior was not changed.

CPU checks passed: GB10 experiments 120, FA2 policy 18, optimizer chunking 7.
Runtime: Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12, GB10 SM121.

## Failure and scope

The first two tool revisions failed because the hashing helper expected a Path instead
of a string. Both failures are preserved. Revision 3 entered the production
`run_training_from_config`, loaded the model and dataset, then failed constructing
Trainer's tracker with `UsageError: No API key configured`. No training update ran.
These were distinct fresh directories after explicit tool corrections, not automatic retries.

The diagnostic attempted to disable tracking with an empty project name and tracker
list. Config parsing normalizes the empty project to `fastvideo`; the tracker factory
then selects W&B. The existing supported `training.tracker.trackers: [none]` setting
is a possible next configuration, but **was not run in this batch**. Authentication is
not intrinsically required by the optimizer. No production workaround was installed.

8192 paired measurement, comparison with the historical dense parameter hashes,
31200 phase attribution, 200-step consistency, and the optional size scan are all
**not run**. There is no new memory or timing result and no best chunk-size claim.
The historical 26.8 → 23.6 GiB result came from a private loop and is not certified
for the production Trainer by this batch.

The two GPU locks were held for each attempted runtime. The active compatibility
service and manual training hold were temporarily stopped and restored on exit.
Duty and Ollama were already inactive; the temperature guard was kept active.
No formal timing was collected, and no GPU frequency comparison is claimed.
No PR was opened. Raw logs, configurations and model/data inputs remain local.

## Reproduction

Use the project Python environment; placeholders denote explicit local inputs.
Each invocation requires a previously nonexistent output directory. The command
below reproduces the recorded failure, not a successful benchmark.

```bash
python -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo/tests/attention/test_fa2_determinism_policy.py fastvideo/tests/train/utils/test_optimizer_foreach_chunk.py -q --confcutdir=fastvideo/tests
python fastvideo-kernel/experiments/gb10_z/code/chunk_entry_preflight.py \
  --model "$MODEL" --data "$TRAIN_DIRECTORY" --output "$NEW_OUTPUT" \
  --lock "$LOCK_A" --lock "$LOCK_B"
```

The tool uses the shipped calibration YAML, seed 1000, 8192 tokens, chunk 0,
one requested step, strict PyTorch determinism and no checkpoint saving. Its only
training entry is the production YAML entrypoint; it does not swap optimizer code.
JSON null equality values mean unmeasured, not passed. SHA256SUMS hashes the
sanitized published evidence; it does not stand in for raw-log hashes.
