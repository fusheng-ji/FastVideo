# Data aligned; chunk option lost during YAML parsing

A one-row parquet was derived from ordinal row 0 of the original cached dataset.
The Arrow round-trip was exactly equal. Production DataLoader cycles that single
row, matching the historical private loop's repeated sample without replacing
Trainer, model, optimizer, kernel, or dataloader code. Input file hashes are in
input-receipt.json; no sample IDs or tensors are published.

## Confirmed results

Two fresh four-update runs completed through the production YAML entrypoint,
requesting chunk 64 and chunk 0 in the seeded random order. The chunk-0 run's
final **825/825 parameter hashes exactly matched** the historical private dense
four-update baseline. Its losses and gradient norms also matched:

| Update | Loss | Gradient norm |
| --- | --- | --- |
| 1 | 0.0868588387966156 | 0.48990559577941895 |
| 2 | 0.08840402215719223 | 0.3220910429954529 |
| 3 | 0.1797984391450882 | 1.4921125173568726 |
| 4 | 0.2647973299026489 | 2.849712371826172 |

This resolves the prior input-stream mismatch. Both new runs also had equal
per-step parameter hashes, losses, gradient norms, and recorded RNG states.
However, **this is not a valid chunk-0/chunk-64 comparison**:

- YAML requested `training.optimizer.foreach_chunk_size: 64`.
- `load_run_config` produced `OptimizerConfig.foreach_chunk_size == 0`.
- The observed optimizer was plain `AdamW`, with effective chunk size 0.
- `fastvideo/train/utils/config.py` explicitly constructs OptimizerConfig but
  does not pass foreach_chunk_size, so its dataclass default wins.

config-resolution.json records a separate minimal configuration probe. It did
initialize CUDA during imports/config loading; it is not claimed as a CPU-only
self-test or a timing run. The original preflight suites were not repeated.

The queue was stopped after the first two runs; remaining two pairs were not
started. Raw execution status remains completed for those runs, while
comparison.json rejects acceptance with `chunk_optimizer_selection`. No claim
about chunk memory savings or speed is made. Telemetry includes loading and
hash-observation intervals and is not a certified paired timing distribution.
The formal initialization differs from the old private harness, so historical
26.8/23.6 GiB peaks are not claimed reproduced.

## Reproduction and next gate

Prepare a private directory containing only the original parquet's first row,
retaining its schema and tensor values. Use the existing complete model and
explicit scheduler locks. Run each command in a new process and output directory.

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

Recorded run source hashes precede a diagnostic-only safeguard added after the
failure: the current tool rejects a requested/resolved chunk mismatch before
model construction. It will therefore fail early on chunk 64 with this production
branch, rather than reproduce the misleading completed run. The comparison tool
also requires the correct optimizer classes and effective chunk values. Six new
CPU comparator tests passed; pre-commit respected repository exclusions.

No production fix was made, respecting the requested boundary. The needed next
change is to wire the YAML field into OptimizerConfig and test actual YAML parsing,
not just direct dataclass construction. That requires a separate production change
before valid paired GPU measurements can resume. No automatic retries occurred.

The compatibility service and manual hold were restored after each run. The
thermal guard and clock policy remained unchanged. vLLM, Duty, and Ollama were
not active when runs launched. The two GPU locks were held for each training run.
Historical failures and successful smoke evidence remain unchanged. No PR opened.
