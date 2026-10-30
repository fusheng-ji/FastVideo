# Matching text encoder restored; formal-entrypoint smoke passed

The missing five text-encoder shards were recovered from existing local files.
Each SHA256 was checked against the Hugging Face LFS metadata at immutable revision
`0fad780a534b6463e45facd96134c9f345acfa5b` of
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. Configuration and weight index matched the
existing target files byte for byte. All destination shards were hashed again.
Only names, hashes and sizes are published in text-encoder-provenance.json;
weights and raw inputs remain local.

The fresh 8192-token, chunk-0 smoke used `training.tracker.trackers: [none]`
and completed one real update through the unchanged production Trainer.
Loss was 0.12295346707105637; all 825 parameter tensors changed and their
post-update hashes were saved. There was no automatic retry or production patch.
Original prerequisite/CPU checks were not rerun. This is a smoke result, not a
paired performance measurement. Its 52261589504-byte peak allocated includes
formal-path behavior and must not be presented as reproducing the old peak.

## Historical recipe differences

The historical private peak harness repeats parquet row 0 and explicitly disables
negative conditioning. The production entrypoint builds a seeded permutation of
64 training rows and initializes negative conditioning. The latter requires the
restored text encoder. Neither behavior was patched or bypassed. Therefore an
observed difference from the old private baseline is not evidence that chunked
AdamW itself changes arithmetic. A four-update chunk-0 bridge checks the requested
historical endpoint comparison; its outcome is recorded separately.

## Reproduction and controls

Use explicit local inputs and two scheduler locks. Each output must be new.
The first command is the successful one-update smoke. The second uses the added
private budget argument for the four-update historical comparison.

```bash
python fastvideo-kernel/experiments/gb10_z/code/chunk_entry_preflight.py \
  --model "$MODEL" --data "$TRAIN_DIRECTORY" --output "$SMOKE_OUTPUT" \
  --lock "$LOCK_A" --lock "$LOCK_B"
python fastvideo-kernel/experiments/gb10_z/code/chunk_entry_preflight.py --steps 4 \
  --model "$MODEL" --data "$TRAIN_DIRECTORY" --output "$BRIDGE_OUTPUT" \
  --lock "$LOCK_A" --lock "$LOCK_B"
```

The v3 observer is configured through YAML and records hashes after each update.
Timing includes diagnostic synchronization and must not be used for a speed claim.
Raw frequency telemetry is published with sample indices instead of timestamps.
Active competing compatibility service and the manual hold were paused and restored;
the vLLM service was already failed/inactive, Duty and Ollama inactive. Thermal guard
and frequency control policy remained unchanged. No PR was opened.

## Four-update historical comparison: failed

The formal chunk-0 run completed four updates. All 825 final parameter hashes
differed from the historical four-update private dense run. The comparison has
known input-stream confounding, so it does not identify an optimizer defect.
Per the requested mismatch stop rule, no paired chunk-64 timing, 31200 phase
measurement, 200-step comparison or chunk sweep followed. First-step losses also
differ: formal 0.12295346707105637 versus historical 0.0868588387966156.
A future comparison must first define a genuinely shared data stream and separate
formal-entrypoint initialization memory from steady-state training; the acceptance
thresholds have not been relaxed. Services were restored after both runs.
