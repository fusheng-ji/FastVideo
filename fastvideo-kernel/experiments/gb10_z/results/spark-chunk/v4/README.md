# GB10 formal-entrypoint chunk validation v4

Base: `45f25e5c071a3d8b339077a91bcc3ab9b3861a59`; results branch: `spark/gb10-chunk-validation-v4`.
No production code or kernel defaults changed. These are new measurements, not inherited acceptance.
Environment: GB10 SM121, Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12.

## A: partial attribution, failed peak snapshot

The history-enabled 8192/chunk=0 run failed during the first optimizer update in the diagnostic accounting assertion: snapshot active allocation categories did not sum to `memory_allocated`. The snapshot and differing totals were not persisted before that assertion. No retry was made. Exact peak attribution is unresolved; no completed update is claimed.

The historical formal/private peak gap is 23,479,666,688 bytes (21.867 GiB). After negative encoding, 194 encoder parameter/buffer tensor weak references still resolve to GPU storage totaling 22,723,641,344 bytes (21.163 GiB), although the encoder module object itself is gone. This accounts for most of the gap; the remaining 756,025,344 bytes (0.704 GiB) are unassigned. A reference cycle or collection timing is a hypothesis, not an established root cause.

Negative conditioning cache is only 4,194,500 bytes; VAE storage is 293,186,636 bytes and is not wholly new relative to the private recipe. Arithmetic subtraction of encoder storage from the historical formal peak gives 29,537,948,160 bytes (27.509 GiB). This is a conditional estimate, not a measured release or promised peak: no module was released, and the limiting phase could move.

`A/memory.json` contains measured phases through clip. The requested full peak category breakdown, optimizer/zero-grad phases and peak snapshot are unavailable for A. See `A-analysis.json` for exact bytes and failure scope.

## B: 31200, three randomized pairs, two updates each

All three pairs passed exact loss, gradient norm, initial state and all 825 parameter hash comparisons after both updates. There was no OOM. Each configuration reproduced the same peak across all three rounds.

| Chunk | Allocated peak GiB | Reserved peak GiB | Limiting phase | Encoder tensors remaining at end |
|---|---:|---:|---|---:|
| 0 | 50.657315 | 51.523438 | first optimizer update | 0 |
| 64 | 54.042542 | 58.863281 | second backward | 194 |

Chunk=64 increased the whole-run allocated peak by 3,634,860,032 bytes (3.385227 GiB). It did not save peak memory in this batch. Encoder storage disappears during chunk=0's first optimizer update but survives both chunk=64 updates. This lifetime difference confounds attribution to optimizer temporaries alone. No explicit collection/release was introduced. Phase peak counters are cumulative, not reset independently at each phase.

`B-summary.json`, `B-state.json`, per-run `phases.csv`, `memory.json`, `observation.json` and frequency telemetry preserve the measurements. Update timing includes diagnostic work and is not a clean throughput benchmark. Original telemetry timestamps were replaced with sample indices. Schedule seed is 20261030; order is 64/0, 0/64, 0/64. Data is the same single parquet row repeated, with equality and source fingerprints in `B-input-receipt.json`.

## C and D

200-step consistency was not run because A failed, as permitted by the task. It is not certified. The optional chunk-size scan was not run; no best chunk size can be established. Prior v3 8192 four-step results remain historical evidence, not a new v4 replication.

## Reproduction

Supply local paths through the variables below; no weights, tensors, videos, video identifiers, raw snapshots or machine paths are included. Use the matching Wan2.1-1.3B weights and row-zero inputs from the prior recipe. Text encoder source: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision `0fad780a534b6463e45facd96134c9f345acfa5b`. Source hashes are in `source-hashes.json`.

Before EACH run, record competitive service state, stop Duty/Ollama and compatibility/restart services, confirm GPU availability, and arrange restoration on normal or exceptional exit. Keep the thermal guard unchanged. Record GPU temperature, power, utilization, memory and graphics clock at one-second intervals. The entrypoint acquires both existing GPU locks. Per-run exit receipts recorded restored active hold/compatibility/guard services; the hold unit was absent at a later inspection, so its subsequent lifecycle is unknown.

```bash
CODE=fastvideo-kernel/experiments/gb10_z/code
# OUTPUT must not already exist. A is the recorded failing diagnostic, not a repaired run.
python "$CODE/chunk_v4_entry.py" --history --tokens 8192 --steps 4 --chunk 0 --model "$MODEL" --data "$ONE_ROW_8192" --output "$NEW_A" --lock "$LOCK_A" --lock "$LOCK_B"
# Six fresh directories, using the order in B-schedule.json:
python "$CODE/chunk_v4_entry.py" --tokens 31200 --steps 2 --chunk "$CHUNK" --model "$MODEL" --data "$ONE_ROW_31200" --output "$NEW_B" --lock "$LOCK_A" --lock "$LOCK_B"
python "$CODE/chunk_v4_compare.py" --steps 2 "$DENSE/observation.json" "$CHUNK64/observation.json"
```

Generated YAML sets `training.tracker.trackers: [none]`, requested chunk size, seed 1000 and strict deterministic mode. Four new comparator CPU tests passed; changed-file pre-commit passed with repository exclusions respected. Production diff against the base is empty. No production repair, automatic retry, PR or new certification was performed.

Run `sha256sum -c SHA256SUMS` in this directory to verify the sanitized package. These hashes describe exported files, not omitted raw artifacts.
