# Tracker-disabled smoke: missing text encoder

This continuation retains all v2 and original v3 records. It uses unchanged
v3 tool sources from `738f352f54bf8ffff735d4cf808af1719eb4460f`, descended from
`49ed5c8f91a5207852c180d8fc37485156a40718`. Existing prerequisite and CPU checks
were reused, not rerun. No production code or kernel default was changed.

Exactly one fresh smoke process was launched: 8192 tokens, seed 1000, chunk 0,
one requested update, strict determinism, and YAML
`training.tracker.trackers: [none]`. Tracker initialization succeeded without
W&B credentials. The production Trainer reached `method.on_train_start`, then
Wan's `ensure_negative_conditioning` attempted to load the text encoder and
failed because its local weight files were absent. No training update ran;
loss and parameter hashes were not produced. The observer callback was not
reached. This is an input-completeness failure, not an optimizer mismatch.

There was no retry, input download, or bypass of negative conditioning. Prior
private-loop results are not evidence that this formal-entrypoint smoke passed.
Before a separately authorized future attempt, provide a complete matching
local model including text-encoder weights. This batch does not certify that
supplying them will resolve any subsequent issue.

8192 memory reproduction, 31200 phase attribution, 200-step equality, and best
chunk size remain unmeasured. All four downstream experiments are explicitly
not run. The historical 26.8 to 23.6 GiB result was not reproduced here.

## Execution and reproduction

Runtime versions are in smoke-01.json. The active GPU inference service and its
compatibility service were temporarily stopped together with the manual GPU
hold, then restored on failure. Duty and Ollama were inactive. The thermal guard
and clock policy were unchanged. Both scheduler GPU locks were acquired before
runtime imports. Smoke frequency telemetry is included with timestamps replaced
by sequential sample numbers; it is not training performance evidence.

Use the project Python environment and explicit local inputs. This command
reproduces the recorded failure if the text encoder remains absent. Always use
a fresh output directory; no automated retry is provided.

```bash
python fastvideo-kernel/experiments/gb10_z/code/chunk_entry_preflight.py \
  --model "$MODEL" --data "$TRAIN_DIRECTORY" --output "$NEW_OUTPUT" \
  --lock "$LOCK_A" --lock "$LOCK_B"
```

The YAML is generated from the calibration recipe with one requested step,
chunk 0, tracker none, and the experiment-only SmokeObserver callback. Raw logs,
resolved machine paths, and model/data inputs stay local. Only sanitized scalar
results and telemetry are published. SHA256SUMS covers these published files.
No PR was opened.
