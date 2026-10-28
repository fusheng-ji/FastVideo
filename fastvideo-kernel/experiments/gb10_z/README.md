# GB10 Z experimental handoff

Spark agent: start with [SPARK_HANDOFF.md](SPARK_HANDOFF.md).

Z uses P64 packed event slots and bypasses spill-map lookup when there is no spill.
The kernel defaults remain dense and TMA. This is a research handoff, not a new public attention API.
The source kernel previously validated on GB10 has SHA256
`477cf40132176b4e430295ca830e3e35a9928698b68ba99ba5f52dbe7ae03407`.
See `SOURCE_MAP.json` for historical sources and adapted-file hashes.

The ordinary-mode calibration completed 9 runs, 180 updates and 864 validation forwards.
It does **not** certify statistical equivalence, long-term convergence or video quality.
Historical strict-mode acceptance and ordinary-mode observations must not be mixed.
The portable entrypoints have CPU checks only: this handoff did not rerun any GPU experiment.

## CPU checks

From the repository root, using the project's installed development environment:

```bash
python -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo/tests/attention/test_fa2_determinism_policy.py -q --confcutdir=fastvideo/tests/attention
```

The first command covers packed addressing/defaults, optimizer-window behavior, evaluation RNG isolation,
FA2 extraction/lifetime, and portable path/fingerprint rejection. It does not initialize CUDA.
FA2 GPU regression tests are included separately and are not selected by these commands.

## Prepare local inputs

Install the framework and kernel dependencies before preparing a manifest. The original environment used
Torch 2.12.0+cu130, Triton 3.7.0 and FA2 2.8.3+cu130torch2.12 on aarch64 GB10.
A different environment is a new replication and cannot inherit the historical acceptance stamp.

Inputs are intentionally not in Git: Wan2.1-T2V-1.3B model weights, cached training latents,
32 validation-video latents, full boundary captures and raw gradients/parameters.
Exact original input hashes and provenance are on the `pr_asset` branch.
The training parquet must retain its original schema and sampler ordering. Using newly preprocessed
videos instead of byte-matching original inputs is a new experiment, not an exact reproduction.

Define absolute paths for `MODEL`, `TRAIN_PARQUET`, `VALIDATION_PARQUET`, and `WORK` on the new host.
Use an empty output directory; do not reuse an old attempt. No host services are controlled by these tools.

```bash
TOOL=fastvideo-kernel/experiments/gb10_z/code
python "$TOOL/portable.py" prepare \
  --source-manifest "$WORK/source.json" --validation-manifest "$WORK/validation.json" \
  --model "$MODEL" --config fastvideo-kernel/experiments/gb10_z/configs/calibration-8192.yaml \
  --train-data "$TRAIN_PARQUET" --validation-data "$VALIDATION_PARQUET"
python "$TOOL/portable.py" check --source-manifest "$WORK/source.json"
```

Preparation streams hashes of inputs, weights and installed kernel sources, without GPU execution.
It creates a local validation ID manifest; do not publish that manifest or local paths.
The source manifest pins the runtime installation, not a claim that arbitrary files match the historical run.

## Reproduce one calibration run

Use two distinct host GPU lock paths, `LOCK_A` and `LOCK_B`, agreed with the receiving machine's scheduler.
Keep other GPU workloads and host thermal policy under that scheduler's control.
Both locks are held before importing the GPU kernel and released on exit. There is no automatic retry.

```bash
python "$TOOL/portable.py" run --source-manifest "$WORK/source.json" \
  --lock "$LOCK_A" --lock "$LOCK_B" --script control_train.py -- \
  --script "$TOOL/train_entry.py" --run-dir "$WORK/runs/41001-A" \
  --tokens 8192 --policy dense --artifact-root "$WORK" \
  --factor 1 --samples 20 --seed 41001 --repeat-id A
```

Run the frozen matrix in order: 41001 A/B/Z, 41002 B/Z/A, 41003 Z/A/B.
Use `--policy Z --repeat-id Z` for Z; A and B use dense. Every run is a separate process.
The recipe enforces 20 updates, G1, 8192 tokens, 30 FA2 backward calls with False and 825 parameter events.
Evaluation at updates 0/10/20 uses all 32 validation rows and restores training RNG.
Do not expand the seed set or training budget through this calibration entrypoint.

```bash
python "$TOOL/compare_calibration.py" --root "$WORK" --seed 41001
# Repeat comparison for 41002 and 41003, then:
python "$TOOL/report_calibration.py" --root "$WORK"
```

The separate `probe_model.py` captures/repeats ordinary-mode model boundaries without updates.
Use its `--help` for capture layers; pass `--snapshot` as the checkout and `--original-first-step`
as the original local pairing receipt. This receipt is not regenerated from a new run to claim an old match.

## Recover and replay an existing FA2 capture

```bash
python "$TOOL/portable.py" run --source-manifest "$WORK/source.json" \
  --lock "$LOCK_A" --lock "$LOCK_B" --script fa2_extract.py -- \
  --source "$FULL_CAPTURE" --output "$WORK/minimal-capture"
python "$TOOL/portable.py" run --source-manifest "$WORK/source.json" \
  --lock "$LOCK_A" --lock "$LOCK_B" --script fa2_replay.py -- \
  --input "$WORK/minimal-capture" --output "$WORK/native-replay"
```

Extraction and replay are separate processes. Replay preserves captured strides and runs ten False and ten True
native backwards. It checks unchanged inputs, finite outputs, deterministic repetition and weak-reference release.
Repeat separately for each length/format capture. OOM or any invariant failure stops that attempt.
No model matrix needs to be rerun when these original captures are available.
