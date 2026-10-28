# GB10 Z code and redacted experiment handoff

Code commit: [`41ef03d7e7453dedde129439a08109149abbe515`](https://github.com/fusheng-ji/FastVideo/commit/41ef03d7e7453dedde129439a08109149abbe515)

Code branch: `perf/attn-qat-gb10-stats-overflow`. Artifact branch: `pr_asset`.
Start with the code branch's `fastvideo-kernel/experiments/gb10_z/README.md` for CPU checks and explicit-path reproduction commands.

## What the evidence establishes

- Z strict deterministic-mode model coordinator completed its 82 jobs with acceptance. Historical job names retain `P64`; `strict-Z/selection.json` identifies the selected Z policy.
- Old L64 model performance failed, with about 19.69% to 41.09% overhead. Its failure is retained under `historical-L64`.
- Ordinary-mode fixed-boundary checks and model repeatability observations are in `ordinary-diagnostics`. The old queue failure remains a failure; a later successful replay does not rewrite it.
- Corrected FA2 replay completed 80 native backwards. All gradient buffers were released and final allocated memory returned to zero. False-mode numerical variability and True-mode exact repetition are reported separately.
- Calibration completed 9 runs, 180 updates, and 864 validation forwards: seeds 41001/41002/41003, dense-A/dense-B/Z, 8192 tokens, G1, 20 updates. Pairing and coverage checks passed.
- **Ordinary-mode statistical equivalence is NOT certified.** The reused 32-video validation set is not an independent final quality test. No long-term convergence, video-quality or other-device claim is made.

## Results and provenance

- `calibration/result.json`: per-seed validation losses and final parameter/gradient differences.
- `calibration/validation-loss.png` and `paired-differences.png`: descriptive plots.
- `calibration/runs/`: redacted run audits, metrics, evaluation RNG receipts and summaries.
- `strict-Z/results/`: historical first-step, windows, 20-update and performance-pair results, including failed attempts where present.
- `fa2-lifecycle/`: native replay results, raw logs and memory curves.
- `publication-checks/`: CPU tests for the relocated handoff (111 passed), FA2 CPU policy tests (18 passed), and repository-config pre-commit results. Kernel/experiment files are intentionally excluded by that configuration. Mypy was checked in an identical-config mirror because the source worktree directory name is not a valid Python package name.
- `source-map.json`: original source identifiers and old/new hashes. The Z kernel is byte-identical to the validated source. Portable entrypoints were adapted and **not GPU-revalidated**.
- `provenance.json`: original/export hashes; numeric and boolean fields checked unchanged, except explicitly redacted process identity fields.
- `SHA256SUMS`: checksums of the published files, independent of the original evidence hashes.

The export replaces user/host paths, addresses and video IDs with placeholders or stable anonymous IDs.
The private ID mapping remains on the originating machine. This package contains no model weights,
video data, QKV captures, raw gradients or raw parameters.

## Receiving-machine checks

```bash
git clone --branch perf/attn-qat-gb10-stats-overflow https://github.com/fusheng-ji/FastVideo.git FastVideo-Z
cd FastVideo-Z
git checkout 41ef03d7e7453dedde129439a08109149abbe515
python -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo/tests/attention/test_fa2_determinism_policy.py -q --confcutdir=fastvideo/tests/attention
git fetch origin pr_asset
git worktree add --detach ../FastVideo-Z-assets origin/pr_asset
cd ../FastVideo-Z-assets/gb10-qat/z-handoff-2026-10-28
sha256sum -c SHA256SUMS
```

Install the project development dependencies before running pytest. CPU tests do not require transferring the training data.
For CPU plot regeneration, copy the `calibration` directory to a disposable output directory, then run:

```bash
python scripts/report_calibration.py --root /absolute/path/to/copied/calibration
```

The training/model-data preparation and full/minimal capture commands are documented in the code README.
`required-inputs.json` provides dataset/model revisions and hashes of the original local artifacts.
Exact parquet caches, model weights and capture tensors must be supplied separately. Public upstream data alone does not guarantee the same preprocessing, split or order; newly generated data must be labeled as a new replication.
No additional GPU work, statistical confirmation experiment, commit to upstream, or PR edit was performed for this handoff.
