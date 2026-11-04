# GB10 CI kernel lane reproduction: all four lanes pass

The supplied driver and summarizer from `fc42584606bb522747989bdcd2b67ef7f1b46e1a` were merged onto `ad05347308408d29ae43bb876b549a7293c222e9`; the measured driver commit is `e4275ddb40c2097c8eac9b7b2c9751237bc16ff9`. No supplied script, summarizer, production code or PR branch was modified. Attempt 1 and attempt 2 are retained; attempt 3's checkpoint files are archived in `attempt-3/`, with only the requested stop-inspection line added to its README.

Before the new run, attempt 3's driver and exact container were already absent; no signal or broad process-pattern kill was used. Its exit/cleanup receipts were missing. Inspection confirmed no reproduction containers, both GPU locks available and existing services active.

## Environment and preconditions

Host: NVIDIA GB10 (SM121), NVIDIA driver 580.173.02, Linux 6.17.0-1031-nvidia aarch64, Python 3.12.13, Docker client/server 29.6.2. The prescribed CPU self-check completed with **22 passed**; Docker GPU preflight succeeded. CUDA visibility was unchanged. Check-only printed `check-only: PyPI check passed and wheel verified`, then the dry-run commands were inspected before stopping competing services.

The existing verified FlashAttention wheel was reused (**zero new download attempts**), as was CI image `sha256:f901f5e1f34ce88ed332b6843a5cb9276643b2accb69416d34c552e050d096b6`. No image rebuild or dependency-environment modification was made. No wheel or uv cache is included in this commit.

Every container reported the same non-kernel package versions: Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12, flashinfer-python 0.7.2.post1, nvidia-cusparselt-cu13 0.8.1; uv 0.13.6. All four fastvideo-kernel versions were 0.3.5. The four `environment.json` files report `checks_passed: true`, capability `[12, 1]`, no provenance problems, and no missing or differing Python files. The four `source.txt` files match the task's exact source SHAs and show both submodules at their pinned commits (leading space preserved).

## Results and answers

| Ref | Install | Kernel build | Environment checks_passed | Lane exit | Passed | Failed/errors | Skipped | Xfailed |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| main | pass | pass | true | 0 | 416 | 0 | 85 | 2 |
| pr1948 | pass | pass | true | 0 | 416 | 0 | 85 | 2 |
| pr1934 | pass | pass | true | 0 | 435 | 0 | 87 | 0 |
| pr1938 | pass | pass | true | 0 | 502 | 0 | 87 | 0 |

1. **main and PR #1948 both pass:** yes. Both have JUnit output, zero failures/errors and lane exit 0. Expected skips/xfails are listed separately.
2. **PR-only failures:** none on #1934; none on #1938. There are no failed test IDs, assertion messages or failure tracebacks to list. Both PR-only failure lists are empty. The Buildkite failure was **not reproduced** in this pinned GB10 environment; this does not establish why Buildkite failed or that its runner environment is identical.
3. **Base image digest:** `sha256:caf466e02a916e4f29238a69399674a60dbe3f63c430e66cb3b46e75e1489df5`. The exact image IDs, source SHAs, package versions and JUnit files are provided to compare with the CI runner.
4. **Setup/checks_passed:** every ref passed install, kernel build and environment checks, with `checks_passed: true`. There is no `failed_stage` or setup-failure log tail for this run. All `environment.err` files were empty and are omitted. Main's initial editable-install preparation took about 8m50s; it completed successfully, and subsequent containers reused the shared uv cache. No manual retry or change between refs occurred.

The supplied script and service wrapper exited 0. No reproduction containers remained and both locks were free afterward. Service states were restored to their actual pre-run states: compatibility service active, user vLLM unit already failed before the run and still failed afterward, training hold inactive, Ollama inactive; thermal guard active. Thus this report does not claim that the previously failed user vLLM unit was repaired or restarted.

## Commands and evidence handling

Exact commands, with local inputs and lock/output paths replaced by placeholders:

```bash
git fetch origin validate/ci-kernel-lane-repro
git merge --no-edit fc42584606bb522747989bdcd2b67ef7f1b46e1a
git fetch https://github.com/hao-ai-lab/FastVideo.git main
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_summary.py fastvideo-kernel/experiments/gb10_z/tests/test_container_network_probe.py fastvideo-kernel/experiments/gb10_z/tests/test_ci_kernel_lane_repro.py fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_container.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
docker run --rm --gpus all ubuntu:22.04 true
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-check" --check-only
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-dry" --lock "$LOCK_A" --lock "$LOCK_B" --dry-run
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

The existing service wrapper stopped competitors before Step 1 and restored their prior states after it. The unmodified driver held both locks for each labelled GPU container and applied the prescribed 90-minute lane timeout. No Docker or network setting was changed, and no host-side fallback was used.

`summary.md` and `summary.json` are copied unchanged from the supplied summarizer. Per-ref JUnit hostname attributes were removed; container path prefixes were replaced with `CHECKOUT`, `VENV` and `KERNEL_WHEEL_DIR` in JUnit/environment exports. Counts, test IDs, statuses, numerical values, source hashes and provenance checks were preserved. Empty environment errors, full logs, binaries, wheels, caches, worktrees and model/data artifacts are excluded. `SHA256SUMS` hashes the exported files, not the unredacted originals. The original local raw evidence remains available on this machine.
