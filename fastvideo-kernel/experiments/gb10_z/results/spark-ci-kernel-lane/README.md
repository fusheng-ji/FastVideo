# GB10 CI kernel lane reproduction: image build interrupted

Baseline: `b133f96d487d934b2c86ec6f2630db5dc578d03f`. This attempt did not reach the kernel lanes and cannot establish whether CI failures reproduce. No PR branch, production code, reproduction script or summarizer was modified.

Host: NVIDIA GB10, driver 580.173.02, Docker client/server 29.6.2, Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3. CPU self-check: first run had 2 passed and 1 skipped because the pinned main commit was absent; after fetching upstream main, all 3 passed. The Docker GPU preflight succeeded. The supplied dry run completed and its commands were inspected.

The base image was pulled successfully (digest in `base-image.json`), but building the Spark CI image stayed at the pinned FlashAttention wheel installation with no new output for over 20 minutes. An independent, read-only HTTP range request from a container using the unmodified base image timed out in TLS handshake, although container DNS and TCP connections to GitHub and its release-assets host succeeded and a host HTTP HEAD returned 200. This is supporting connectivity evidence; it does not prove the precise reason the installer stalled.

The stalled build was deliberately terminated to restore the competing services. Docker reported `CANCELED` / `failed to solve: Canceled: context canceled` and the attempt exited 130. This is an interrupted build, not an independently reported wheel-install failure. No automatic retry or host-side fallback was performed. Both competing services that were active before the attempt were active again afterward, and the GPU thermal guard remained active. No `ci-image.json`, lane JUnit reports, lane summaries, or per-ref results were generated.

Commands (local paths represented by placeholders):

```bash
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_summary.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
docker run --rm --gpus all ubuntu:22.04 true
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-dry" --lock "$LOCK_A" --lock "$LOCK_B" --dry-run
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

The GPU preflight and reproduction were executed with competing services stopped and restored through the existing service wrapper. The preflight held both GPU locks; the supplied reproduction script acquires them for each container lane.

1. **main and PR #1948:** not run; whether they pass is unknown.
2. **PR #1934 and PR #1938:** not run; failing tests and assertion messages are unknown.
3. **Base image digest:** `sha256:caf466e02a916e4f29238a69399674a60dbe3f63c430e66cb3b46e75e1489df5`. This identifies the pulled base only; no completed CI image or negative reproduction result exists.
