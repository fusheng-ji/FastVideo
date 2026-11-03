# GB10 CI kernel lane: run in progress

This is an interim checkpoint, not a completed reproduction or a final setup failure. Source update `11f370eb1a36d4d159abec23278d1b759a777a82` was merged onto results commit `c49633178519ebbc8a081df91388aa82973cba14`. Attempt 2's five files are preserved unchanged in `attempt-2/`, and `attempt-1.json` is retained. No supplied script, summarizer, production code or PR branch was edited.

Host: NVIDIA GB10, NVIDIA driver 580.173.02, Docker client/server 29.6.2. CPU checks: **14 passed**. Docker GPU preflight passed. CUDA visibility was not disabled.

The check-only preflight printed `check-only: PyPI check passed and wheel verified`. Services remained running throughout Step 0. The 242,333,202-byte FlashAttention wheel downloaded in **one attempt** and matched SHA256 `3274144f92481df3d7deefae087ed5449b06389507de1f97d9b8109e053c6ff3`. Step 1 reused that verified cache. The wheel is not in Git.

The dry-run commands were inspected, then Step 1 started in a fresh directory with competing services stopped and the existing cleanup trap responsible for restoring their original state on exit. The image built successfully, using the verified wheel via the script's local HTTP server. Metadata for the base, CI image, wheel and Step 1 PyPI probe is included verbatim. The script prepared all four detached worktrees and pinned header submodules; their `source.txt` files are included.

At this checkpoint, `main` is still in `kernel_build` setup: editable FastVideo installation and CUDA compilation completed, but `build.sh`'s package installation is downloading `nvidia-cusparselt-cu13==0.8.1` (about 211 MiB). After a long partial transfer, uv reported a transient failure fetching the package from files.pythonhosted.org and began a retry under the supplied `UV_HTTP_RETRIES=10`. `main/kernel_build-progress-tail.txt` preserves a sanitized last-50-line excerpt at that point. This is setup-network evidence, not a failed kernel test, and the setup process has not yet exited. No experiment was restarted or manually retried. Services are still stopped for the active Step 1 run; they have not yet been restored because the run has not exited.

Commands (local paths are placeholders):

```bash
git fetch https://github.com/hao-ai-lab/FastVideo.git main
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_summary.py fastvideo-kernel/experiments/gb10_z/tests/test_container_network_probe.py fastvideo-kernel/experiments/gb10_z/tests/test_ci_kernel_lane_repro.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
docker run --rm --gpus all ubuntu:22.04 true
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-check" --check-only
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-dry" --lock "$LOCK_A" --lock "$LOCK_B" --dry-run
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

No final summary, per-ref environment verification, JUnit report or lane exit code exists yet. Their absence must not be interpreted as a pass or a final setup failure.

## Current answers (pending completion)

1. **main and PR #1948 both pass?** Pending: neither lane has run.
2. **PR-only failures on #1934 and #1938?** Pending: neither lane has run.
3. **Base image digest:** `sha256:caf466e02a916e4f29238a69399674a60dbe3f63c430e66cb3b46e75e1489df5`. CI image ID: `sha256:f901f5e1f34ce88ed332b6843a5cb9276643b2accb69416d34c552e050d096b6`. These identify the environment, not a negative reproduction result.
4. **Wheel attempts and setup-network failures?** One wheel-download attempt. Main's kernel setup has encountered a transient dependency-transfer failure and is still retrying internally; no ref has a final setup exit status yet.
