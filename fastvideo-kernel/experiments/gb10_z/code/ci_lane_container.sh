#!/usr/bin/env bash
# One ref's kernel lane inside the CI image (see ci_kernel_lane_repro.sh): install the checkout, build its kernel and
# install it the way docker/Dockerfile does (build.sh --wheel-dir, which installs the wheel with --no-deps, so no other
# package changes), check that the installed packages come from the checkout, then run the lane script.
#
# Exit 10/11/12 with OUT/failed_stage = install/kernel_build/environment when a setup stage fails; otherwise exit 0 and
# record the lane's own exit status in OUT/lane_exit_code. The REPRO_* defaults are the container paths; tests and
# local runs point them elsewhere.
set -uo pipefail

SRC="${REPRO_SRC:-/src}"
OUT="${REPRO_OUT:-/out}"
VENV="${REPRO_VENV:-/opt/venv}"
KERNEL_WHEEL_DIR="${REPRO_KERNEL_WHEEL_DIR:-/tmp/fastvideo-kernel-wheel}"
LANE_SCRIPT="${REPRO_LANE_SCRIPT:-.buildkite/scripts/lanes/kernel_tests.sh}"
LANE_TIMEOUT="${REPRO_LANE_TIMEOUT:-90m}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${HOST_UID:-}" ]]; then
    # Files created as root in the mounted checkout, output and uv cache go back to the host user.
    trap 'chown -R "$HOST_UID:$HOST_GID" "$SRC" "$OUT" ${UV_CACHE_DIR:+"$UV_CACHE_DIR"} 2>/dev/null' EXIT
fi
fail() {
    echo "$1" >"$OUT/failed_stage"
    exit "$2"
}

source "$VENV/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
cd "$SRC" || fail checkout 13

uv pip install --no-deps -e . >"$OUT/install.log" 2>&1 || fail install 10
rm -rf "$KERNEL_WHEEL_DIR"
(cd fastvideo-kernel && ./build.sh --wheel-dir "$KERNEL_WHEEL_DIR") >"$OUT/kernel_build.log" 2>&1 || fail kernel_build 11
REPRO_SRC="$SRC" REPRO_KERNEL_WHEEL_DIR="$KERNEL_WHEEL_DIR" python -P "$HERE/ci_lane_environment.py" \
    >"$OUT/environment.json" 2>"$OUT/environment.err" || fail environment 12

timeout "$LANE_TIMEOUT" bash "$LANE_SCRIPT" >"$OUT/lane.log" 2>&1
echo $? >"$OUT/lane_exit_code"
exit 0
