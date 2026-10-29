#!/usr/bin/env bash
# Explicit inputs: PYTHON, WORK (fresh), LOCK_A, LOCK_B, MODEL, DATA_8192, DATA_31200.
# Host scheduler must release the manual hold and provide an idle GPU beforehand.
# No service manipulation, automatic retry or overwrite is performed here.
set -euo pipefail
: "${PYTHON:?}" "${WORK:?}" "${LOCK_A:?}" "${LOCK_B:?}" "${MODEL:?}" "${DATA_8192:?}" "${DATA_31200:?}"
test ! -e "$WORK"
mkdir -p "$WORK"
TOOL=fastvideo-kernel/experiments/gb10_z/code
"$PYTHON" -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
"$PYTHON" -m pytest fastvideo/tests/attention/test_fa2_determinism_policy.py -q --confcutdir=fastvideo/tests/attention
for N in 8192 16384 31200; do
  for POLICY in dense recompute Z; do
    MODE=save
    if [[ $POLICY == recompute ]]; then MODE=recompute; fi
    FASTVIDEO_ATTN_QAT_SM121_DV_STATS="$MODE" "$PYTHON" "$TOOL/spark_measure.py" \
      --kind kernel --policy "$POLICY" --tokens "$N" --heads 12 --warmup 2 --measure 5 \
      --output "$WORK/kernel-$N-$POLICY" --lock "$LOCK_A" --lock "$LOCK_B"
  done
done
for N in 8192 16384 31200; do
  DATA=$DATA_8192
  if [[ $N == 31200 ]]; then DATA=$DATA_31200; fi
  for POLICY in dense recompute Z; do
    MODE=save
    if [[ $POLICY == recompute ]]; then MODE=recompute; fi
    FASTVIDEO_ATTN_QAT_SM121_DV_STATS="$MODE" "$PYTHON" "$TOOL/spark_measure.py" \
      --kind model --policy "$POLICY" --tokens "$N" --warmup 1 --measure 2 \
      --model "$MODEL" --data "$DATA" --output "$WORK/model-$N-$POLICY" \
      --lock "$LOCK_A" --lock "$LOCK_B"
  done
done
for N in 8192 16384 31200; do
  FASTVIDEO_ATTN_QAT_SM121_DV_STATS=save "$PYTHON" "$TOOL/spark_measure.py" \
    --kind reference --policy dense --tokens "$N" --heads 1 --warmup 1 --measure 1 \
    --output "$WORK/reference-$N" --lock "$LOCK_A" --lock "$LOCK_B"
done
for SCOPE in one-tile all-tiles; do
  FASTVIDEO_ATTN_QAT_SM121_DV_STATS=save "$PYTHON" "$TOOL/spark_overflow.py" \
    --scope "$SCOPE" --output "$WORK/overflow-$SCOPE" --lock "$LOCK_A" --lock "$LOCK_B"
done
