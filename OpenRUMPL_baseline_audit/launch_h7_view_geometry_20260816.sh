#!/usr/bin/env bash
# H7a: geometry/view candidate verifier on the current 22-candidate cache.
# The two jobs use identical architecture and data with independent seeds.
# They wait for H6 only so that the two GPUs are never oversubscribed.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816
H6=${ROOT}/h6_cardinality_curriculum_ab
STAGE=${ROOT}/e2_h3_k2heavy_input_protocol_v1
OUT=${ROOT}/h7_view_geometry_ab

mkdir -p "${OUT}"
wait_started=$(date +%s)
while [[ ! -s "${H6}/CURRICULUM/H6_CURRICULUM_B2_H76_20E_LR5E5_T1_seed0_20260816.done" || \
        ! -s "${H6}/FIXED_MIXED/H6_FIXED_MIXED_B2_H76_20E_LR5E5_T1_seed0_20260816.done" ]]; do
  if [[ -s "${H6}/FAILED" ]]; then
    echo "[H7] H6 failed; refusing to start H7" >&2
    exit 1
  fi
  now=$(date +%s)
  if (( now - wait_started > 43200 )); then
    echo "[H7] timed out waiting for H6" >&2
    exit 1
  fi
  echo "[H7] waiting for H6 completion $(date --iso-8601=seconds)" >>"${OUT}/orchestrator.log"
  sleep 30
done

test -s "${STAGE}/train_h3_22c.npz" && test -s "${STAGE}/validation_h3_22c.npz"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run_one() {
  local seed="$1" gpu="$2"
  local dir="${OUT}/seed${seed}"
  mkdir -p "${dir}"
  if [[ -s "${dir}/result.json" ]]; then
    echo "[H7] seed ${seed} already complete" >>"${OUT}/orchestrator.log"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export H7_JOINT_ATTENTION=none
    "${PY}" -u "${AUDIT}/train_h7_view_geometry_20260816.py" \
      --train-shards "${STAGE}/train_h3_22c.npz" \
      --validation-cache "${STAGE}/validation_h3_22c.npz" \
      --output-dir "${dir}" --attention-depth 2 \
      --pretrain-epochs 10 --finetune-epochs 5 --batch-size 256 \
      --temperature 1.8 --target-temperature-mm 5.0 --oracle-weight 1.0 \
      --identity-hinge 0 --workers 0 --seed "${seed}" --gpu 0 \
      >"${dir}/train.log" 2>&1
  ) &
}

run_one 0 0
run_one 1 1
wait
date --iso-8601=seconds >"${OUT}/COMPLETED"
echo "[H7] complete $(date --iso-8601=seconds)" >>"${OUT}/orchestrator.log"
