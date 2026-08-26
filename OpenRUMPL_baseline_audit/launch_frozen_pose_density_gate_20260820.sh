#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820
OUT=${ROOT}/failure_informed_map/frozen_pose_density_subject8.json
LOG=${ROOT}/failure_informed_map/frozen_pose_density_subject8.log

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${AUDIT}
export PYTHONUNBUFFERED=1

"${PY}" -u "${AUDIT}/diagnose_frozen_pose_density_20260820.py" \
  --train-cache "${BASE}/train_c2_22c.npz" \
  --validation-cache "${BASE}/validation_c2_22c.npz" \
  --e2-checkpoint "${ROOT}/e2_c2_unbiased_scorer/seed0/model_best.pth.tar" \
  --proposal-checkpoint "${ROOT}/e2_limb_utility/seed0/model_best.pth.tar" \
  --k96-checkpoint "${ROOT}/e2_pose_dsac_limb_proposal/seed0_30e_tmux/model_best.pth.tar" \
  --output "${OUT}" --holdout-subject 8 \
  --betas 0 0.02 0.05 0.1 0.2 0.4 0.8 \
  --features bone_direction root_relative --shrinkage 0.05 \
  --batch-size 128 --workers 0 --seed 0 --gpu 0 \
  2>&1 | tee "${LOG}"
