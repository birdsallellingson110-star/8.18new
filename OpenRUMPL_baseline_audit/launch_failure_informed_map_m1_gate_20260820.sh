#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820
OUT=${ROOT}/failure_informed_map/m1_gate_seed0

mkdir -p "${OUT}"
test -s "${BASE}/train_c2_22c.npz"
test -s "${BASE}/validation_c2_22c.npz"
test -s "${ROOT}/e2_c2_unbiased_scorer/seed0/model_best.pth.tar"
test -s "${ROOT}/e2_limb_utility/seed0/model_best.pth.tar"
test -s "${ROOT}/e2_pose_dsac_limb_proposal/seed0_30e_tmux/model_best.pth.tar"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${AUDIT}
export PYTHONUNBUFFERED=1

"${PY}" -u "${AUDIT}/train_failure_informed_map_20260820.py" \
  --train-cache "${BASE}/train_c2_22c.npz" \
  --validation-cache "${BASE}/validation_c2_22c.npz" \
  --e2-checkpoint "${ROOT}/e2_c2_unbiased_scorer/seed0/model_best.pth.tar" \
  --proposal-checkpoint "${ROOT}/e2_limb_utility/seed0/model_best.pth.tar" \
  --k96-checkpoint "${ROOT}/e2_pose_dsac_limb_proposal/seed0_30e_tmux/model_best.pth.tar" \
  --output-dir "${OUT}" \
  --epochs 3 --batch-size 128 --learning-rate 2e-4 \
  --d-model 96 --attention-heads 4 --attention-layers 2 --dropout 0.0 \
  --prior-precision 30.0 --trust-bias -4.0 \
  --relative-loss-weight 0.25 --workers 0 --seed 0 --gpu 0 \
  2>&1 | tee "${OUT}/train.log"
