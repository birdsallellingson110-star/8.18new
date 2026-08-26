#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820
OUT=${ROOT}/failure_informed_map/observation_conditioned_k96_screen_seed0

mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${AUDIT}
export PYTHONUNBUFFERED=1

"${PY}" -u "${AUDIT}/train_observation_conditioned_k96_scorer_20260820.py" \
  --train-cache "${BASE}/train_c2_22c.npz" \
  --validation-cache "${BASE}/validation_c2_22c.npz" \
  --e2-checkpoint "${ROOT}/e2_c2_unbiased_scorer/seed0/model_best.pth.tar" \
  --proposal-checkpoint "${ROOT}/e2_limb_utility/seed0/model_best.pth.tar" \
  --k96-checkpoint "${ROOT}/e2_pose_dsac_limb_proposal/seed0_30e_tmux/model_best.pth.tar" \
  --output-dir "${OUT}" --holdout-subject 8 --holdout-stride 5 \
  --epochs 3 --batch-size 16 --learning-rate 2e-4 \
  --d-model 64 --heads 4 --depth 1 --relative-score-limit 2.0 \
  --score-temperature 0.5 --gate-mm 0.15 --max-train-samples 12000 \
  --workers 0 --seed 0 --gpu 0 \
  2>&1 | tee "${OUT}/train.log"
