#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820
OUTROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e3_dransac
mkdir -p "${OUTROOT}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${AUDIT}"
export PYTHONUNBUFFERED=1

run_one() {
  local name=$1 lr=$2 tau_end=$3 kl=$4 entropy=$5 seed=$6
  local out="${OUTROOT}/${name}"
  mkdir -p "${out}"
  "${PY}" -u "${AUDIT}/train_e3_differentiable_ransac_proposal_20260821.py" \
    --train-cache "${BASE}/train_c2_22c.npz" \
    --validation-cache "${BASE}/validation_c2_22c.npz" \
    --e2-checkpoint "${ROOT}/e2_c2_unbiased_scorer/seed0/model_best.pth.tar" \
    --proposal-checkpoint "${ROOT}/e2_limb_utility/seed0/model_best.pth.tar" \
    --k96-checkpoint "${ROOT}/e2_pose_dsac_limb_proposal/seed0_30e_tmux/model_best.pth.tar" \
    --output-dir "${out}" --epochs 3 --batch-size 24 \
    --learning-rate "${lr}" --hypotheses 96 --tau-start 0.8 --tau-end "${tau_end}" \
    --proposal-kl-weight "${kl}" --entropy-weight "${entropy}" \
    --holdout-subject 8 --holdout-stride 5 --max-train-samples 12000 \
    --gate-mm 0.15 --workers 0 --seed "${seed}" --gpu 0 \
    2>&1 | tee "${out}/train.log"
}

run_one official_tau_seed0 2e-5 0.3 0.02 0.01 0 &
p0=$!
run_one conservative_anchor_seed0 1e-5 0.5 0.10 0.00 0 &
p1=$!
wait "${p0}"
wait "${p1}"
