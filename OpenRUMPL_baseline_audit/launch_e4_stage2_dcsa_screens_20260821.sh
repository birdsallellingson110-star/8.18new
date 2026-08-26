#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820
TOK=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e4_anticollapse
OUTROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e4_stage2
mkdir -p "${OUTROOT}"
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${AUDIT}" PYTHONUNBUFFERED=1

run_one() {
  local name=$1 tokenizer=$2 lr=$3
  local out="${OUTROOT}/${name}"; mkdir -p "${out}"
  "${PY}" -u "${AUDIT}/train_e4_discrete_continuous_k96_scorer_20260821.py" \
    --train-cache "${BASE}/train_c2_22c.npz" \
    --validation-cache "${BASE}/validation_c2_22c.npz" \
    --e2-checkpoint "${ROOT}/e2_c2_unbiased_scorer/seed0/model_best.pth.tar" \
    --proposal-checkpoint "${ROOT}/e2_limb_utility/seed0/model_best.pth.tar" \
    --k96-checkpoint "${ROOT}/e2_pose_dsac_limb_proposal/seed0_30e_tmux/model_best.pth.tar" \
    --tokenizer-checkpoint "${TOK}/${tokenizer}/tokenizer_best.pth.tar" \
    --output-dir "${out}" --holdout-subject 8 --holdout-stride 5 \
    --max-train-samples 12000 --epochs 3 --batch-size 8 \
    --learning-rate "${lr}" --d-model 64 --heads 4 --depth 1 --dropout 0 \
    --relative-score-limit 2.0 --gate-mm 0.15 --workers 0 --seed 0 --gpu 0 \
    2>&1 | tee "${out}/train.log"
}

run_one fsq4375_dcsa_seed0 fsq34_4375_seed0 2e-4 & p0=$!
run_one fsq1000_dcsa_seed0 fsq34_1000_seed0 2e-4 & p1=$!
run_one simvq1024_dcsa_seed0 simvq34_cb1024_seed0 2e-4 & p2=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"
