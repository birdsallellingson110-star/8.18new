#!/usr/bin/env bash
# H11: official GHT-style hypothesis scoring controls on the audited GBT-HRNet
# coordinate cache.  H11A keeps the original 11 H76 candidates and replaces
# only the utility head with GHT's whole-pose ScoreNN.  H11B uses the existing
# 22-candidate H76+pairwise pool and the absolute candidate-error/expected-risk
# objective.  The two outputs are independent models and must not be spliced.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h11_ght_official_score_ab
H3=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/e2_h3_k2heavy_input_protocol_v1

mkdir -p "${ROOT}/H11A_pose_score" "${ROOT}/H11B_absolute_pairwise"
export PYTHONPATH="${AUDIT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

run_h11a() {
  local out="${ROOT}/H11A_pose_score"
  if [[ -s "${out}/complete.done" ]]; then
    echo '[H11] H11A already complete'
    return
  fi
  CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${AUDIT}/train_h76_hypothesis_utility_20260811.py" \
    --train-shards "${H3}/train_h3_11c.npz" \
    --validation-cache "${H3}/validation_h3_11c.npz" \
    --variant pose --output-dir "${out}" \
    --epochs 15 --batch-size 512 --lr 5e-4 --weight-decay 1e-4 \
    --temperature 1.8 --workers 4 --seed 0 --gpu 0 \
    >"${out}/train.log" 2>&1
  date --iso-8601=seconds >"${out}/complete.done"
}

run_h11b() {
  local out="${ROOT}/H11B_absolute_pairwise"
  if [[ -s "${out}/complete.done" ]]; then
    echo '[H11] H11B already complete'
    return
  fi
  CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${AUDIT}/train_h76_pairwise_absolute_score_20260814.py" \
    --train-shards "${H3}/train_h3_22c.npz" \
    --validation-cache "${H3}/validation_h3_22c.npz" \
    --output-dir "${out}" \
    --attention-depth 2 --pretrain-epochs 10 --finetune-epochs 5 \
    --temperature 1.8 --target-scale-mm 10.0 --ranking-weight 0.25 \
    --batch-size 512 --workers 4 --seed 0 --gpu 0 --loss-mode absolute_rank \
    >"${out}/train.log" 2>&1
  date --iso-8601=seconds >"${out}/complete.done"
}

run_h11a & p_a=$!
run_h11b & p_b=$!
status=0
wait "${p_a}" || status=1
wait "${p_b}" || status=1
if (( status != 0 )); then
  echo '[H11] at least one GHT scorer failed' >&2
  exit "${status}"
fi
echo "[H11] complete $(date --iso-8601=seconds)"
