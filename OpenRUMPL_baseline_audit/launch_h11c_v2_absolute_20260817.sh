#!/usr/bin/env bash
# H11C: unified absolute candidate-error scoring on V2/V3/V4 tasks.
# This is the controlled follow-up to H11B: identical 22-candidate pool,
# objective, split, temperature and frozen HRNet/RUMPL geometry; only the
# six V2 tasks are added to both training and selection.  Two seeds run on the
# two cards so that a V2 gain is not accepted from a single random initialization.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h11c_v2_absolute_pairwise
H3=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260816/e2_h3_k2heavy_input_protocol_v1

mkdir -p "${ROOT}/seed0" "${ROOT}/seed1"
export PYTHONPATH="${AUDIT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

run_seed() {
  local seed="$1"
  local gpu="$2"
  local out="${ROOT}/seed${seed}"
  if [[ -s "${out}/complete.done" ]]; then
    echo "[H11C] seed${seed} already complete"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
    "${AUDIT}/train_h76_pairwise_absolute_score_20260814.py" \
    --train-shards "${H3}/train_h3_22c.npz" \
    --validation-cache "${H3}/validation_h3_22c.npz" \
    --output-dir "${out}" --include-v2 \
    --attention-depth 2 --pretrain-epochs 10 --finetune-epochs 5 \
    --temperature 1.8 --target-scale-mm 10.0 --ranking-weight 0.25 \
    --batch-size 512 --workers 4 --seed "${seed}" --gpu 0 \
    --loss-mode absolute_rank >"${out}/train.log" 2>&1
  date --iso-8601=seconds >"${out}/complete.done"
}

run_seed 0 0 & p0=$!
run_seed 1 1 & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
if (( status != 0 )); then
  echo '[H11C] at least one seed failed' >&2
  exit "${status}"
fi
echo "[H11C] complete $(date --iso-8601=seconds)"
