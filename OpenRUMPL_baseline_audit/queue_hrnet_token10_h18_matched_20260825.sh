#!/usr/bin/env bash
# Build matched temporal inputs for the token-dropout generator + seed1 E2,
# then compare continuous-time H18 adaptation ranges on GPU1.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=${HRNET_DOWNSTREAM_ROOT:-/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_token10_generalization_20260825}
E2=${ROOT}/canonical_e2
H18=${ROOT}/canonical_h18
SOURCE=${HRNET_H18_INIT:-/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/best_available_modules_20260825_safe_candidates/hrnet/canonical_h18/model_batch8_accum8_eval32/model_best.pth.tar}

export CUDA_VISIBLE_DEVICES="${HRNET_H18_VISIBLE_GPU:-1}"
export PYTHONPATH=${AUDIT}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

mkdir -p "${H18}/cache" "${H18}/scores" \
  "${H18}/fused/train" "${H18}/fused/validation"

while [[ ! -s "${H18}/cache/validation_11c.npz" || \
         ! -s "${H18}/scores/train_e2_scores.npy" ]]; do
  sleep 20
done

if [[ ! -s "${H18}/cache/validation_22c.npz" ]]; then
  "${PY}" -u "${AUDIT}/append_confidence_candidates_current_input_20260815.py" \
    --input "${H18}/cache/validation_11c.npz" \
    --output "${H18}/cache/validation_22c.npz" \
    >"${H18}/cache/append_validation.log" 2>&1
fi

if [[ ! -s "${H18}/scores/validation_e2_scores.npy" ]]; then
  "${PY}" -u "${AUDIT}/build_e2_base_scores_20260818.py" \
    --cache "${H18}/cache/validation_22c.npz" \
    --checkpoint "${E2}/identity_hinge/seed1/model_best.pth.tar" \
    --output "${H18}/scores/validation_e2_scores.npy" \
    --batch-size 256 --gpu 0 >"${H18}/scores/validation.log" 2>&1
fi

build_fused() {
  local split=$1 cache scores
  if [[ "${split}" == train ]]; then
    cache=${E2}/cache/train_22c.npz
  else
    cache=${H18}/cache/validation_22c.npz
  fi
  scores=${H18}/scores/${split}_e2_scores.npy
  [[ -s "${H18}/fused/${split}/manifest.json" ]] && return 0
  "${PY}" -u "${AUDIT}/build_e2_fused_temporal_cache_20260818.py" \
    --cache "${cache}" --scores "${scores}" \
    --output-dir "${H18}/fused/${split}" --temperature-v2 0.4 \
    --temperature-v3 1.8 --temperature-v4 1.8 --chunk-size 256 --gpu 0 \
    >"${H18}/fused/${split}.log" 2>&1
}

build_fused train & fused_train=$!
build_fused validation & fused_validation=$!
wait "${fused_train}" "${fused_validation}"

train_h18() {
  local name=$1 scale_min=$2 scale_max=$3
  local out=${H18}/${name}
  [[ -s "${out}/result.json" ]] && return 0
  mkdir -p "${out}"
  "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${E2}/cache/train_22c.npz" \
    --train-fused "${H18}/fused/train/fused_poses.npy" \
    --train-pkl /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl \
    --validation-cache "${H18}/cache/validation_22c.npz" \
    --validation-fused "${H18}/fused/validation/fused_poses.npy" \
    --validation-pkl /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl \
    --output-dir "${out}" --init-checkpoint "${SOURCE}" \
    --window-length 9 --frame-stride 5 --epochs 6 --batch-size 8 \
    --grad-accum-steps 8 --eval-batch-size 32 --hidden-dim 96 --layers 2 \
    --lr 2e-5 --weight-decay 5e-4 --residual-scale-m 0.10 \
    --workers 0 --gpu 0 --seed 25 --camera-independent \
    --continuous-time --source-fps 50 --reference-dt-s 0.1 \
    --max-time-period-s 2.0 --time-scale-min "${scale_min}" \
    --time-scale-max "${scale_max}" >"${out}/train.log" 2>&1
}

train_h18 model_continuous_nowarp 1.0 1.0 & nowarp=$!
if [[ "${HRNET_H18_NOWARP_ONLY:-0}" == "1" ]]; then
  wait "${nowarp}"
else
  train_h18 model_continuous_conservative 0.5 2.0 & conservative=$!
  train_h18 model_continuous_strong 0.3333333333 3.0 & strong=$!
  wait "${nowarp}" "${conservative}" "${strong}"
fi

date --iso-8601=seconds >"${H18}/COMPLETED"
echo "[token10 matched H18] complete ${H18}"
