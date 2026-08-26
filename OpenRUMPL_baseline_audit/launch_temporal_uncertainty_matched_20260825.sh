#!/usr/bin/env bash
# Matched H18 optimization: label-free uncertainty/cardinality conditioning,
# with and without all-frame sequence supervision. Existing H18 remains intact.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824
export CUDA_VISIBLE_DEVICES="${TEMP_VISIBLE_GPU:-1}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run_one() {
  local train_cache=$1 train_fused=$2 train_pkl=$3
  local val_cache=$4 val_fused=$5 val_pkl=$6
  local train_uncertainty=$7 val_uncertainty=$8 init=$9 output=${10}
  local sequence_weight=${11} seed=${12}
  [[ -s "${output}/result.json" ]] && return 0
  mkdir -p "${output}"
  "${PY}" -u "${AUDIT}/train_e2_clean_temporal_residual_20260818.py" \
    --train-cache "${train_cache}" --train-fused "${train_fused}" \
    --train-pkl "${train_pkl}" --validation-cache "${val_cache}" \
    --validation-fused "${val_fused}" --validation-pkl "${val_pkl}" \
    --train-uncertainty "${train_uncertainty}" \
    --validation-uncertainty "${val_uncertainty}" \
    --output-dir "${output}" --init-checkpoint "${init}" \
    --window-length 9 --frame-stride 5 --epochs 6 --batch-size 8 \
    --grad-accum-steps 8 --eval-batch-size 32 --hidden-dim 96 \
    --layers 2 --lr 2e-5 --weight-decay 5e-4 --residual-scale-m 0.10 \
    --gpu 0 --seed "${seed}" --camera-independent --continuous-time \
    --source-fps 50 --reference-dt-s 0.1 --max-time-period-s 2.0 \
    --uncertainty-gate --stage-balanced-loss \
    --sequence-loss-weight "${sequence_weight}" >"${output}/train.log" 2>&1
}

run_frontend_pair() {
  local name=$1 base=$2 train_pkl=$3 val_pkl=$4 init=$5
  local train_cache="${base}/canonical_e2/cache/train_22c.npz"
  local val_cache="${base}/canonical_h18/cache/validation_22c.npz"
  local train_fused="${base}/canonical_h18/fused/train/fused_poses.npy"
  local val_fused="${base}/canonical_h18/fused/validation/fused_poses.npy"
  local train_u="${base}/canonical_h18/uncertainty/train.npy"
  local val_u="${base}/canonical_h18/uncertainty/validation.npy"
  test -s "${init}" -a -s "${train_u}" -a -s "${val_u}"
  run_one "${train_cache}" "${train_fused}" "${train_pkl}" \
    "${val_cache}" "${val_fused}" "${val_pkl}" "${train_u}" "${val_u}" \
    "${init}" "${base}/canonical_h18/model_uncertainty_stagebalanced" 0.0 31 & a=$!
  run_one "${train_cache}" "${train_fused}" "${train_pkl}" \
    "${val_cache}" "${val_fused}" "${val_pkl}" "${train_u}" "${val_u}" \
    "${init}" "${base}/canonical_h18/model_uncertainty_seq025" 0.25 32 & b=$!
  wait "${a}" "${b}"
  date --iso-8601=seconds >"${base}/canonical_h18/UNCERTAINTY_MATCHED_COMPLETED"
  echo "[temporal uncertainty] ${name} complete"
}

run_hrnet() {
  local base="${ROOT}/hrnet_token10_generalization_20260825"
  run_frontend_pair hrnet "${base}" \
    /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl \
    /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl \
    "${base}/canonical_h18/model_continuous_nowarp/model_best.pth.tar"
}

run_resnet() {
  local base="${ROOT}/stage1_h36m_dual_frontend/resnet152"
  while [[ ! -s "${base}/canonical_h18/model/result.json" ]]; do sleep 30; done
  run_frontend_pair resnet152 "${base}" \
    /mnt/data/cjyoutput/gbt_aligned_resnet_20260817_gpu1/frontend/train/h36m_train_res152.pkl \
    /mnt/data/cjyoutput/gbt_aligned_resnet_20260822/frontend_temporal_v2_gtinput/validation/h36m_validation_res152_temporal.pkl \
    "${base}/canonical_h18/model/model_best.pth.tar"
}

run_hrnet & hr=$!
run_resnet & rn=$!
wait "${hr}" "${rn}"
echo "[temporal uncertainty] all matched frontends complete"
