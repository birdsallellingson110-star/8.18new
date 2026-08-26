#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/lixiaob/cjy/rumpl_venv310/bin/python
TRAINER=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/best_available_modules_20260825_safe_candidates/hrnet
SOURCE=${ROOT}/canonical_h18/model_batch8_accum8_eval32/model_best.pth.tar
SOURCE=${H18_SOURCE_CHECKPOINT:-${SOURCE}}
OUTPUT=${H18_OUTPUT_DIR:-${ROOT}/canonical_h18/model_generalization_v2_continuous_timewarp}
GPU=${H18_GPU:-0}
BATCH_SIZE=${H18_BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${H18_GRAD_ACCUM_STEPS:-8}
EVAL_BATCH_SIZE=${H18_EVAL_BATCH_SIZE:-32}
EPOCHS=${H18_EPOCHS:-6}
LR=${H18_LR:-2e-5}
SEED=${H18_SEED:-25}
TIME_SCALE_MIN=${H18_TIME_SCALE_MIN:-0.3333333333}
TIME_SCALE_MAX=${H18_TIME_SCALE_MAX:-3.0}

exec "${PYTHON}" -u "${TRAINER}" \
  --train-cache "${ROOT}/canonical_e2/cache/train_22c.npz" \
  --train-fused "${ROOT}/canonical_h18/fused/train/fused_poses.npy" \
  --train-pkl /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl \
  --validation-cache "${ROOT}/canonical_h18/cache/validation_22c.npz" \
  --validation-fused "${ROOT}/canonical_h18/fused/validation/fused_poses.npy" \
  --validation-pkl /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl \
  --output-dir "${OUTPUT}" \
  --init-checkpoint "${SOURCE}" \
  --window-length 9 \
  --frame-stride 5 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --hidden-dim 96 \
  --layers 2 \
  --lr "${LR}" \
  --weight-decay 5e-4 \
  --residual-scale-m 0.10 \
  --workers 0 \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  --camera-independent \
  --continuous-time \
  --source-fps 50 \
  --reference-dt-s 0.1 \
  --max-time-period-s 2.0 \
  --time-scale-min "${TIME_SCALE_MIN}" \
  --time-scale-max "${TIME_SCALE_MAX}"
