#!/usr/bin/env bash
# Reproducible launcher for the selected H76/Vol/Algebraic quality gate.
set -euo pipefail

gpu=${GPU:-0}
seed=${SEED:-0}
epochs=${EPOCHS:-120}
tag=${TAG:-main}
expected_risk=${EXPECTED_RISK_WEIGHT:-0}
fused_weight=${FUSED_MPJPE_WEIGHT:-1}
soft_target=${SOFT_TARGET_WEIGHT:-0}

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
SCRIPT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_unified_three_branch_gate_20260813.py
ROOT=/mnt/data/cjyoutput/external_fair_comparison_20260813
EVAL=${ROOT}/lt_input_rumpl_ablation/epoch_eval/V234Start/epoch_20

args=(
  --train-h76-v2 "${EVAL}/V2_train200fpa/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl"
  --train-vol-v2 "${ROOT}/lt_vol_train200fpa_v2_predictions.npz"
  --train-geometry-v2 "${ROOT}/ray_geometry_acute_epoch20_train200fpa_v2.npz"
  --train-candidate-residual-v2 "${ROOT}/three_candidate_ray_residual_train200fpa_v2.npz"
  --train-h76-v3 "${EVAL}/V3_train200fpa/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl"
  --train-vol-v3 "${ROOT}/lt_vol_train200fpa_v3_predictions.npz"
  --train-geometry-v3 "${ROOT}/ray_geometry_acute_epoch20_train200fpa_v3.npz"
  --train-candidate-residual-v3 "${ROOT}/three_candidate_ray_residual_train200fpa_v3.npz"
  --train-h76-v4 "${EVAL}/V4_train200fpa/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl"
  --train-vol-v4 "${ROOT}/lt_vol_train200fpa_v4_predictions.npz"
  --train-geometry-v4 "${ROOT}/ray_geometry_acute_epoch20_train200fpa_v4.npz"
  --train-candidate-residual-v4 "${ROOT}/three_candidate_ray_residual_train200fpa_v4.npz"
  --train-alg "${ROOT}/lt_official_alg_train200fpa_predictions.npz"
  --test-h76-v2 "${EVAL}/V2/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl"
  --test-vol-v2 "${ROOT}/lt_vol_controlled_full2021_v2_predictions.npz"
  --test-geometry-v2 "${ROOT}/ray_geometry_acute_epoch20_test_v2.npz"
  --test-candidate-residual-v2 "${ROOT}/three_candidate_ray_residual_test_v2.npz"
  --test-h76-v3 "${EVAL}/V3/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl"
  --test-vol-v3 "${ROOT}/lt_vol_uncertainty_full2021_v3_predictions.npz"
  --test-geometry-v3 "${ROOT}/ray_geometry_acute_epoch20_test_v3.npz"
  --test-candidate-residual-v3 "${ROOT}/three_candidate_ray_residual_test_v3.npz"
  --test-h76-v4 "${EVAL}/V4/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl"
  --test-vol-v4 "${ROOT}/lt_vol_controlled_full2021_v4_predictions.npz"
  --test-geometry-v4 "${ROOT}/ray_geometry_acute_epoch20_test_v4.npz"
  --test-candidate-residual-v4 "${ROOT}/three_candidate_ray_residual_test_v4.npz"
  --test-alg "${ROOT}/lt_official_alg_full2021_predictions.npz"
  --epochs "${epochs}" --hidden 512 --hidden2 256 --seed "${seed}"
  --expected-risk-weight "${expected_risk}"
  --fused-mpjpe-weight "${fused_weight}"
  --soft-target-weight "${soft_target}"
  --device cuda:0
  --output "${ROOT}/unified_three_branch_${tag}_cv_seed${seed}.json"
  --weights-output "${ROOT}/unified_three_branch_${tag}_cv_seed${seed}.pth"
)

export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"${PY}" -u "${SCRIPT}" "${args[@]}" \
  > "${ROOT}/unified_three_branch_${tag}_cv_seed${seed}.log" 2>&1
