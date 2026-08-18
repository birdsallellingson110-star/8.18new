#!/usr/bin/env bash
# Feature-level Epipolar Transformer input correction -> frozen H76.
# GPU0/GPU1 first export disjoint train descriptors, then run the feature and
# parameter-matched geometry-only control concurrently.
set -euo pipefail

PY_RUMPL=/home/lixiaob/cjy/rumpl_venv310/bin/python
PY_FEATURE=/mnt/data/cjydata/envs/raymixste/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
ROOT=${BASE}/Epipolar_Ray_Input_20260813
FEATURE_ROOT=${BASE}/RIGR_P2_feature_export_20260812
H76_STAGE=${BASE}/Counterfactual_View_Utility_20260811
TRAIN_CACHE=${H76_STAGE}/train_hypotheses
VAL_CACHE=${H76_STAGE}/H76_validation_all_subsets.npz
TRAIN_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_train.pkl
VAL_PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_validation.pkl
GROUP_IDS=${FEATURE_ROOT}/balanced20k_group_indices.npy
CFG=${BASE}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
H76=$(tr -d '\r\n' <${BASE}/H76_h50_centered_plucker/checkpoints/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803.txt)

mkdir -p "${ROOT}/descriptors" "${ROOT}/logs" \
  "${ROOT}/feature_seed0" "${ROOT}/geometry_control_seed0"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib:${AUDIT}

export_train_shard() {
  local physical_gpu=$1
  local shard=$2
  local output=${ROOT}/descriptors/train_shard${shard}of2.npy
  if [[ -s "${output}" && -s "${output}.group_indices.npy" ]]; then
    echo "train descriptor shard ${shard} exists; skip"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=${physical_gpu} "${PY_FEATURE}" -u \
    "${AUDIT}/export_epipolar_query_features_20260813.py" \
    --input-pkl "${TRAIN_PKL}" \
    --h76-cache \
      "${TRAIN_CACHE}/H76_train_all_subsets_shard0of2.npz" \
      "${TRAIN_CACHE}/H76_train_all_subsets_shard1of2.npz" \
    --feature-shards \
      "${FEATURE_ROOT}/balanced20k/shard0.npz" \
      "${FEATURE_ROOT}/balanced20k/shard1.npz" \
      "${FEATURE_ROOT}/balanced20k/shard2.npz" \
      "${FEATURE_ROOT}/balanced20k/shard3.npz" \
    --group-indices-file "${GROUP_IDS}" \
    --shard-index "${shard}" --num-shards 2 \
    --depth-samples 64 --ray-window-m 1.5 --device cuda:0 \
    --output "${output}" --log-every 100 \
    >"${ROOT}/logs/export_train_shard${shard}.log" 2>&1
}

export_validation() {
  local output=${ROOT}/descriptors/validation.npy
  if [[ -s "${output}" && -s "${output}.group_indices.npy" ]]; then
    echo "validation descriptors exist; skip"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=0 "${PY_FEATURE}" -u \
    "${AUDIT}/export_epipolar_query_features_20260813.py" \
    --input-pkl "${VAL_PKL}" --h76-cache "${VAL_CACHE}" \
    --feature-shards \
      "${FEATURE_ROOT}/val_shard0.npz" \
      "${FEATURE_ROOT}/val_shard1.npz" \
      "${FEATURE_ROOT}/val_shard2.npz" \
      "${FEATURE_ROOT}/val_shard3.npz" \
    --depth-samples 64 --ray-window-m 1.5 --device cuda:0 \
    --output "${output}" --log-every 100 \
    >"${ROOT}/logs/export_validation.log" 2>&1
}

echo "[$(date --iso-8601=seconds)] descriptor export start"
(export_train_shard 0 0; export_validation) &
pid0=$!
export_train_shard 1 1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
echo "[$(date --iso-8601=seconds)] descriptor export complete"

train_variant() {
  local physical_gpu=$1
  local variant=$2
  local output=$3
  CUDA_VISIBLE_DEVICES=${physical_gpu} "${PY_RUMPL}" -u \
    "${AUDIT}/train_epipolar_feature_ray_corrector_20260813.py" \
    --cfg "${CFG}" --h76-checkpoint "${H76}" \
    --train-cache \
      "${TRAIN_CACHE}/H76_train_all_subsets_shard0of2.npz" \
      "${TRAIN_CACHE}/H76_train_all_subsets_shard1of2.npz" \
    --validation-cache "${VAL_CACHE}" \
    --train-descriptors \
      "${ROOT}/descriptors/train_shard0of2.npy" \
      "${ROOT}/descriptors/train_shard1of2.npy" \
    --validation-descriptors "${ROOT}/descriptors/validation.npy" \
    --variant "${variant}" --output-dir "${output}" \
    --epochs 8 --batch-size 192 --workers 0 --holdout-modulo 10 \
    --lr 3e-4 --weight-decay 1e-4 --max-angle-degrees 0.5 \
    --angle-regularizer 1e-4 --seed 0 --device cuda:0 \
    >"${output}/train.log" 2>&1
}

echo "[$(date --iso-8601=seconds)] paired training start"
train_variant 0 feature "${ROOT}/feature_seed0" &
train0=$!
train_variant 1 geometry "${ROOT}/geometry_control_seed0" &
train1=$!
wait "${train0}"
wait "${train1}"
echo "[$(date --iso-8601=seconds)] paired training complete" | tee "${ROOT}/completed.txt"
