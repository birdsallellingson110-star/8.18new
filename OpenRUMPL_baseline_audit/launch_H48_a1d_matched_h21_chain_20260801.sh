#!/usr/bin/env bash
# H48: retrain H21 on A1D seed coordinates, export matched inputs, retrain RUMPL.
set -euo pipefail

physical_gpu=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=${ROOT}/H46_H48_root_cause
H21_OUT=${BASE}/H48_H21_a1d_matched_v2focus_reg005
A1D=${ROOT}/A1D_dense_residual_balanced/final.pth
A1D_TYPE=mmpose_hrnet_coco_a1d_legswap
TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
TYPE_DIR=${DATA}/datasets_mmpose/annot_filtered_5_64_${TYPE}
tag=H48_A1DmatchedH21_RUMPL_triAnchor_seed0_20260801
train_log=${BASE}/logs/${tag}_train.log
eval_root=${BASE}/eval/${tag}
done_file=${BASE}/completed/${tag}.done
lock_file=${BASE}/locks/${tag}.lock

mkdir -p "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/completed" \
  "${BASE}/locks" "${eval_root}" "${H21_OUT}" "${TYPE_DIR}"
exec 9>"${lock_file}"
flock 9
if [[ -s "${done_file}" ]]; then
  echo "[H48] skip completed ${tag}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

if [[ ! -s "${H21_OUT}/final.pth" ]]; then
  {
    echo "[H48] train H21 on A1D coordinates $(date --iso-8601=seconds)"
    sha256sum "${AUDIT}/train_iterative_pose_query_refiner.py" "${A1D}"
  } | tee "${H21_OUT}/launcher.log"
  "${PY}" -u "${AUDIT}/train_iterative_pose_query_refiner.py" \
    --input-pkl "${DATA}/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --seed-mmpose-pkl \
      "${DATA}/datasets_mmpose/annot_filtered_5_64_${A1D_TYPE}/h36m_train.pkl" \
    --dense-shards "${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz \
    --steps 5000 --view-probabilities 3 1 1 \
    --learning-rate 0.0003 --weight-decay 0.0001 \
    --hard-case-pixels 4 --maximum-hard-weight 5 --delta-penalty 0.05 \
    --irls-iterations 5 --seed 0 --device cuda:0 --output-dir "${H21_OUT}" \
    >"${H21_OUT}/train.log" 2>&1
fi
test -s "${H21_OUT}/final.pth"

export_split() {
  local split=$1
  local out=$2
  local base_mmpose=$3
  shift 3
  local shards=("$@")
  if [[ -s "${out}" ]]; then
    echo "[H48] skip existing ${split} input"
    return
  fi
  echo "[H48] export ${split} $(date --iso-8601=seconds)"
  "${PY}" -u "${AUDIT}/export_h21_refined_mmpose_pkl.py" \
    --input-pkl "${DATA}/datasets/annot_filtered_5_64/h36m_${split}.pkl" \
    --base-mmpose-pkl "${base_mmpose}" \
    --dense-shards "${shards[@]}" \
    --h21-checkpoint "${H21_OUT}/final.pth" \
    --mode a1d_h21 --a1d-checkpoint "${A1D}" --a1d-depth-samples 64 \
    --device cuda:0 --output "${out}" >"${TYPE_DIR}/export_${split}.log" 2>&1
}

export_split validation "${TYPE_DIR}/h36m_validation.pkl" \
  "${DATA}/datasets_mmpose/annot_filtered_5_64_${A1D_TYPE}/h36m_validation.pkl" \
  "${ROOT}/A0_h36m_val_heatmap_topk8"/shard{0..3}.npz
export_split train "${TYPE_DIR}/h36m_train.pkl" \
  "${DATA}/datasets_mmpose/annot_filtered_5_64_${A1D_TYPE}/h36m_train.pkl" \
  "${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz

export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_RELATIVE_VIEW_FUSION=0 VFT_FULL_RANDOM_MASK=0
export RUMPL_ANCHOR_CENTERED_RAYS=0 RUMPL_INPUT_PLUCKER=0 RUMPL_INPUT_HARMONIC_L=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export RUMPL_GBT_SET_DECODER=0 GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_TOKEN_DROPOUT=0 CAA_LAMBDA=0 DEPRO_LAMBDA=0
export REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0

cd "${REPO}"
checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
  -path "*${tag}*/model_best.pth.tar" -print | sort | tail -n 1)
if [[ -z "${checkpoint}" ]]; then
  {
    echo "[H48] train RUMPL tag=${tag} $(date --iso-8601=seconds)"
    echo "[H48] control=H35 only_variable=H21 trained on matched A1D seed distribution"
    sha256sum "${REPO}/lib/models/multiview_rumpl.py" \
      "${AUDIT}/train_iterative_pose_query_refiner.py" \
      "${AUDIT}/export_h21_refined_mmpose_pkl.py" "${H21_OUT}/final.pth"
  } | tee "${train_log}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 6 --seed 0 \
    --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
    --validate-on-two-datasets 1 --use-mmpose-val 0 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
    >>"${train_log}" 2>&1
  checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
    -path "*${tag}*/model_best.pth.tar" -print | sort | tail -n 1)
fi
test -n "${checkpoint}"
printf '%s\n' "${checkpoint}" >"${BASE}/checkpoints/${tag}.txt"

for n_views in 2 3 4; do
  eval_dir=${eval_root}/V${n_views}
  mkdir -p "${eval_dir}"
  if [[ "${n_views}" -eq 2 ]]; then test_views=(1 2)
  elif [[ "${n_views}" -eq 3 ]]; then test_views=(1 2 3)
  else test_views=(1 2 3 4)
  fi
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers 6 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-views "${test_views[@]}" --test-on-all-cameras true \
    --test-mmpose-type "${TYPE}" >"${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

echo "[H48] end tag=${tag} $(date --iso-8601=seconds)" | tee -a "${train_log}"
date --iso-8601=seconds >"${done_file}"
