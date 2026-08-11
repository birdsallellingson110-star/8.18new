#!/usr/bin/env bash
# Fast H21 screen: train on A1D seeds and export validation only.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {delta001|delta010|hard8|balanced|multiheavy|v2only|balanced10k} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
A1D=${ROOT}/A1D_dense_residual_balanced/final.pth
A1D_TYPE=mmpose_hrnet_coco_a1d_legswap
BASE=${ROOT}/H55_H58_h21_screen
steps=5000
resume=()

case "${variant}" in
  delta001)
    code=H55; delta=0.01; hard_px=4; hard_max=5; views=(3 1 1)
    suffix=delta001
    ;;
  delta010)
    code=H56; delta=0.10; hard_px=4; hard_max=5; views=(3 1 1)
    suffix=delta010
    ;;
  hard8)
    code=H57; delta=0.05; hard_px=3; hard_max=8; views=(3 1 1)
    suffix=hardpx3_max8
    ;;
  balanced)
    code=H58; delta=0.05; hard_px=4; hard_max=5; views=(1 1 1)
    suffix=balanced_views
    ;;
  multiheavy)
    code=H60; delta=0.05; hard_px=4; hard_max=5; views=(1 2 2)
    suffix=multiheavy_views
    ;;
  v2only)
    code=H61; delta=0.05; hard_px=4; hard_max=5; views=(1 0 0)
    suffix=v2only
    ;;
  balanced10k)
    code=H62; delta=0.05; hard_px=4; hard_max=5; views=(1 1 1)
    suffix=balanced10k
    steps=10000
    resume=(--resume "${ROOT}/H55_H58_h21_screen/H58_balanced_views_seed0/final.pth")
    ;;
  *) echo "Unsupported variant: ${variant}" >&2; exit 2 ;;
esac

out=${BASE}/${code}_${suffix}_seed0
type=mmpose_hrnet_coco_a1d_h21_${code,,}_${suffix}_legswap
type_dir=${DATA}/datasets_mmpose/annot_filtered_5_64_${type}
log=${out}/launcher.log
done_file=${out}/validation_export.done
mkdir -p "${out}" "${type_dir}"

if [[ -s "${done_file}" ]]; then
  echo "[${code}] skip completed"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

if [[ ! -s "${out}/final.pth" ]]; then
  {
    echo "[${code}] variant=${variant} delta=${delta} hard_px=${hard_px} hard_max=${hard_max} views=${views[*]}"
    echo "[${code}] start=$(date --iso-8601=seconds)"
    sha256sum "${AUDIT}/train_iterative_pose_query_refiner.py" "${A1D}"
  } | tee "${log}"
  "${PY}" -u "${AUDIT}/train_iterative_pose_query_refiner.py" \
    --input-pkl "${DATA}/datasets/annot_filtered_5_64/h36m_train.pkl" \
    --seed-mmpose-pkl \
      "${DATA}/datasets_mmpose/annot_filtered_5_64_${A1D_TYPE}/h36m_train.pkl" \
    --dense-shards "${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz \
    --steps "${steps}" --view-probabilities "${views[@]}" \
    --learning-rate 0.0003 --weight-decay 0.0001 \
    --hard-case-pixels "${hard_px}" --maximum-hard-weight "${hard_max}" \
    --delta-penalty "${delta}" --irls-iterations 5 \
    --seed 0 --device cuda:0 "${resume[@]}" --output-dir "${out}" \
    >"${out}/train.log" 2>&1
fi
test -s "${out}/final.pth"

validation=${type_dir}/h36m_validation.pkl
if [[ ! -s "${validation}" ]]; then
  "${PY}" -u "${AUDIT}/export_h21_refined_mmpose_pkl.py" \
    --input-pkl "${DATA}/datasets/annot_filtered_5_64/h36m_validation.pkl" \
    --base-mmpose-pkl \
      "${DATA}/datasets_mmpose/annot_filtered_5_64_${A1D_TYPE}/h36m_validation.pkl" \
    --dense-shards "${ROOT}/A0_h36m_val_heatmap_topk8"/shard{0..3}.npz \
    --h21-checkpoint "${out}/final.pth" \
    --mode a1d_h21 --a1d-checkpoint "${A1D}" --a1d-depth-samples 64 \
    --device cuda:0 --output "${validation}" \
    >"${type_dir}/export_validation.log" 2>&1
fi
test -s "${validation}"
printf '%s\n' "${validation}" >"${out}/validation_pkl.txt"
date --iso-8601=seconds >"${done_file}"
echo "[${code}] completed $(date --iso-8601=seconds)" | tee -a "${log}"
