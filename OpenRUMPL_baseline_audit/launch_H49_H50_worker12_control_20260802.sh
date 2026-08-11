#!/usr/bin/env bash
# H49/H50: strict workers=12 comparison of old versus A1D-matched H21 inputs.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {old_h21|matched_h21} PHYSICAL_GPU [SEED]" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
seed=${3:-0}
if [[ "${seed}" != 0 && "${seed}" != 1 && "${seed}" != 2 ]]; then
  echo "SEED must be 0, 1, or 2" >&2
  exit 2
fi
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=${ROOT}/H49_H50_worker12_control

case "${variant}" in
  old_h21)
    if [[ "${seed}" == 0 ]]; then code=H49
    elif [[ "${seed}" == 1 ]]; then code=H51
    else code=H53
    fi
    type=mmpose_hrnet_coco_a1d_h21_legswap
    tag=${code}_H35_oldH21_workers12_seed${seed}_20260802
    note="H35 old H21 input; workers=12 paired seed ${seed}"
    ;;
  matched_h21)
    if [[ "${seed}" == 0 ]]; then code=H50
    elif [[ "${seed}" == 1 ]]; then code=H52
    else code=H54
    fi
    type=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
    tag=${code}_A1DmatchedH21_workers12_seed${seed}_20260802
    note="H48 A1D-matched H21 input; workers=12 paired seed ${seed}"
    ;;
  *)
    echo "Unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

train_log=${BASE}/logs/${tag}_train.log
eval_root=${BASE}/eval/${tag}
done_file=${BASE}/completed/${tag}.done
lock_file=${BASE}/locks/${tag}.lock
data_dir=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_${type}
mkdir -p "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/completed" \
  "${BASE}/locks" "${eval_root}"
test -s "${CFG}"
test -s "${data_dir}/h36m_train.pkl"
test -s "${data_dir}/h36m_validation.pkl"

exec 9>"${lock_file}"
flock 9
if [[ -s "${done_file}" ]]; then
  echo "[${code}] skip completed ${tag}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1

# Disable every unrelated experimental branch.
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
    echo "[${code}] start tag=${tag} time=$(date --iso-8601=seconds)"
    echo "[${code}] paired_control workers=12 seed=${seed} type=${type}"
    echo "[${code}] note=${note}"
    sha256sum "${REPO}/lib/models/multiview_rumpl.py" "${CFG}" \
      "${data_dir}/h36m_train.pkl" "${data_dir}/h36m_validation.pkl"
  } | tee "${train_log}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 12 --seed "${seed}" \
    --train-mmpose-type "${type}" --test-mmpose-type "${type}" \
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
    --workers 12 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-views "${test_views[@]}" --test-on-all-cameras true \
    --test-mmpose-type "${type}" >"${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

echo "[${code}] end tag=${tag} time=$(date --iso-8601=seconds)" | tee -a "${train_log}"
date --iso-8601=seconds >"${done_file}"
