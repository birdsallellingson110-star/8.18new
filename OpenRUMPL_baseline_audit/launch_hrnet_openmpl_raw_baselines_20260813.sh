#!/usr/bin/env bash
# Fair raw-coordinate baselines on the public OpenMPL H36M HRNet cache.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {rumpl|h76} PHYSICAL_GPU" >&2
  exit 2
fi

arm=$1
gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
OUT=/mnt/data/cjyoutput/hrnet_openmpl_coordinate_protocol_20260813
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE_TRAIN=mmpose_hrnet_coco_legswap
TYPE_TEST=mmpose_hrnet_coco_inferencer_legswap

case "${arm}" in
  rumpl)
    tag=CHR_R0_OpenMPLRawHRNet_RUMPL_w322_seed0_20260813
    note="original RUMPL; raw OpenMPL HRNet coordinates"
    tri_anchor=0
    centered=0
    plucker=0
    ;;
  h76)
    tag=CHR_H76_OpenMPLRawHRNet_TriAnchorCenteredPlucker_w322_seed0_20260813
    note="RUMPL plus tri-anchor, anchor-centered rays and Plucker line coordinates"
    tri_anchor=1
    centered=1
    plucker=1
    ;;
  *)
    echo "unknown arm: ${arm}" >&2
    exit 2
    ;;
esac

base=${OUT}/${arm}
log=${base}/logs/${tag}_train.log
eval_root=${base}/eval/${tag}
done_file=${base}/completed/${tag}.done
lock=${base}/locks/${tag}.lock
mkdir -p "${base}"/{logs,checkpoints,completed,locks} "${eval_root}"
exec 9>"${lock}"
flock 9
[[ -s "${done_file}" ]] && exit 0

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
export RUMPL_TRI_ANCHOR="${tri_anchor}"
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS="${centered}"
export RUMPL_INPUT_PLUCKER="${plucker}"
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0

# Strictly disable all unrelated experimental paths.
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export GBT_FUSION_GEOM=0 GBT_TOKEN_DROPOUT=0 RUMPL_GBT_SET_DECODER=0
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0
export BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
export RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0

cd "${REPO}"
{
  echo "start=$(date --iso-8601=seconds) arm=${arm} gpu=${gpu}"
  echo "input_train=${TYPE_TRAIN} input_test=${TYPE_TEST}"
  echo "only_variable=${note}"
  sha256sum "${CFG}" \
    /mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_${TYPE_TRAIN}/h36m_train.pkl \
    /mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_${TYPE_TEST}/h36m_validation.pkl
} >"${log}"

"${PY}" -u run/train_rumpl.py \
  --cfg "${CFG}" --gpus 0 --workers 12 --seed 0 \
  --train-mmpose-type "${TYPE_TRAIN}" --test-mmpose-type "${TYPE_TEST}" \
  --validate-on-two-datasets 1 --use-mmpose-val 0 \
  --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
  >>"${log}" 2>&1

checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
  -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
test -s "${checkpoint}"
printf '%s\n' "${checkpoint}" >"${base}/checkpoints/${tag}.txt"

for views in 2 3 4; do
  eval_dir=${eval_root}/V${views}
  mkdir -p "${eval_dir}"
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers 12 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-on-all-cameras true --n-views-combinations "${views}" \
    --model-num-views 4 \
    --test-mmpose-type "${TYPE_TEST}" >"${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -s "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds >"${done_file}"
