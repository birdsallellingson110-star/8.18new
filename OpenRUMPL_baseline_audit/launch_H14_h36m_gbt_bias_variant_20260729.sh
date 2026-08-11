#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {baseline|plain|conf|geom|both} PHYSICAL_GPU" >&2
  exit 2
fi

mode=$1
physical_gpu=$2

REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=/mnt/data/cjyoutput/h36m_paper_repro_20260728/H12_real_h36m_train.yaml
BASE=/mnt/data/cjyoutput/h36m_gbt_bias_20260729
TYPE=mmpose_hrnet_coco_inferencer_legswap

case "${mode}" in
  baseline)
    singleframe_gbt=0
    conf_bias=0
    geom_bias=0
    geom_norm=0
    ;;
  plain)
    singleframe_gbt=1
    conf_bias=0
    geom_bias=0
    geom_norm=0
    ;;
  conf)
    singleframe_gbt=1
    conf_bias=1
    geom_bias=0
    geom_norm=0
    ;;
  geom)
    singleframe_gbt=1
    conf_bias=0
    geom_bias=1
    geom_norm=1
    ;;
  both)
    singleframe_gbt=1
    conf_bias=1
    geom_bias=1
    geom_norm=1
    ;;
  *)
    echo "Unsupported mode: ${mode}" >&2
    exit 2
    ;;
esac

tag="H14_${mode}_clean_realH36M_fixed2_seed0_20260729"
train_log="${BASE}/logs/${tag}_train.log"
eval_root="${BASE}/eval/${tag}"

mkdir -p "${BASE}/logs" "${BASE}/checkpoints" "${eval_root}"
test -s "${CFG}"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_SINGLEFRAME_GBT="${singleframe_gbt}"
export RUMPL_SF_GBT_ENCODER_DEPTH=3
export RUMPL_SF_GBT_DECODER_DEPTH=2
export RUMPL_SF_GBT_PFT_DEPTH=4
export RUMPL_SF_GBT_CONF_BIAS="${conf_bias}"
export RUMPL_SF_GBT_GEOM_BIAS="${geom_bias}"
export RUMPL_SF_GBT_GEOM_NORM="${geom_norm}"

# Disable every unrelated optional branch. This is a single-variable bias
# ablation on one shared single-frame GBT backbone.
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_GATED_JOINT_ADAPTER=0
export RUMPL_ALT_JOINT_VIEW=0
export RUMPL_TRI_ANCHOR=0
export RUMPL_KPA=0
export RUMPL_ADAFUSE_VW=0
export RUMPL_POSE_CODEBOOK=0
export RUMPL_MULTI_HYP=1
export RUMPL_CONF_FILM=0
export GBT_LEARNED_RELIABILITY=0
export GBT_ORACLE_RELIABILITY=0
export GBT_TOKEN_DROPOUT=0
export CAA_LAMBDA=0
export DEPRO_LAMBDA=0
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=0

cd "${REPO}"
echo "[H14] start mode=${mode} gpu=${physical_gpu} sf_gbt=${singleframe_gbt} conf=${conf_bias} geom=${geom_bias} norm=${geom_norm} missing=0 fixed_views=2 $(date --iso-8601=seconds)" | tee "${train_log}"
"${PY}" -u run/train_rumpl.py \
  --cfg "${CFG}" \
  --gpus 0 \
  --workers 16 \
  --validate-on-two-datasets 1 \
  --use-mmpose-val 0 \
  --apply-noise-missing 0 \
  --missing-level 0.0 \
  --exp-name "${tag}" \
  >>"${train_log}" 2>&1

checkpoint=$(find \
  /mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999 \
  -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
  -print | sort | tail -n 1)
if [[ -z "${checkpoint}" ]]; then
  echo "[H14] missing checkpoint for ${tag}" | tee -a "${train_log}" >&2
  exit 3
fi
printf '%s\n' "${checkpoint}" >"${BASE}/checkpoints/${tag}.txt"

for n_views in 2 3 4; do
  eval_dir="${eval_root}/V${n_views}"
  mkdir -p "${eval_dir}"
  if [[ "${n_views}" -eq 2 ]]; then
    test_views=(1 2)
  elif [[ "${n_views}" -eq 3 ]]; then
    test_views=(1 2 3)
  else
    test_views=(1 2 3 4)
  fi

  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${eval_dir}" \
    --workers 16 \
    --gpu 0 \
    --use-mmpose-val true \
    --flip-lower-body-kp-test true \
    --test-views "${test_views[@]}" \
    --test-on-all-cameras true \
    --test-mmpose-type "${TYPE}" \
    >"${eval_dir}/eval.log" 2>&1

  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py \
    --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" \
    >"${eval_dir}/table2.log" 2>&1
done

echo "[H14] end mode=${mode} $(date --iso-8601=seconds)" | tee -a "${train_log}"
