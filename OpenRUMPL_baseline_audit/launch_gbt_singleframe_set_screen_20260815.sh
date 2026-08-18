#!/usr/bin/env bash
# GBT-aligned T=1 structural screen on the strict C-HRNet cache.
#
# Both arms replace RUMPL's VFT/PFT readout with the existing GBT set
# encoder/joint-query decoder implementation.  They retain H76's validated
# confidence-weighted triangulation anchor and anchor-centered Pluecker rays;
# the only difference is whether GBT confidence/ray-distance attention bias is
# enabled.  This is a 5-epoch screen, not the final 300k-iteration paper run.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
TYPE=gbt_yolox_x_score001_fallback_legswap
DATA_ROOT=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
TRAIN=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl
VAL=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl
TYPE_DIR=${DATA_ROOT}/data/datasets_mmpose/annot_filtered_5_64_${TYPE}
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/gbt_singleframe_screen
mkdir -p "${OUT}" "${TYPE_DIR}"
test -s "${CFG}" && test -s "${TRAIN}" && test -s "${VAL}"
for split in train validation; do
  target=${TYPE_DIR}/h36m_${split}.pkl
  source=${TRAIN}
  [[ "${split}" == validation ]] && source=${VAL}
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
      echo "refusing mismatched ${target}" >&2; exit 2;
    }
  else
    ln -s "${source}" "${target}"
  fi
done

run_one() {
  local variant=$1 gpu=$2 biased=$3
  local tag="GBTSET_${variant}_H76_T1_5E_seed0_20260815"
  local root=${OUT}/${variant}
  local log=${root}/${tag}.log
  local done=${root}/${tag}.done
  mkdir -p "${root}"
  [[ -s "${done}" ]] && { echo "${variant} already complete"; return; }
  (
    export CUDA_VISIBLE_DEVICES=${gpu}
    export PYTHONPATH=${AUDIT}
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
    export TRAIN_FIXED_NUM_VIEWS=2 TRAIN_FIXED_NUM_VIEWS_EPOCHS=0
    export RUMPL_VIEW_COUNT_WEIGHTS=1,0,0 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
    export RUMPL_END_EPOCH=5 RUMPL_FLIP_LOWER_BODY_KP_TEST=0
    export RUMPL_LOSS_TYPE=JointsMSELoss
    export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
    export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1 RUMPL_PFT_REPEAT_LAST=1
    unset RUMPL_INPUT_HARMONIC_L
    export RUMPL_GBT_SET_DECODER=1 RUMPL_GBT_SET_DEPTH=3 RUMPL_GBT_SET_DECODER_DEPTH=2
    export RUMPL_GBT_SET_PLUCKER=1 RUMPL_GBT_SET_HARMONIC_L=15 RUMPL_GBT_SET_NO_CONF_CONCAT=1
    export RUMPL_GBT_SET_BIASED=${biased} RUMPL_GBT_SET_TOKEN_DROPOUT=0
    export GBT_CONF_INIT=0.1 GBT_GEOM_INIT=1.0
    export GBT_LEARNABLE_BIAS=0 GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
    export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
    export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
    export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
    export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0 RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
    export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0 RUMPL_GBT_QUERY_RESIDUAL=0
    export RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
    export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0
    export MONO_W=0 MONO_GT_W=0 RUMPL_TRAIN_SCOPE=all
    {
      echo "tag=${tag} gpu=${gpu} set_biased=${biased} start=$(date --iso-8601=seconds)"
      echo "T=1; fixed random K=2; H76 anchor+centered Plucker; harmonic_L=15; 5E screen"
      echo "train=${TRAIN} val=${VAL} type=${TYPE}"
      sha256sum "${CFG}" "${TRAIN}" "${VAL}"
    } >"${log}"
    cd "${REPO}"
    "${PY}" -u run/train_rumpl.py \
      --cfg "${CFG}" --gpus 0 --workers 8 --seed 0 \
      --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
      --validate-on-two-datasets 0 --use-mmpose-val 1 \
      --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
      >>"${log}" 2>&1
    ckpt=$(find "${MODEL_ROOT}" -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" -print | sort | tail -1)
    test -s "${ckpt}"
    printf '%s\n' "${ckpt}" >"${root}/checkpoint.txt"
    for views in 2 3 4; do
      eval_dir=${root}/eval/V${views}; mkdir -p "${eval_dir}"
      RUMPL_EVAL_STRICT=1 "${PY}" -u run/eval_rumpl_checkpoint.py \
        --cfg "${CFG}" --checkpoint "${ckpt}" --output-dir "${eval_dir}" \
        --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
        --test-on-all-cameras true --n-views-combinations "${views}" \
        --model-num-views 4 --test-mmpose-type "${TYPE}" \
        >"${eval_dir}/eval.log" 2>&1
      prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
      test -s "${prediction}"
      "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
        --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
    done
    date --iso-8601=seconds >"${done}"
  ) &
}

run_one PLAIN 0 0
run_one BIASED 1 1
wait
echo "[GBTSET] complete $(date --iso-8601=seconds)"
