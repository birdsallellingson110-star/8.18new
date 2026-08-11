#!/usr/bin/env bash
# H0: original RUMPL tri-anchor residual trained on A1D-refined 2D PKLs.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

physical_gpu=$1
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H0_a1d_refined_rumpl_tri_anchor.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE=mmpose_hrnet_coco_a1d_legswap
BASE=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H0_a1d_refined_rumpl_tri_anchor

code=H22_CUR
description=fixedK2First8_thenWeighted3to1to1
fixed_views=2
fixed_epochs=8
view_count_weights=3,1,1
tag="H0_${code}_a1dRefined2D_originalRUMPL_triAnchor_${description}_seed0_20260731"
train_log="${BASE}/logs/${tag}_train.log"
eval_root="${BASE}/eval/${tag}"
done_file="${BASE}/completed/${tag}.done"
lock_file="${BASE}/locks/${tag}.lock"

mkdir -p \
  "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/completed" \
  "${BASE}/locks" "${eval_root}"
test -s "${CFG}"

exec 9>"${lock_file}"
flock 9
if [[ -s "${done_file}" ]]; then
  echo "[H0] skip completed ${tag}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS="${fixed_views}"
export TRAIN_FIXED_NUM_VIEWS_EPOCHS="${fixed_epochs}"
export RUMPL_VIEW_COUNT_WEIGHTS="${view_count_weights}"

export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05

export RUMPL_GBT_SET_DECODER=0
export RUMPL_SINGLEFRAME_GBT=0
export RUMPL_SF_GBT=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_ALT_JOINT_VIEW=0
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export RUMPL_RAY_DEPTH_AUX=0
export RUMPL_ADAFUSE_VW=0
export RUMPL_KPA=0
export GBT_TOKEN_DROPOUT=0
export CAA_LAMBDA=0
export DEPRO_LAMBDA=0
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0

cd "${REPO}"
checkpoint=$(find "${MODEL_OUTPUT}" \
  -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
  -print | sort | tail -n 1)
if [[ -n "${checkpoint}" ]]; then
  echo "[H0] reuse existing checkpoint ${checkpoint}" | tee -a "${train_log}"
else
  {
    echo "[H0] start tag=${tag}"
    echo "[H0] time=$(date --iso-8601=seconds) physical_gpu=${physical_gpu}"
    echo "[H0] data=A1D_refined_2D TYPE=${TYPE}"
    echo "[H0] protocol=curriculum fixed_views=${fixed_views} epochs=${fixed_epochs} weights=${view_count_weights}"
    echo "[H0] backbone=original_RUMPL only_change=A1D_2D_input_plus_tri_anchor_residual"
  } | tee "${train_log}"

  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" \
    --gpus 0 \
    --workers 12 \
    --validate-on-two-datasets 1 \
    --use-mmpose-val 0 \
    --apply-noise-missing 0 \
    --missing-level 0.0 \
    --exp-name "${tag}" \
    >>"${train_log}" 2>&1

  checkpoint=$(find "${MODEL_OUTPUT}" \
    -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
    -print | sort | tail -n 1)
fi
if [[ -z "${checkpoint}" ]]; then
  echo "[H0] missing checkpoint for ${tag}" | tee -a "${train_log}" >&2
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
    --workers 12 \
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

echo "[H0] end tag=${tag} time=$(date --iso-8601=seconds)" | tee -a "${train_log}"
printf '%s\n' "$(date --iso-8601=seconds)" >"${done_file}"
