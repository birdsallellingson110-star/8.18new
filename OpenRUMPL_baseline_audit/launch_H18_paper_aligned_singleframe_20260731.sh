#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {nodrop|drop20|center_nodrop|center_drop20} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2

REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=/mnt/data/cjyoutput/h36m_paper_repro_20260728/H12_real_h36m_train.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE=mmpose_hrnet_coco_inferencer_legswap

case "${variant}" in
  nodrop)
    family=H18
    code=P0
    token_dropout=0
    tri_anchor=0
    description=harmonic15_noconfconcat_mse_bias
    ;;
  drop20)
    family=H18
    code=P1
    token_dropout=0.2
    tri_anchor=0
    description=harmonic15_noconfconcat_mse_bias_tokendrop20
    ;;
  center_nodrop)
    family=H19
    code=P2
    token_dropout=0
    tri_anchor=1
    description=harmonic15_noconfconcat_mse_bias_tricenter
    ;;
  center_drop20)
    family=H19
    code=P3
    token_dropout=0.2
    tri_anchor=1
    description=harmonic15_noconfconcat_mse_bias_tricenter_tokendrop20
    ;;
  *)
    echo "Unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

if [[ "${family}" == H18 ]]; then
  BASE=/mnt/data/cjyoutput/h36m_gbt_paper_aligned_singleframe_20260731
else
  BASE=/mnt/data/cjyoutput/h36m_gbt_paper_centered_singleframe_20260731
fi
tag="${family}_${code}_${description}_fixed2_balancedPairs_realH36M_seed0_20260731"
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
  echo "[H18] skip completed ${tag}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=0

# Paper-aligned single-frame GBT core.  We deliberately keep temporal,
# synthetic cameras and scene centering disabled for a clean comparison.
export RUMPL_GBT_SET_DECODER=1
export RUMPL_GBT_SET_DEPTH=3
export RUMPL_GBT_SET_DECODER_DEPTH=2
export RUMPL_GBT_SET_PLUCKER=1
export RUMPL_GBT_SET_BIASED=1
export RUMPL_GBT_SET_HARMONIC_L=15
export RUMPL_GBT_SET_NO_CONF_CONCAT=1
export RUMPL_GBT_SET_TOKEN_DROPOUT="${token_dropout}"
export RUMPL_LOSS_TYPE=JointsMSELoss
export GBT_CONF_INIT=0.1
export GBT_GEOM_INIT=1.0

# Disable the old RUMPL bias path and all unrelated experimental branches.
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export RUMPL_TRI_ANCHOR="${tri_anchor}"
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
  echo "[${family}] reuse existing checkpoint ${checkpoint}" | tee -a "${train_log}"
else
  {
    echo "[${family}] start tag=${tag}"
    echo "[${family}] time=$(date --iso-8601=seconds) physical_gpu=${physical_gpu}"
    echo "[${family}] data=real_H36M clean_mmpose missing=0 fixed_views=2 balanced_pairs=1"
    echo "[${family}] encoder=3 decoder=2 plucker=1 harmonic_L=15 heads=8"
    echo "[${family}] confidence=attention_bias_only geometry_bias=1 loss=MSE token_dropout=${token_dropout}"
    echo "[${family}] temporal=0 synthetic_views=0 tri_center=${tri_anchor}"
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
  echo "[${family}] missing checkpoint for ${tag}" | tee -a "${train_log}" >&2
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

echo "[${family}] end tag=${tag} time=$(date --iso-8601=seconds)" | tee -a "${train_log}"
printf '%s\n' "$(date --iso-8601=seconds)" >"${done_file}"
