#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {set|plucker|biased|full} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2

REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=/mnt/data/cjyoutput/h36m_paper_repro_20260728/H12_real_h36m_train.yaml
BASE=/mnt/data/cjyoutput/h36m_gbt_set_decoder_20260730
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE=mmpose_hrnet_coco_inferencer_legswap

case "${variant}" in
  set)
    code=S0
    plucker=0
    biased=0
    token_dropout=0
    description=set_encoder_joint_query_decoder
    ;;
  plucker)
    code=S1
    plucker=1
    biased=0
    token_dropout=0
    description=set_decoder_plucker
    ;;
  biased)
    code=S2
    plucker=1
    biased=1
    token_dropout=0
    description=set_decoder_plucker_global_bias
    ;;
  full)
    code=S3
    plucker=1
    biased=1
    token_dropout=0.2
    description=set_decoder_plucker_global_bias_tokendrop20
    ;;
  *)
    echo "Unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

tag="H17_${code}_${description}_fixed2_balancedPairs_realH36M_seed0_20260730"
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
  echo "[H17] skip completed ${tag}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=0

# Single-frame counterpart of GBT: global set encoder over JxV ray tokens,
# followed by a fixed set of 17 learned joint queries.  Its output cardinality
# does not depend on the number of input views.
export RUMPL_GBT_SET_DECODER=1
export RUMPL_GBT_SET_DEPTH=3
export RUMPL_GBT_SET_DECODER_DEPTH=2
export RUMPL_GBT_SET_PLUCKER="${plucker}"
export RUMPL_GBT_SET_BIASED="${biased}"
export RUMPL_GBT_SET_TOKEN_DROPOUT="${token_dropout}"
export GBT_CONF_INIT=0.1
export GBT_GEOM_INIT=1.0

# Disable RUMPL's old per-joint fusion-token bias and every unrelated branch.
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export RUMPL_TRI_ANCHOR=0
export RUMPL_KPA=0
export GBT_TOKEN_DROPOUT=0
export CAA_LAMBDA=0
export DEPRO_LAMBDA=0
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0

cd "${REPO}"
{
  echo "[H17] start tag=${tag}"
  echo "[H17] time=$(date --iso-8601=seconds) physical_gpu=${physical_gpu}"
  echo "[H17] data=real_H36M clean_mmpose missing=0 fixed_views=2 balanced_pairs=1"
  echo "[H17] set_encoder=3 query_decoder=2 plucker=${plucker} global_bias=${biased} token_dropout=${token_dropout}"
  echo "[H17] temporal=0 synthetic_views=0 scene_centering=0"
} | tee "${train_log}"

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

checkpoint=$(find "${MODEL_OUTPUT}" \
  -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
  -print | sort | tail -n 1)
if [[ -z "${checkpoint}" ]]; then
  echo "[H17] missing checkpoint for ${tag}" | tee -a "${train_log}" >&2
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

echo "[H17] end tag=${tag} time=$(date --iso-8601=seconds)" | tee -a "${train_log}"
printf '%s\n' "$(date --iso-8601=seconds)" >"${done_file}"
