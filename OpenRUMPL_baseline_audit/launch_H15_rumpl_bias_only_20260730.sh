#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {baseline|conf|geom|both} PHYSICAL_GPU" >&2
  exit 2
fi

mode=$1
physical_gpu=$2

REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=/mnt/data/cjyoutput/h36m_paper_repro_20260728/H12_real_h36m_train.yaml
BASE=/mnt/data/cjyoutput/h36m_rumpl_bias_only_20260730
TYPE=mmpose_hrnet_coco_inferencer_legswap

case "${mode}" in
  baseline)
    conf_bias=0
    geom_bias=0
    fusion_geom=0
    label=R0
    ;;
  conf)
    conf_bias=1
    geom_bias=0
    fusion_geom=0
    label=R1
    ;;
  geom)
    conf_bias=0
    geom_bias=1
    fusion_geom=1
    label=R2
    ;;
  both)
    conf_bias=1
    geom_bias=1
    fusion_geom=1
    label=R3
    ;;
  *)
    echo "Unsupported mode: ${mode}" >&2
    exit 2
    ;;
esac

tag="H15_${label}_rumpl_${mode}_clean_realH36M_random2to4_seed0_20260730"
train_log="${BASE}/logs/${tag}_train.log"
eval_root="${BASE}/eval/${tag}"
done_file="${BASE}/completed/${tag}.done"
lock_file="${BASE}/locks/${tag}.lock"

mkdir -p \
  "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/completed" \
  "${BASE}/locks" "${eval_root}"
test -s "${CFG}"

# The original two-GPU queue may reach a mode that was already launched in
# parallel. Serialize by mode and skip it after the first successful run.
exec 9>"${lock_file}"
flock 9
if [[ -s "${done_file}" ]]; then
  echo "[H15] skip completed ${label} mode=${mode}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export RUMPL_FIX_SCHEDULER_ORDER=1

# The original RUMPL backbone remains unchanged. Only the attention-logit
# confidence/ray-distance bias is ablated.
if [[ "${mode}" == "baseline" ]]; then
  export GBT_LEARNABLE_BIAS=0
else
  export GBT_LEARNABLE_BIAS=1
fi
export GBT_USE_CONF_BIAS="${conf_bias}"
export GBT_USE_GEOM_BIAS="${geom_bias}"
export GBT_CONF_INIT=0.1
export GBT_GEOM_INIT=1.0
export GBT_FUSION_GEOM="${fusion_geom}"

# Disable every unrelated optional module and preserve H12 random 2-4-view
# sampling. Missing-joint corruption is disabled explicitly below.
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
unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS

cd "${REPO}"
echo "[H15] start ${label} mode=${mode} gpu=${physical_gpu} conf=${conf_bias} geom=${geom_bias} fusion_geom=${fusion_geom} missing=0 views=random2to4 $(date --iso-8601=seconds)" | tee "${train_log}"
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
  echo "[H15] missing checkpoint for ${tag}" | tee -a "${train_log}" >&2
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

echo "[H15] end ${label} mode=${mode} $(date --iso-8601=seconds)" | tee -a "${train_log}"
printf '%s\n' "$(date --iso-8601=seconds)" >"${done_file}"
