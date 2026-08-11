#!/usr/bin/env bash
# H32: original RUMPL + triangulation residual + GBT conf/geom attention bias.
# Keeps RUMPL_GBT_SET_DECODER=0 (do NOT replace the backbone).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {a1d|original} PHYSICAL_GPU [bias_mode=both|conf|geom]" >&2
  exit 2
fi

data_mode=$1
physical_gpu=$2
bias_mode=${3:-both}

REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=${ROOT}/H32_tri_anchor_gbt_bias

case "${data_mode}" in
  a1d)
    CFG=${ROOT}/H0_a1d_refined_rumpl_tri_anchor.yaml
    TYPE=mmpose_hrnet_coco_a1d_legswap
    data_tag=a1dRefined2D
    ;;
  original)
    CFG=/mnt/data/cjyoutput/h36m_paper_repro_20260728/H12_real_h36m_train.yaml
    TYPE=mmpose_hrnet_coco_inferencer_legswap
    data_tag=original2D
    ;;
  *)
    echo "Unsupported data_mode: ${data_mode}" >&2
    exit 2
    ;;
esac

case "${bias_mode}" in
  both) conf_bias=1; geom_bias=1; fusion_geom=1; bias_tag=confGeom ;;
  conf) conf_bias=1; geom_bias=0; fusion_geom=0; bias_tag=confOnly ;;
  geom) conf_bias=0; geom_bias=1; fusion_geom=1; bias_tag=geomOnly ;;
  *) echo "Unsupported bias_mode: ${bias_mode}" >&2; exit 2 ;;
esac

# Same curriculum as H22/H0: fixed-2 for 8 epochs, then V2-heavy 3:1:1.
fixed_views=2
fixed_epochs=8
view_count_weights=3,1,1
tag="H32_${data_tag}_${bias_tag}_triAnchor_curriculum_seed0_20260731"
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
  echo "[H32] skip completed ${tag}"
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

export GBT_LEARNABLE_BIAS=1
export GBT_USE_CONF_BIAS="${conf_bias}"
export GBT_USE_GEOM_BIAS="${geom_bias}"
export GBT_CONF_INIT=0.1
export GBT_GEOM_INIT=1.0
export GBT_FUSION_GEOM="${fusion_geom}"

export RUMPL_GBT_SET_DECODER=0
export RUMPL_SINGLEFRAME_GBT=0
export RUMPL_SF_GBT=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_ALT_JOINT_VIEW=0
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
  echo "[H32] reuse existing checkpoint ${checkpoint}" | tee -a "${train_log}"
else
  {
    echo "[H32] start tag=${tag}"
    echo "[H32] time=$(date --iso-8601=seconds) gpu=${physical_gpu}"
    echo "[H32] TARGETS V2<40 V4<30 | data=${data_mode} bias=${bias_mode}"
    echo "[H32] backbone=RUMPL+triAnchor+GBT_bias set_decoder=0"
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
  echo "[H32] missing checkpoint for ${tag}" | tee -a "${train_log}" >&2
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

  # For original data mode, train yaml still uses legswap train / inferencer val.
  # For a1d mode, CFG already points both TRAIN/TEST to a1d_legswap.
  test_type="${TYPE}"
  if [[ "${data_mode}" == "original" ]]; then
    test_type=mmpose_hrnet_coco_inferencer_legswap
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
    --test-mmpose-type "${test_type}" \
    >"${eval_dir}/eval.log" 2>&1

  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py \
    --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" \
    >"${eval_dir}/table2.log" 2>&1
done

echo "[H32] end tag=${tag} time=$(date --iso-8601=seconds)" | tee -a "${train_log}"
printf '%s\n' "$(date --iso-8601=seconds)" >"${done_file}"
