#!/usr/bin/env bash
# H41-H45: controlled paper-grounded view-fusion ablations on the H35 control.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {mtf_point|mtf_mask04|gif_mask02|gif_mask05|gif_mask08} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE=mmpose_hrnet_coco_a1d_h21_legswap
BASE=${ROOT}/H41_H45_paper_mask_fusion

case "${variant}" in
  mtf_point)
    code=H41
    relative=1
    mask_rate=0.0
    mask_diagonal=1
    source_note="MTF official point-mode subtraction attention adapted as a zero-gated RUMPL residual"
    ;;
  mtf_mask04)
    code=H42
    relative=0
    mask_rate=0.4
    mask_diagonal=0
    source_note="MTF public default MASK_RATE=0.4 with protected view self-edges, applied to RUMPL VFT"
    ;;
  gif_mask02)
    code=H43
    relative=0
    mask_rate=0.2
    mask_diagonal=1
    source_note="Masked Gifformer fully-random attention-edge mask M=0.2"
    ;;
  gif_mask05)
    code=H44
    relative=0
    mask_rate=0.5
    mask_diagonal=1
    source_note="Masked Gifformer fully-random attention-edge mask M=0.5 (paper-best robustness ablation)"
    ;;
  gif_mask08)
    code=H45
    relative=0
    mask_rate=0.8
    mask_diagonal=1
    source_note="Masked Gifformer fully-random attention-edge mask M=0.8"
    ;;
  *)
    echo "Unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

tag="${code}_${variant}_H35_A1D_H21_triAnchor_seed0_20260801"
train_log="${BASE}/logs/${tag}_train.log"
eval_root="${BASE}/eval/${tag}"
done_file="${BASE}/completed/${tag}.done"
lock_file="${BASE}/locks/${tag}.lock"
mkdir -p "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/completed" "${BASE}/locks" "${eval_root}"
test -s "${CFG}"
test -s "/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_filtered_5_64_${TYPE}/h36m_train.pkl"

exec 9>"${lock_file}"
flock 9
if [[ -s "${done_file}" ]]; then
  echo "[${code}] skip completed ${tag}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1

# H35 invariants.
export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05

# The only manipulated variables in H41-H45.
export RUMPL_RELATIVE_VIEW_FUSION="${relative}"
export VFT_FULL_RANDOM_MASK="${mask_rate}"
export VFT_MASK_DIAGONAL="${mask_diagonal}"
export VFT_MASK_MIN_VIEWS=2

# Disable every unrelated experimental branch explicitly.
export RUMPL_ANCHOR_CENTERED_RAYS=0
export RUMPL_INPUT_PLUCKER=0
export RUMPL_INPUT_HARMONIC_L=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export RUMPL_GBT_SET_DECODER=0
export RUMPL_GBT_SET_PLUCKER=0
export RUMPL_GBT_SET_HARMONIC_L=0
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0
export RUMPL_SINGLEFRAME_GBT=0
export RUMPL_SF_GBT=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_ALT_JOINT_VIEW=0
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
checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
  -path "*${tag}*/model_best.pth.tar" -print | sort | tail -n 1)
if [[ -z "${checkpoint}" ]]; then
  {
    echo "[${code}] start tag=${tag}"
    echo "[${code}] time=$(date --iso-8601=seconds) physical_gpu=${physical_gpu}"
    echo "[${code}] control=H35 data=A1D-H21 triAnchor=1 curriculum=fixedV2x8_then_3to1to1"
    echo "[${code}] isolated_variable=${source_note}"
    echo "[${code}] relative=${relative} mask_rate=${mask_rate} mask_diagonal=${mask_diagonal}"
    sha256sum "${REPO}/lib/models/multiview_rumpl.py" "${CFG}"
    git -C /mnt/data/cjycode/open_source_fusion_20260731/MTF-Transformer rev-parse HEAD
  } | tee "${train_log}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 6 \
    --validate-on-two-datasets 1 --use-mmpose-val 0 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${tag}" \
    >>"${train_log}" 2>&1
  checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
    -path "*${tag}*/model_best.pth.tar" -print | sort | tail -n 1)
fi
test -n "${checkpoint}"
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
    --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers 6 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-views "${test_views[@]}" --test-on-all-cameras true \
    --test-mmpose-type "${TYPE}" >"${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

echo "[${code}] end tag=${tag} time=$(date --iso-8601=seconds)" | tee -a "${train_log}"
printf '%s\n' "$(date --iso-8601=seconds)" >"${done_file}"
