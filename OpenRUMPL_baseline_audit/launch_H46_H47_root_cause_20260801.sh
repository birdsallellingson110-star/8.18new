#!/usr/bin/env bash
# H46/H47: baseline reproducibility and the public RUMPL PFT repeat-last quirk.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {exact_replay|pft_single_pass} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
TYPE=mmpose_hrnet_coco_a1d_h21_legswap
BASE=${ROOT}/H46_H48_root_cause

case "${variant}" in
  exact_replay)
    code=H46
    repeat_last=1
    tag=H46_H35_exact_replay_seed0_20260801
    note="exact H35 replay after refactoring the public repeat-last loop without changing semantics"
    ;;
  pft_single_pass)
    code=H47
    repeat_last=0
    tag=H47_H35_PFT_single_pass_seed0_20260801
    note="execute every PFT block once instead of repeating the final public RUMPL block"
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
mkdir -p "${BASE}/logs" "${BASE}/checkpoints" "${BASE}/completed" "${BASE}/locks" "${eval_root}"
test -s "${CFG}"

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
export RUMPL_PFT_REPEAT_LAST="${repeat_last}"

# Disable every unrelated branch.
export RUMPL_RELATIVE_VIEW_FUSION=0
export VFT_FULL_RANDOM_MASK=0
export RUMPL_ANCHOR_CENTERED_RAYS=0
export RUMPL_INPUT_PLUCKER=0
export RUMPL_INPUT_HARMONIC_L=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export RUMPL_GBT_SET_DECODER=0
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_FUSION_GEOM=0
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
    echo "[${code}] start tag=${tag} time=$(date --iso-8601=seconds)"
    echo "[${code}] control=H35 isolated_variable=${note}"
    echo "[${code}] RUMPL_PFT_REPEAT_LAST=${repeat_last}"
    sha256sum "${REPO}/lib/models/multiview_rumpl.py" "${CFG}"
  } | tee "${train_log}"
  "${PY}" -u run/train_rumpl.py \
    --cfg "${CFG}" --gpus 0 --workers 6 --seed 0 \
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
  RUMPL_PFT_REPEAT_LAST="${repeat_last}" "${PY}" -u run/eval_rumpl_checkpoint.py \
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
date --iso-8601=seconds >"${done_file}"
