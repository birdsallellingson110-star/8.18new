#!/usr/bin/env bash
# Audit whether 2-view-only model_best selection hides a better final epoch.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {h50|h54|h59} PHYSICAL_GPU" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_OUTPUT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=${ROOT}/H64_H66_final_state_audit

case "${variant}" in
  h50)
    code=H64
    source_tag=H50_A1DmatchedH21_workers12_seed0_20260802
    type=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
    ;;
  h54)
    code=H65
    source_tag=H54_A1DmatchedH21_workers12_seed2_20260802
    type=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
    ;;
  h59)
    code=H66
    source_tag=H59_H58balancedH21_RUMPL_workers12_seed0_20260802
    type=mmpose_hrnet_coco_a1d_h21_h58_balanced_views_legswap
    ;;
  *) echo "unsupported variant: ${variant}" >&2; exit 2 ;;
esac

tag=${code}_${source_tag}_finalState_20260803
eval_root=${BASE}/eval/${tag}
done_file=${BASE}/completed/${tag}.done
mkdir -p "${eval_root}" "${BASE}/completed" "${BASE}/checkpoints"
if [[ -s "${done_file}" ]]; then
  echo "[${code}] skip completed"
  exit 0
fi

checkpoint=$(find "${MODEL_OUTPUT}" -maxdepth 2 -type f \
  -path "*${source_tag}*/final_state.pth.tar" -print | sort | tail -n 1)
test -n "${checkpoint}"
printf '%s\n' "${checkpoint}" >"${BASE}/checkpoints/${tag}.txt"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_RELATIVE_VIEW_FUSION=0 VFT_FULL_RANDOM_MASK=0
export RUMPL_ANCHOR_CENTERED_RAYS=0 RUMPL_INPUT_PLUCKER=0 RUMPL_INPUT_HARMONIC_L=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export RUMPL_GBT_SET_DECODER=0 GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0

cd "${REPO}"
for n_views in 2 3 4; do
  eval_dir=${eval_root}/V${n_views}
  mkdir -p "${eval_dir}"
  if [[ "${n_views}" -eq 2 ]]; then test_views=(1 2)
  elif [[ "${n_views}" -eq 3 ]]; then test_views=(1 2 3)
  else test_views=(1 2 3 4)
  fi
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers 12 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
    --test-views "${test_views[@]}" --test-on-all-cameras true \
    --test-mmpose-type "${type}" >"${eval_dir}/eval.log" 2>&1
  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds >"${done_file}"
echo "[${code}] completed"
