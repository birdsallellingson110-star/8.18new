#!/usr/bin/env bash
# GBT-aligned CMU cross-dataset eval (best effort with local data).
# Paper: H36M train -> CMU test, absolute All-17 MPJPE (mm), 4 fixed HD cams, T=9.
# Local PKL only has cams 3,6,12,13,23 — proxy 4-view = 3,6,13,23 (includes 13; no 2,10,19).
set -euo pipefail

physical_gpu=${1:-0}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=${AUDIT}/H81_cmu_eval_h35backbone.yaml
CKPT=$(tr -d '\r\n' < /mnt/data/cjyoutput/open_source_fusion_audit_20260731/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt)
OUT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H81_gbt_aligned_cmu_eval_20260805

test -s "${CKPT}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4 RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1 RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_PER_JOINT_RESIDUAL_GATE=1 RUMPL_EVAL_STRICT=0

mkdir -p "${OUT}"

run_fixed() {
  local tag=$1
  shift
  local views=("$@")
  local edir=${OUT}/${tag}
  mkdir -p "${edir}"
  cd "${REPO}"
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" --checkpoint "${CKPT}" --output-dir "${edir}" \
    --workers 8 --gpu 0 --use-mmpose-val true \
    --test-on-all-cameras false --test-views "${views[@]}" --train-views 1 2 3 4 \
    >>"${edir}/eval.log" 2>&1
  pkl=$(find "${edir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  "${PY}" -u "${AUDIT}/eval_gbt_aligned_cmu_metrics.py" \
    --dict-pkl "${pkl}" --tag "${tag}" \
    --output-json "${edir}/gbt_aligned.json" | tee "${edir}/metrics.log"
}

# Proxy for GBT 4 HD test cameras; paper uses (2,13,10,19).
run_fixed gbt_proxy_4cam_3_6_13_23 3 6 13 23
run_fixed rumpl_5cam_standard 3 6 12 13 23
run_fixed gbt_like_4cam_3_6_12_13 3 6 12 13

"${PY}" -u "${AUDIT}/diagnose_h81_cmu_gbt_gap.py" \
  --eval-root "${OUT}" \
  --viewgroups-root /mnt/data/cjyoutput/open_source_fusion_audit_20260731/H81_cmu_matched_joints_eval_20260805 \
  --output-json "${OUT}/gap_analysis.json" | tee "${OUT}/gap_analysis.log"

echo "done -> ${OUT}"
