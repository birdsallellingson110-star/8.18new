#!/usr/bin/env bash
# H81 on CMU: all V2..V5 camera combinations + mean-over-combos (RUMPL protocol).
set -euo pipefail

physical_gpu=${1:-0}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=${AUDIT}/H81_cmu_eval_h35backbone.yaml
CKPT=$(tr -d '\r\n' < /mnt/data/cjyoutput/open_source_fusion_audit_20260731/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt)
OUT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H81_cmu_matched_joints_eval_20260805

test -s "${CKPT}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${AUDIT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_ANCHOR_CENTER_PER_JOINT=0
export RUMPL_INPUT_PLUCKER=1
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=1
export RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_EVAL_STRICT=0

mkdir -p "${OUT}"

run_vn() {
  local n=$1
  local edir=${OUT}/V${n}_all_combinations
  mkdir -p "${edir}"
  echo "[H81 CMU] V${n} all C(5,${n}) $(date --iso-8601=seconds)" | tee "${edir}/run.log"
  cd "${REPO}"
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" \
    --checkpoint "${CKPT}" \
    --output-dir "${edir}" \
    --workers 8 \
    --gpu 0 \
    --use-mmpose-val true \
    --n-views-combinations "${n}" \
    --all-views-cmu 3 6 12 13 23 \
    --train-views 1 2 3 4 \
    >>"${edir}/eval.log" 2>&1
  local pkl
  pkl=$(find "${edir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${pkl}"
  "${PY}" -u "${AUDIT}/eval_cmu_viewgroups.py" \
    --dict-pkl "${pkl}" \
    --n-views "${n}" \
    --output-json "${edir}/viewgroups.json" | tee "${edir}/summary.log"
}

for n in 2 3 4 5; do
  run_vn "${n}"
done

"${PY}" -u "${AUDIT}/aggregate_h81_cmu_viewgroups.py" \
  --eval-root "${OUT}" \
  --output-json "${OUT}/viewgroups_master.json"

echo "[H81 CMU] done -> ${OUT}"
