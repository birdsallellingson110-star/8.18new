#!/usr/bin/env bash
# Run the real-H36M end-to-end temporal smoke only after current GPU1 work exits.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
OUT=${ROOT}/H40_temporal_gbt_protocol/smoke
CFG=${ROOT}/H0_a1d_refined_rumpl_tri_anchor.yaml
BASE_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731_2026-08-01_04-13-28/model_best.pth.tar
LOG=${ROOT}/H40_temporal_smoke_queue.log

mkdir -p "${OUT}"
exec >>"${LOG}" 2>&1
echo "[H40 smoke queue] waiting $(date --iso-8601=seconds)"

while pgrep -f 'H37_J[01]_globalJV|H39_U0_undistortPoints|export_h21_refined_mmpose_pkl.py' >/dev/null; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES=1
cd "${REPO}"
for variant in unbiased biased; do
  args=()
  if [[ "${variant}" == biased ]]; then
    args+=(--biased)
  fi
  variant_out=${OUT}/${variant}
  mkdir -p "${variant_out}"
  echo "[H40 smoke] start variant=${variant} $(date --iso-8601=seconds)"
  "${PY}" -u run/train_temporal_gbt_rumpl.py \
    --cfg "${CFG}" \
    --base-checkpoint "${BASE_CKPT}" \
    --output-dir "${variant_out}" \
    --window-length 9 --frame-stride 5 --num-views 2 \
    --depth 3 --heads 8 --token-dropout 0.2 \
    --optimizer-steps 2 --warmup-steps 1 \
    --micro-batch-size 1 --effective-batch-size 1 \
    --workers 0 --max-windows 64 --device cuda:0 \
    --log-every 1 --save-every 2 \
    "${args[@]}" \
    >"${variant_out}/train.log" 2>&1
  echo "[H40 smoke] end variant=${variant} $(date --iso-8601=seconds)"
done

touch "${OUT}/completed.lock"
echo "[H40 smoke queue] complete $(date --iso-8601=seconds)"
