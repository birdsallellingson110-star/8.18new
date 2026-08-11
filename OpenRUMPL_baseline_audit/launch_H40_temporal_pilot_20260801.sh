#!/usr/bin/env bash
# H40 convergence pilot: paired temporal joint-view Transformer runs on real H36M.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

physical_gpu=$1
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
OUT=${ROOT}/H40_temporal_gbt_protocol/pilot_200step
CFG=${ROOT}/H0_a1d_refined_rumpl_tri_anchor.yaml
BASE_CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731_2026-08-01_04-13-28/model_best.pth.tar

mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
cd "${REPO}"

pids=()
for variant in unbiased biased; do
  args=()
  if [[ "${variant}" == biased ]]; then
    args+=(--biased)
  fi
  variant_out=${OUT}/${variant}
  mkdir -p "${variant_out}"
  "${PY}" -u run/train_temporal_gbt_rumpl.py \
    --cfg "${CFG}" \
    --base-checkpoint "${BASE_CKPT}" \
    --output-dir "${variant_out}" \
    --window-length 9 --frame-stride 5 --num-views 2 \
    --depth 3 --heads 8 --token-dropout 0.2 \
    --optimizer-steps 200 --warmup-steps 20 \
    --micro-batch-size 1 --effective-batch-size 8 \
    --workers 2 --max-windows 4096 --device cuda:0 \
    --amp-dtype bf16 --log-every 10 --save-every 200 \
    "${args[@]}" \
    >"${variant_out}/train.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${OUT}/completed.lock"
