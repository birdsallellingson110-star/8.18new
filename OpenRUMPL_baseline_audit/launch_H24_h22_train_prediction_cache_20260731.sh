#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
PER_ACTION=${2:-300}
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CFG=/mnt/data/cjyoutput/h36m_paper_repro_20260728/H12_real_h36m_train.yaml
CHECKPOINT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H20_H22_CUR_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_clean_realH36M_seed0_20260731_2026-07-31_13-00-30/model_best.pth.tar
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H24_h22_train_predictions

mkdir -p "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export RUMPL_TRI_ANCHOR=1
export RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05

cd "${REPO}"
for views in 2 3 4; do
  output="${ROOT}/V${views}"
  mkdir -p "${output}"
  if [[ "${views}" -eq 2 ]]; then
    selected_views=(1 2)
  elif [[ "${views}" -eq 3 ]]; then
    selected_views=(1 2 3)
  else
    selected_views=(1 2 3 4)
  fi
  if [[ -s "${output}/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl" \
        && -s "${output}/groups.json" ]]; then
    echo "[H24-cache] skip V${views}: complete"
    continue
  fi
  echo "[H24-cache] start V${views} $(date --iso-8601=seconds)"
  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${output}" \
    --workers 12 \
    --gpu 0 \
    --use-mmpose-val true \
    --flip-lower-body-kp-test false \
    --test-subset train \
    --test-views "${selected_views[@]}" \
    --test-on-all-cameras true \
    --test-mmpose-type mmpose_hrnet_coco_legswap \
    --sample-groups-per-action "${PER_ACTION}" \
    --sample-seed 24 \
    --selection-manifest "${output}/groups.json" \
    >"${output}/eval.log" 2>&1
  echo "[H24-cache] end V${views} $(date --iso-8601=seconds)"
done

echo "[H24-cache] complete $(date --iso-8601=seconds)"
