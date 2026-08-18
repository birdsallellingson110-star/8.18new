#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
INPUT=${ROOT}/Ada2D_H36M_union20_train_stride20_20260814/h36m_train_stride20.pkl
IMAGES=${DATA}/images
CFG=${AUDIT}/configs/td-hm_res152_8xb32-210e_lt_h36m_384x384_noflip_bgr_17.py
CKPT=${ROOT}/Ada2D_H36M_union20_train_final_20260814/res152_h36m_ep10_h36m17.pth
OUT=${ROOT}/Ada2D_H36M_union20_train_heatmaps_20260814
mkdir -p "${OUT}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

run_shard() {
  local shard="$1" gpu="$2"
  local stem="${OUT}/shard${shard}"
  if [[ -s "${stem}.npz" && -s "${stem}.heatmaps.npy" && -s "${stem}.json" ]]; then
    echo "shard ${shard} already complete"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
    "${AUDIT}/export_h36m_mmpose_heatmap_topk.py" \
    --input-pkl "${INPUT}" --images-root "${IMAGES}" \
    --config "${CFG}" --checkpoint "${CKPT}" \
    --device cuda:0 --shard-id "${shard}" --num-shards 4 \
    --topk 8 --nms-kernel 5 \
    --dense-output "${stem}.heatmaps.npy" --output "${stem}.npz" \
    >"${stem}.log" 2>&1
}

pids=()
for shard in 0 1 2 3; do
  run_shard "${shard}" "$((shard % 2))" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"

