#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
TRAIN=${ROOT}/Ada2D_H36M_union20_train_final_20260814
VAL=${ROOT}/Ada2D_H36M_union20_val_20260814
TRAIN_SUB=${ROOT}/Ada2D_H36M_union20_train_stride20_20260814
CHECKPOINT=${TRAIN}/res152_h36m_ep10.pth
CHECKPOINT17=${TRAIN}/res152_h36m_ep10_h36m17.pth
CONFIG17=${AUDIT}/configs/td-hm_res152_8xb32-210e_lt_h36m_384x384_noflip_bgr_17.py
INPUT=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
TRAIN_INPUT=${DATA}/data/datasets/annot_filtered_5_64/h36m_train.pkl
IMAGES=${DATA}/images

mkdir -p "${VAL}" "${TRAIN_SUB}"
while [[ ! -s "${CHECKPOINT}" ]] || ! grep -q '"epoch_done": 10' "${TRAIN}/train.log"; do
  sleep 30
done
# Let the training parent close its dataloader workers and release both GPUs.
sleep 60

if [[ ! -s "${CHECKPOINT17}" ]]; then
  "${PY}" "${AUDIT}/convert_ada_union20_h36m17_20260814.py" \
    --input "${CHECKPOINT}" --output "${CHECKPOINT17}" \
    >"${VAL}/convert.log" 2>&1
fi

"${PY}" "${AUDIT}/prepare_h36m_group_stride_pkl_20260814.py" \
  --input "${TRAIN_INPUT}" --output "${TRAIN_SUB}/h36m_train_stride20.pkl" \
  --stride 20 >"${TRAIN_SUB}/prepare.log" 2>&1

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
run_shard() {
  local shard="$1" gpu="$2"
  local stem="${VAL}/shard${shard}"
  if [[ -s "${stem}.npz" && -s "${stem}.heatmaps.npy" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u \
    "${AUDIT}/export_h36m_mmpose_heatmap_topk.py" \
    --input-pkl "${INPUT}" --images-root "${IMAGES}" \
    --config "${CONFIG17}" --checkpoint "${CHECKPOINT17}" \
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
