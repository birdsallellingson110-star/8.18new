#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?usage: VARIANT(ray|control) GPU}
GPU=${2:-0}
case "${VARIANT}" in
  ray) MODE=ray-cross; BATCH=16 ;;
  control) MODE=none; BATCH=64 ;;
  *) echo "variant must be ray or control" >&2; exit 2 ;;
esac

PY=/mnt/data/cjydata/envs/raymixste/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
TEMP=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818
K96=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/k96_temporal_cache
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e5_ray_dual_stream/${VARIANT}_seed0
mkdir -p "${OUT}"

exec "${PY}" -u "${AUDIT}/train_e5_ray_conditioned_dual_stream_20260821.py" \
  --train-cache "${BASE}/train_c2_22c.npz" \
  --train-fused "${K96}/train/fused_poses.npy" \
  --train-pkl /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl \
  --validation-cache "${TEMP}/h15_temporal_c2_oracle/validation_c2_22c.npz" \
  --validation-fused "${K96}/validation/fused_poses.npy" \
  --validation-pkl /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl \
  --output-dir "${OUT}" --window-length 9 --frame-stride 5 \
  --epochs 4 --batch-size "${BATCH}" --hidden-dim 64 --layers 2 --heads 8 \
  --lr 5e-5 --weight-decay 5e-4 --root-mode learned \
  --observation-mode "${MODE}" --gate-mm 0.15 --preload-fused \
  --gpu "${GPU}" --seed 0
