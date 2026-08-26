#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/data/cjydata/envs/raymixste/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820
BASE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2
OUT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/k96_temporal_cache
COMMON=(
  --train-cache "${BASE}/train_c2_22c.npz"
  --e2-checkpoint "${ROOT}/e2_c2_unbiased_scorer/seed0/model_best.pth.tar"
  --proposal-checkpoint "${ROOT}/e2_limb_utility/seed0/model_best.pth.tar"
  --k96-checkpoint "${ROOT}/e2_pose_dsac_limb_proposal/seed0_30e_tmux/model_best.pth.tar"
  --batch-size 192 --seed 0 --gpu 0
)
mkdir -p "${OUT}/train" "${OUT}/validation"
"${PY}" -u "${AUDIT}/build_k96_temporal_anchor_cache_20260821.py" \
  --cache "${BASE}/train_c2_22c.npz" --output-dir "${OUT}/train" \
  "${COMMON[@]}"
"${PY}" -u "${AUDIT}/build_k96_temporal_anchor_cache_20260821.py" \
  --cache /mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h15_temporal_c2_oracle/validation_c2_22c.npz \
  --output-dir "${OUT}/validation" "${COMMON[@]}"
