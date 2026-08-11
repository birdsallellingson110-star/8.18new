#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=${BASE}/H13_real_h36m_fixed4_nomissing_stepsmatched.yaml
NAME=H13_real_h36m_fixed4_nomissing_stepsmatched_seed0_20260729
LOG=${BASE}/H13_real_h36m_fixed4_nomissing_stepsmatched_train.log

export CUDA_VISIBLE_DEVICES=1
export RUMPL_FIX_SCHEDULER_ORDER=1
export TRAIN_FIXED_NUM_VIEWS=4
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=0

# H13 is the clean fixed-four-view control, without any optional module.
unset RUMPL_TRI_ANCHOR RUMPL_TRI_ANCHOR_REG RUMPL_TRI_ANCHOR_CONF_EPS
unset GBT_LEARNABLE_BIAS GBT_USE_CONF_BIAS GBT_USE_GEOM_BIAS
unset GBT_GLOBAL_JV_DEPTH GBT_GLOBAL_JV_BIASED GBT_GLOBAL_JV_GATED
unset RUMPL_KPA

cd "${REPO}"
exec "${PY}" -u run/train_rumpl.py \
  --cfg "${CFG}" \
  --gpus 0 \
  --workers 16 \
  --validate-on-two-datasets 1 \
  --use-mmpose-val 0 \
  --test-views 1 2 3 4 \
  --apply-noise-missing 0 \
  --exp-name "${NAME}" \
  >"${LOG}" 2>&1

