#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=${BASE}/H12_real_h36m_train.yaml
NAME=H12_real_h36m_rumpl_nomodule_fullimageval_seed0_20260729
LOG=${BASE}/H12_real_h36m_rumpl_nomodule_fullimageval_seed0_train.log

# Physical GPU 1 is exposed as logical GPU 0 to the training process.
export CUDA_VISIBLE_DEVICES=1
export RUMPL_FIX_SCHEDULER_ORDER=1

# Keep this run as the unmodified RUMPL control.  These variables are used by
# prior optional-module experiments and must not leak through a parent shell.
unset GBT_LEARNABLE_BIAS GBT_USE_CONF_BIAS GBT_USE_GEOM_BIAS
unset GBT_GLOBAL_JV_DEPTH GBT_GLOBAL_JV_BIASED GBT_GLOBAL_JV_GATED
unset RUMPL_TRI_ANCHOR RUMPL_KPA

cd "${REPO}"
exec "${PY}" -u run/train_rumpl.py \
  --cfg "${CFG}" \
  --gpus 0 \
  --workers 16 \
  --validate-on-two-datasets 1 \
  --use-mmpose-val 0 \
  --exp-name "${NAME}" \
  >"${LOG}" 2>&1
