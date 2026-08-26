#!/usr/bin/env bash
# Match the old C2 training exposure without blocking the requested E2/H18
# pipeline.  Canonical HRNet has 20 scratch + 20 continuation epochs already;
# this adds 100 epochs for an approximately 140-epoch total versus old C2's
# approximately 143 epochs.
set -euo pipefail

BASE=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair/branches_20260825
for name in \
  cont10_lr3e6_fixed \
  cont10_lr1e6_fixed \
  pelvisprior20_lr1e5_reg1e3 \
  pelvisprior20_lr1e5_reg1e2; do
  while [[ ! -s "${BASE}/${name}/COMPLETED" ]]; do sleep 30; done
done

exec /usr/bin/env \
  REPAIR_NAME=matched_exposure100_lr1e6 \
  REPAIR_LR=1e-6 \
  REPAIR_EPOCHS=100 \
  REPAIR_LR_STEPS=60,85 \
  REPAIR_PELVIS_PRIOR=0 \
  REPAIR_BODY_REG=1e-4 \
  bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_hrnet_canonical_repair_branch_20260825.sh
