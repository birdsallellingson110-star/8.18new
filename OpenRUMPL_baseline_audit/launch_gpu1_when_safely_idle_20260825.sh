#!/usr/bin/env bash
# Never preempt the existing GPU1 owner.  Start the matched-exposure
# pelvis-prior branch only after GPU1 has had no compute process and low memory
# use for ten consecutive 30-second checks (five minutes).
set -euo pipefail

GPU_UUID=GPU-61895d5d-bc22-342d-4c8a-e26906fcc4ab
CHECK_INTERVAL=30
CHECKS_REQUIRED=10
IDLE_MEMORY_MIB=1500
BASE=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair
SELECTION=${BASE}/best_available_modules_20260825/selection.json
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python

idle_checks=0
while (( idle_checks < CHECKS_REQUIRED )); do
  compute_count=$(nvidia-smi \
    --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | awk -v uuid="${GPU_UUID}" '$1 == uuid {count += 1} END {print count + 0}')
  memory_used=$(nvidia-smi --id="${GPU_UUID}" \
    --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d ' ')
  if (( compute_count == 0 && memory_used < IDLE_MEMORY_MIB )); then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  printf '%s gpu1_compute=%d memory_mib=%d idle_checks=%d/%d\n' \
    "$(date --iso-8601=seconds)" "${compute_count}" "${memory_used}" \
    "${idle_checks}" "${CHECKS_REQUIRED}"
  if (( idle_checks < CHECKS_REQUIRED )); then sleep "${CHECK_INTERVAL}"; fi
done

test -s "${SELECTION}"
checkpoint=$("${PY}" - "${SELECTION}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
selected = payload["selected"]
if selected["name"] != "pelvisprior20_lr1e5_reg1e2":
    raise RuntimeError(
        "GPU1 matched-exposure queue expected the frozen pelvis-prior 1e-2 checkpoint"
    )
print(selected["checkpoint"])
PY
)
test -s "${checkpoint}"

echo "$(date --iso-8601=seconds) GPU1 safely idle; starting matched exposure"
exec /usr/bin/env \
  REPAIR_NAME=pelvisprior_matched_exposure80_lr1e6_reg1e2_gpu1 \
  REPAIR_LR=1e-6 \
  REPAIR_EPOCHS=80 \
  REPAIR_LR_STEPS=50,70 \
  REPAIR_PELVIS_PRIOR=1 \
  REPAIR_BODY_REG=1e-2 \
  REPAIR_VISIBLE_GPU=1 \
  REPAIR_INIT="${checkpoint}" \
  bash /home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_hrnet_canonical_repair_branch_20260825.sh
