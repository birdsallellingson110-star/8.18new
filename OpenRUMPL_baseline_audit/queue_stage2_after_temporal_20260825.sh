#!/usr/bin/env bash
# Freeze clean temporal checkpoints. Stage-2 is opt-in via RUN_STAGE2=1.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CAMROOT=/mnt/data/cjyoutput/camera_generalization_20260824
HR=${CAMROOT}/hrnet_token10_generalization_20260825/canonical_h18
RN=${CAMROOT}/stage1_h36m_dual_frontend/resnet152/canonical_h18
OCCROOT=/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824
RECORD=${OCCROOT}/canonical_stage2_run_record_20260826

while [[ ! -s "${HR}/UNCERTAINTY_MATCHED_COMPLETED" || \
         ! -s "${RN}/UNCERTAINTY_MATCHED_COMPLETED" ]]; do
  sleep 30
done

"${PY}" "${AUDIT}/select_final_temporal_clean_holdout_20260825.py" \
  >"${CAMROOT}/final_temporal_selection_20260825.log" 2>&1

if [[ "${RUN_STAGE2:-0}" != "1" ]]; then
  echo "[queued Stage-1] final temporal selection complete; Stage-2 disabled"
  exit 0
fi

mkdir -p "${RECORD}"
date --iso-8601=seconds >"${RECORD}/started_at.txt"
cp "${CAMROOT}/final_temporal_selection_20260825.json" \
  "${RECORD}/frozen_temporal_selection.json"
printf '%s\n' \
  'RUN_STAGE2=1 SERIAL_VARIANTS=1 OCC2_GPU=1 OCC3_GPU=1 NUM_SHARDS=4 STAGE2_GPU=1' \
  >"${RECORD}/launch_environment.txt"
sha256sum \
  "${CAMROOT}/final_temporal_selection_20260825.json" \
  "${AUDIT}/queue_stage2_after_temporal_20260825.sh" \
  "${AUDIT}/launch_posefusion_occ23_dense_frontends_20260824.sh" \
  "${AUDIT}/launch_stage2_canonical_occ23_final_eval_20260825.sh" \
  "${AUDIT}/evaluate_frozen_h18_on_occlusion_20260822.py" \
  >"${RECORD}/launch_inputs.sha256"
git -C /home/lixiaob/cjy status --short >"${RECORD}/git_status.txt" 2>&1 || true
git -C /home/lixiaob/cjy rev-parse HEAD >"${RECORD}/git_head.txt" 2>&1 || true
git -C /home/lixiaob/cjy diff -- OpenRUMPL_baseline_audit \
  >"${RECORD}/code_snapshot.diff" 2>&1 || true

SERIAL_VARIANTS=1 OCC2_GPU=1 OCC3_GPU=1 NUM_SHARDS=4 \
  bash "${AUDIT}/launch_posefusion_occ23_dense_frontends_20260824.sh"

STAGE2_GPU=1 bash "${AUDIT}/launch_stage2_canonical_occ23_final_eval_20260825.sh"
date --iso-8601=seconds >"${RECORD}/completed_at.txt"
echo "[queued Stage-2] complete"
