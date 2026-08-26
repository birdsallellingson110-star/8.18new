#!/usr/bin/env bash
# Frozen Stage-1 HRNet/ResNet chains on dense VOC Occ2/Occ3, one variant/GPU.
set -euo pipefail

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
CAMROOT=/mnt/data/cjyoutput/camera_generalization_20260824
ROOT=/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824
RECORD=${ROOT}/canonical_stage2_run_record_20260826_final

test -s "${ROOT}/generation_COMPLETED"
"${PY}" "${AUDIT}/select_final_temporal_clean_holdout_20260825.py" \
  >"${CAMROOT}/final_temporal_selection_20260825.log" 2>&1
test -s "${CAMROOT}/final_temporal_selection_20260825.json"

mkdir -p "${RECORD}"
date --iso-8601=seconds >"${RECORD}/started_at.txt"
cp "${CAMROOT}/final_temporal_selection_20260825.json" \
  "${RECORD}/frozen_temporal_selection.json"
printf '%s\n' \
  'frontends: Occ2 GPU0, Occ3 GPU1, 2 HRNet shards per variant' \
  'evaluation: Occ2 GPU0, Occ3 GPU1; HRNet and ResNet chains concurrent' \
  'frozen Stage1; robust_torso=0; no Stage2 training' \
  >"${RECORD}/launch_environment.txt"
sha256sum \
  "${CAMROOT}/final_temporal_selection_20260825.json" \
  "${AUDIT}/launch_posefusion_occ23_dense_frontends_20260824.sh" \
  "${AUDIT}/launch_stage2_canonical_occ23_final_eval_20260825.sh" \
  >"${RECORD}/launch_inputs.sha256"

SERIAL_VARIANTS=0 OCC2_GPU=0 OCC3_GPU=1 NUM_SHARDS=2 \
  bash "${AUDIT}/launch_posefusion_occ23_dense_frontends_20260824.sh"

STAGE2_OCC2_GPU=0 STAGE2_OCC3_GPU=1 \
  bash "${AUDIT}/launch_stage2_canonical_occ23_final_eval_20260825.sh"

date --iso-8601=seconds >"${RECORD}/completed_at.txt"
echo "[Stage2 frozen dual-GPU] complete"
