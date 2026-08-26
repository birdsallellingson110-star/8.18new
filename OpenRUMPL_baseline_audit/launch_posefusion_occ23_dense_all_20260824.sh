#!/usr/bin/env bash
# Resumable end-to-end dense VOC Occ-2/Occ-3 evaluation pipeline.
set -euo pipefail
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
ROOT=/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824
bash "${AUDIT}/launch_posefusion_occ23_dense_generate_20260824.sh"
bash "${AUDIT}/launch_posefusion_occ23_dense_frontends_20260824.sh"
bash "${AUDIT}/launch_posefusion_occ23_dense_final_eval_20260824.sh"
"${PY}" "${AUDIT}/collect_posefusion_occ23_dense_final_table_20260824.py" \
  --output-json "${ROOT}/final_occ23_table.json" \
  --output-md "${ROOT}/final_occ23_table.md"
echo "[dense VOC Occ-2/Occ-3 all stages] complete"
