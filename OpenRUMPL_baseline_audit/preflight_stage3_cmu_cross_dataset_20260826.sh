#!/usr/bin/env bash
set -euo pipefail

# Read-only portability check for Stage-3 CMU training and CMU->H36M testing.
# Usage:
#   bash OpenRUMPL_baseline_audit/preflight_stage3_cmu_cross_dataset_20260826.sh --code-only
#   CMU_RAW_ROOT=/path/to/cmu bash ... --standard5
#   CMU_RAW_ROOT=/path/to/cmu bash ... --gbt31

MODE="${1:---code-only}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUMPL_PYTHON="${RUMPL_PYTHON:-$PROJECT_ROOT/rumpl_venv310/bin/python}"
CMU_RAW_ROOT="${CMU_RAW_ROOT:-/mnt/data/cjydata/cmu_singleperson_real20}"

TRAIN_SEQUENCES=(
  171026_pose1 171026_pose2 171026_pose3
  171204_pose1 171204_pose2 171204_pose3 171204_pose4
)
TEST_SEQUENCES=(171204_pose5 171204_pose6)
STANDARD5=(03 06 12 13 23)
GBT_TEST4=(02 10 13 19)
ALL31=(
  00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
  16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
)

failures=0

pass() {
  printf 'PASS %s\n' "$1"
}

fail() {
  printf 'FAIL %s\n' "$1" >&2
  failures=$((failures + 1))
}

check_file() {
  local path="$1"
  if [[ -s "$path" ]]; then pass "$path"; else fail "$path"; fi
}

check_video() {
  local sequence="$1" camera="$2"
  check_file "$CMU_RAW_ROOT/$sequence/hdVideos/hd_00_${camera}.mp4"
}

printf 'project_root=%s\n' "$PROJECT_ROOT"
printf 'python=%s\n' "$RUMPL_PYTHON"
printf 'cmu_raw_root=%s\n' "$CMU_RAW_ROOT"
printf 'mode=%s\n' "$MODE"

required_code=(
  "$PROJECT_ROOT/requirements.txt"
  "$PROJECT_ROOT/OpenRUMPL/RUMPL/run/train_rumpl.py"
  "$PROJECT_ROOT/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py"
  "$PROJECT_ROOT/OpenRUMPL/RUMPL/lib/models/temporal_gbt_rumpl.py"
  "$PROJECT_ROOT/OpenRUMPL/RUMPL/lib/dataset/multiview_cmu_panoptic_rumpl.py"
  "$PROJECT_ROOT/OpenRUMPL/RUMPL/data/preprocess_cmu_panoptic.py"
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/convert_cmu_coco_to_h36m_virtual_20260824.py"
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/train_e2_camera_independent_22c_20260824.py"
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py"
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/export_cmu_fourview_hypotheses_20260824.py"
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/evaluate_e2_cross_dataset_20260824.py"
)
for path in "${required_code[@]}"; do check_file "$path"; done

if [[ -x "$RUMPL_PYTHON" ]]; then
  "$RUMPL_PYTHON" - <<'PY'
import numpy, torch, yaml
print(f"PASS python imports torch={torch.__version__} numpy={numpy.__version__}")
PY
else
  fail "executable Python environment: $RUMPL_PYTHON"
fi

"${RUMPL_PYTHON:-python}" -m py_compile \
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/convert_cmu_coco_to_h36m_virtual_20260824.py" \
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/export_cmu_fourview_hypotheses_20260824.py" \
  "$PROJECT_ROOT/OpenRUMPL_baseline_audit/evaluate_e2_cross_dataset_20260824.py" \
  "$PROJECT_ROOT/OpenRUMPL/RUMPL/lib/dataset/multiview_cmu_panoptic_rumpl.py"
pass "Stage-3 Python syntax"

case "$MODE" in
  --code-only)
    ;;
  --standard5)
    for sequence in "${TRAIN_SEQUENCES[@]}" "${TEST_SEQUENCES[@]}"; do
      check_file "$CMU_RAW_ROOT/$sequence/calibration_${sequence}.json"
      check_file "$CMU_RAW_ROOT/$sequence/hdPose3d_stage1_coco19.tar"
      for camera in "${STANDARD5[@]}"; do check_video "$sequence" "$camera"; done
    done
    ;;
  --gbt31)
    for sequence in "${TRAIN_SEQUENCES[@]}" "${TEST_SEQUENCES[@]}"; do
      check_file "$CMU_RAW_ROOT/$sequence/calibration_${sequence}.json"
      check_file "$CMU_RAW_ROOT/$sequence/hdPose3d_stage1_coco19.tar"
      for camera in "${ALL31[@]}"; do check_video "$sequence" "$camera"; done
    done
    ;;
  *)
    printf 'Unknown mode: %s\n' "$MODE" >&2
    exit 2
    ;;
esac

if [[ "$MODE" != "--code-only" ]]; then
  printf 'GBT comparison cameras: %s\n' "${GBT_TEST4[*]}"
  printf 'RUMPL standard-five cameras: %s\n' "${STANDARD5[*]}"
fi

if (( failures > 0 )); then
  printf 'PREFLIGHT_FAILED count=%d\n' "$failures" >&2
  exit 1
fi
printf 'PREFLIGHT_OK\n'
