#!/usr/bin/env bash
# Serialize the new training behind the two-GPU intermediate-checkpoint screen.
set -euo pipefail

SCREEN=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/MixSTE_joint_pft_T9_20260804/eval_fast_intermediate_V4
TRAIN=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_MixSTE_pose_rootprotected_20260808.sh

test -d "${SCREEN}" && test -x "${TRAIN}"
exec 8>"${SCREEN}/pipeline.lock"
flock 8
test -s "${SCREEN}/eval.done"
exec "${TRAIN}"
