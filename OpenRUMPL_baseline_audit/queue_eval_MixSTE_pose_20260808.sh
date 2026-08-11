#!/usr/bin/env bash
# Start matched P1/P2/M0 evaluation immediately after both training arms finish.
set -euo pipefail

RUN=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/MixSTE_pose_rootprotected_20260808
EVAL=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/eval_MixSTE_pose_rootprotected_20260808.sh

test -d "${RUN}" && test -x "${EVAL}"
exec 8>"${RUN}/pipeline.lock"
flock 8
test -s "${RUN}/training.done"
exec "${EVAL}"
