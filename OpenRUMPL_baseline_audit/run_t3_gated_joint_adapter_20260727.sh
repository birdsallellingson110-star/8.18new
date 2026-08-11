#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU [VARIANT]}
variant=${2:-T3_gated_joint_adapter_r5init_seed0_20260727}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit

export RUMPL_GATED_JOINT_ADAPTER=1
export RUMPL_JOINT_ADAPTER_INDICES=2,5,8
export RUMPL_JOINT_ADAPTER_SCALE_INIT=1.2
export RUMPL_ADAPTER_ONLY=1
export RUMPL_INIT_CHECKPOINT=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar
export RUMPL_ALT_JOINT_VIEW=0 RUMPL_VFT_DEPTH=12 RUMPL_PFT_DEPTH=12
export RUMPL_MULTI_HYP=1

export RUMPL_TRAIN_STRUCT_OCC=0 RUMPL_OCC_JOINT_LOSS=0
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_ADAFUSE_VW=0 RUMPL_2D_REFINE=0
export RUMPL_POSE_CODEBOOK=0 RUMPL_KPA=0 RUMPL_CONF_FILM=0

exec "$repo/run_official_like_cmu_seed0_20260722.sh" \
  "$gpu" "$variant" 0 1 16
